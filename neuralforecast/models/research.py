"""Official-source integrations: DAG, KITE, APT, GLAFF, TGForecaster and SpecTF.

Supply an explicit trusted `source_dir` prepared with
`scripts/prepare_research_sources.py`. Architectures execute the upstream code;
this module maps NeuralForecast windows and training objectives to their APIs.
See docs/research_models.md for each model's distinct exogenous input schema.
"""

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from ..losses.pytorch import MSE
from ._exogenous import ExogenousModel
from ._official_source import source_module

__all__ = ["DAG", "KITE", "APT", "GLAFF", "TGForecaster", "SpecTF"]


def _positive(**values):
    for key, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{key} must be a positive integer.")


def _attention(d_model, n_heads, dropout):
    _positive(d_model=d_model, n_heads=n_heads)
    if d_model % n_heads or not 0 <= dropout < 1:
        raise ValueError("d_model must be divisible by n_heads; dropout must be in [0, 1).")


class _SourceModel(ExogenousModel):
    def __init__(self, h, input_size, source_dir, **kwargs):
        source = Path(source_dir).expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"Missing official checkout: {source}")
        if kwargs.get("scaler_type", "identity") != "identity":
            raise ValueError("Use scaler_type='identity'; upstream models handle their own scaling.")
        kwargs.setdefault("training_data_availability_threshold", [1.0, 1.0])
        super().__init__(h=h, input_size=input_size, **kwargs)
        self.source_dir = str(source)
        self.hparams["source_dir"] = self.source_dir
        self._auxiliary_loss = None

    def _module(self, name):
        return source_module(self.source_dir, name)

    def _observed(self, batch):
        y, mask, hist, futr = self._inputs(batch)
        self._complete_history(mask)
        return y, hist, futr

    def training_step(self, batch, batch_idx):
        if "sample_weight" in batch["temporal_cols"]:
            raise ValueError("Upstream auxiliary objectives do not support sample_weight.")
        self._auxiliary_loss = None
        primary = super().training_step(batch, batch_idx)
        auxiliary, self._auxiliary_loss = self._auxiliary_loss, None
        if auxiliary is None:
            return primary
        total = primary + auxiliary
        if not torch.isfinite(total):
            raise ValueError("Non-finite upstream training objective.")
        self.log("train_auxiliary_loss", auxiliary.detach(), on_step=True, on_epoch=False)
        self.log("train_total_loss", total.detach(), on_step=True, on_epoch=False)
        return total

    def _objective_batch(self, batch):
        """Reuse NF's window selection/scaler; expose future y only to a train objective."""
        if "sample_weight" in batch["temporal_cols"]:
            raise ValueError("The upstream objective does not support sample_weight.")
        temporal, static, static_cols, available, weights, columns = self._create_windows(batch, step="train")
        available = self._shard_multivariate_windows(available)
        count = len(available)
        if count == 0:
            raise ValueError("No complete windows for the upstream training objective.")
        size = self.windows_batch_size
        indices = None
        if size is not None:
            indices = (torch.randint(count, (size,)) if count < size
                       else torch.randperm(count)[:size])
        windows = self._sample_windows(
            windows_temporal=temporal, static=static, static_cols=static_cols,
            temporal_cols=columns, w_idxs=indices, final_condition=available,
            sample_weight=weights,
        )
        windows = self._normalization(windows=windows, y_idx=batch["y_idx"])
        y, mask, target, target_mask, hist, futr, stat = self._parse_windows(batch, windows)
        if not mask.bool().all() or not target_mask.bool().all():
            raise ValueError("Training requires complete past and future target windows.")
        if not torch.isfinite(target).all():
            raise ValueError("Training targets must be finite.")
        inputs = dict(insample_y=y, insample_mask=mask, hist_exog=hist,
                      futr_exog=futr, stat_exog=stat)
        return inputs, target

    def _log_objective(self, value, name):
        if value.ndim != 0 or not torch.isfinite(value):
            raise ValueError("Upstream objective must be a finite scalar.")
        self.log("train_loss", value, on_step=True, on_epoch=False, prog_bar=True)
        self.log(name, value.detach(), on_step=True, on_epoch=False)
        self.train_trajectories.append((self.global_step, value.detach().item()))
        return value


class DAG(_SourceModel):
    """Official dual-correlation DAG, with both branches and auxiliary losses.

    `futr_exog_list` must contain at least one numerical covariate known over
    both history and horizon. Past-only, static and categorical columns are
    rejected. This model's name does not imply identified causal effects.
    """
    EXOGENOUS_HIST = False

    def __init__(self, h, input_size, source_dir, d_model=128, d_ff=256,
                 n_heads=4, e_layers=2, patch_len=8, stride=4, dropout=0.1,
                 alpha=0.2, beta=0.1, **kwargs):
        _attention(d_model, n_heads, dropout)
        _positive(d_ff=d_ff, e_layers=e_layers, patch_len=patch_len, stride=stride)
        if input_size < patch_len or not 0 <= alpha <= 1 or beta < 0:
            raise ValueError("Require input_size>=patch_len, alpha in [0,1], beta>=0.")
        super().__init__(h, input_size, source_dir, **kwargs)
        if not self.futr_exog_size:
            raise ValueError("DAG requires futr_exog_list.")
        config = SimpleNamespace(
            seq_len=input_size, pred_len=h, patch_len=patch_len, stride=stride,
            use_c=True, use_t=True, use_c_exog=True, use_t_exog=True,
            alpha=alpha, beta=beta, series_dim=1, infer_use_future=True,
            enc_in=1 + self.futr_exog_size, d_model=d_model, d_ff=d_ff,
            n_heads=n_heads, e_layers=e_layers, dropout=dropout, factor=1,
            activation="gelu", criterion=nn.MSELoss() if isinstance(self.loss, MSE) else nn.L1Loss(),
        )
        self.model = self._module("ts_benchmark.baselines.dag.models.dag_model").DAGModel(config)

    def forward(self, windows_batch):
        y, _, futr = self._observed(windows_batch)
        output, auxiliary = self.model(torch.cat((y, futr[:, :self.input_size]), dim=-1),
                                       futr[:, self.input_size:])
        if self.training:
            self._auxiliary_loss = auxiliary
        return self._point_output(output, y)


class KITE(_SourceModel):
    """Official KITE flow-matching training and conditional sampling.

    Use only past covariates OR only known-future covariates, not both. In the
    latter case the same variables' history and future condition the flow.
    NF MAE/MSE selects point-forecast validation; training uses the upstream
    flow-matching objective (including its auxiliary loss), never sampled MSE.
    """
    def __init__(self, h, input_size, source_dir, flow_dim=128, flow_depth=2,
                 flow_head=4, num_sampling_steps=10, num_samples=50, rank=8,
                 min_sigma=0.15, omega=1.0, p_uncond=0.1, aux_loss_weight=0.1,
                 **kwargs):
        _attention(flow_dim, flow_head, 0)
        _positive(flow_depth=flow_depth, num_sampling_steps=num_sampling_steps,
                  num_samples=num_samples, rank=rank)
        if h < 2 or min_sigma <= 0 or not 0 <= p_uncond < 1 or aux_loss_weight < 0 or omega < 0:
            raise ValueError("KITE requires h>=2 and valid distribution/guidance parameters.")
        super().__init__(h, input_size, source_dir, **kwargs)
        if bool(self.hist_exog_size) == bool(self.futr_exog_size):
            raise ValueError("KITE requires exactly one of hist_exog_list or futr_exog_list.")
        self.num_samples = num_samples
        config = SimpleNamespace(
            seq_len=input_size, horizon=h, flow_dim=flow_dim, flow_depth=flow_depth,
            flow_head=flow_head, num_sampling_steps=num_sampling_steps,
            omega=omega, noise_dropout=0.1,
            input_dim=1 + self.hist_exog_size + self.futr_exog_size, output_dim=1,
            p_uncond=p_uncond, structure_max=0.9, rank=rank, min_sigma=min_sigma,
            fc_type="Linear", rate=12, use_future_exog=bool(self.futr_exog_size),
            agg_method="mean", aux_loss_weight=aux_loss_weight, mlp_ratio=4.0,
            prior_level="sample",
        )
        self.model = self._module("ts_benchmark.baselines.kite.models.KITEModel").KITEModel(config)

    def _condition(self, windows_batch):
        y, hist, futr = self._observed(windows_batch)
        return y, (hist if futr is None else futr[:, :self.input_size]), (
            None if futr is None else futr[:, self.input_size:])

    def training_step(self, batch, batch_idx):
        windows, target = self._objective_batch(batch)
        y, past, future = self._condition(windows)
        if future is None:
            # The upstream unconditional-dropout branch calls zeros_like(y_exo).
            # With use_future_exog=False this placeholder is never read by its net.
            future = y.new_zeros(y.shape[0], self.h, past.shape[-1])
        return self._log_objective(self.model.train_function(y, past, target, future), "train_flow_loss")

    def forward(self, windows_batch):
        y, past, future = self._condition(windows_batch)
        return self._point_output(self.model.inference(y, past, future, num_samples=self.num_samples), y)

    def predict(self, *args, **kwargs):
        if kwargs.get("explainer_config") is not None:
            raise ValueError("KITE sampling does not support gradient explanations.")
        return super().predict(*args, **kwargs)


class APT(_SourceModel):
    """Official APT timestamp/prototype normalization composed with DLinear.

    Exactly two known-future columns: time_of_day and day_of_week, encoded as
    integer code/cardinality - 0.5. This is a calendar-conditioned model, NOT
    a model for arbitrary price/weather covariates. Point loss plus optional
    official prototype regularizers; RevIN uses the official implementation.
    """
    EXOGENOUS_HIST = False

    def __init__(self, h, input_size, source_dir, time_of_day_size=24,
                 timestamp_dim=20, timestamp_hidden=32, num_prototypes=20,
                 top_k=5, orthogonality_weight=0.01, balance_weight=0.01, **kwargs):
        _positive(time_of_day_size=time_of_day_size, timestamp_dim=timestamp_dim,
                  timestamp_hidden=timestamp_hidden, num_prototypes=num_prototypes, top_k=top_k)
        if top_k > num_prototypes or num_prototypes < 2 or orthogonality_weight < 0 or balance_weight < 0:
            raise ValueError("Invalid prototype count, top_k or regularizer weights.")
        super().__init__(h, input_size, source_dir, **kwargs)
        if self.futr_exog_size != 2:
            raise ValueError("APT requires two futr_exog columns: normalized time_of_day, day_of_week.")
        self.time_of_day_size = time_of_day_size
        self.orthogonality_weight, self.balance_weight = orthogonality_weight, balance_weight
        args = dict(tan_timestamp=["time_of_day", "day_of_week"],
                    timestamp_dim=timestamp_dim, timestamp_hidden=timestamp_hidden,
                    num_prototypes=num_prototypes, is_xformer=False,
                    time_of_day_size=time_of_day_size, day_of_week_size=7,
                    enc_in=1, top_k=top_k, model_name="DLinear", normalization_name="RevIN",
                    datasets_name="NeuralForecast", use_tan=True, independent=False,
                    seq_len=input_size, pred_len=h, individual=False)
        root = "baselines.Normalization."
        self.timestamp = self._module(root + "normalization.APT").APT(**args)
        self.backbone = self._module(root + "arch.dlinear.dlinear_arch").DLinear(**args)
        self.normalization = self._module(root + "normalization.Revin").RevIN(**args)
        self._orthogonality = self._module(root + "loss.orthogonality_loss").orthogonality
        self._balance = self._module(root + "loss.balanced_loss").balance_loss

    def forward(self, windows_batch):
        y, _, futr = self._observed(windows_batch)
        sizes = futr.new_tensor([self.time_of_day_size, 7])
        codes = (futr + 0.5) * sizes
        if ((codes - codes.round()).abs() > 1e-4).any() or (codes < -1e-4).any() or (codes.round() >= sizes).any():
            raise ValueError("APT calendar columns must encode valid integer code/cardinality - 0.5.")
        normalized = self.normalization(y, "norm")
        scale, offset = self.timestamp(futr[:, :self.input_size], futr[:, self.input_size:], self.training)
        if not torch.isfinite(scale).all() or (scale.abs() < 1e-6).any():
            raise ValueError("APT generated a singular affine scale; change initialization/learning rate.")
        # Official composition; no future target is passed to the DLinear backbone.
        output = self.backbone(normalized * scale + offset, None, 0, 0, self.training)
        output = self.normalization((output - offset) / scale, "denorm")
        if self.training:
            self._auxiliary_loss = (
                self.orthogonality_weight * self._orthogonality(self.timestamp.get_combined_embeddings())
                + self.balance_weight * self._balance(self.timestamp.get_load())
            )
        return self._point_output(output, y)


class GLAFF(_SourceModel):
    """Official GLAFF Plugin + official DLinear, using six calendar covariates.

    The six known-future timestamp features must follow the upstream data loader's
    order/scaling, described in docs/research_models.md. Not arbitrary numeric
    regressors. NF scaling is disabled to preserve this calendar representation.
    """
    EXOGENOUS_HIST = False

    def __init__(self, h, input_size, source_dir, d_model=128, d_ff=256,
                 n_heads=4, e_layers=2, dropout=0.1, q=0.75, **kwargs):
        _attention(d_model, n_heads, dropout)
        _positive(d_ff=d_ff, e_layers=e_layers)
        if not 0.5 < q < 1:
            raise ValueError("q must be strictly between 0.5 and 1.")
        super().__init__(h, input_size, source_dir, **kwargs)
        if self.futr_exog_size != 6:
            raise ValueError("GLAFF requires exactly six known-future calendar columns.")
        config = SimpleNamespace(hist_len=input_size, pred_len=h, flag="Plugin",
                                 dim=d_model, dff=d_ff, head_num=n_heads,
                                 layer_num=e_layers, dropout=dropout, q=q)
        self.model = self._module("backbone.DLinear.model").DLinear(config, channel=1)

    def forward(self, windows_batch):
        y, _, futr = self._observed(windows_batch)
        return self._point_output(self.model(y, futr[:, :self.input_size], None,
                                             futr[:, self.input_size:]), y)


class TGForecaster(_SourceModel):
    """Official TGTSF text-guided forecast with externally prepared embeddings.

    `futr_exog_list` contains text_dim news embedding columns followed by
    text_dim series-description embedding columns. Values for each forecast
    patch MUST be known at its forecast origin (e.g. published plans/metadata).
    This interface accepts one news vector per patch; no text model is downloaded.
    """
    EXOGENOUS_HIST = False

    def __init__(self, h, input_size, source_dir, text_dim=384, d_model=384,
                 n_heads=4, e_layers=2, cross_layers=1, self_layers=1,
                 mixer_self_layers=1, patch_len=8, dropout=0.1, **kwargs):
        _attention(d_model, n_heads, dropout)
        _attention(text_dim, n_heads, dropout)
        _positive(e_layers=e_layers, patch_len=patch_len, cross_layers=cross_layers,
                  self_layers=self_layers, mixer_self_layers=mixer_self_layers)
        if h % patch_len or input_size < patch_len or text_dim % 2 or d_model != text_dim:
            raise ValueError("Require h divisible by patch_len, input_size>=patch_len, and equal, even d_model/text_dim.")
        super().__init__(h, input_size, source_dir, **kwargs)
        if self.futr_exog_size != 2 * text_dim:
            raise ValueError("TGForecaster requires news then description: 2*text_dim futr_exog columns.")
        self.text_dim, self.patch_len = text_dim, patch_len
        config = SimpleNamespace(enc_in=1, seq_len=input_size, pred_len=h,
            e_layers=e_layers, n_heads=n_heads, d_model=d_model, dropout=dropout,
            fc_dropout=dropout, head_dropout=dropout, individual=False,
            patch_len=patch_len, stride=patch_len, padding_patch=None,
            revin=True, affine=False, subtract_last=False, out_attn_weights=False,
            cross_layers=cross_layers, self_layers=self_layers, text_dim=text_dim,
            mixer_self_layers=mixer_self_layers)
        self.model = self._module("models.TGTSF_torch").Model(config)

    def forward(self, windows_batch):
        y, _, futr = self._observed(windows_batch)
        # ponytail: one precomputed news embedding per patch; use an upstream
        # multi-news dataset adapter to preserve multiple articles independently.
        text = futr[:, self.input_size::self.patch_len]
        news = text[..., :self.text_dim].unsqueeze(2).contiguous()
        description = text[..., self.text_dim:].unsqueeze(2).contiguous()
        news_mask = torch.zeros(news.shape[:-1], dtype=torch.bool, device=y.device)
        return self._point_output(self.model(y, news, description, news_mask), y)


class SpecTF(_SourceModel):
    """Official SpecTF TextEncoder + frequency-domain history/prediction network.

    `hist_exog_list` contains a dense, past-only text embedding at each time step.
    The embedding encoder is external; the official TextEncoder projection and
    spectral fusion are trained here. No future text or target enters forward.
    """
    EXOGENOUS_FUTR = False

    def __init__(self, h, input_size, source_dir, mm_emb_size=32, mm_hidden_size=64,
                 text_emb=6, dropout=0.1, text_dropout=0.1, **kwargs):
        _positive(mm_emb_size=mm_emb_size, mm_hidden_size=mm_hidden_size, text_emb=text_emb)
        if mm_emb_size < 2 or not 0 <= dropout < 1 or not 0 <= text_dropout < 1:
            raise ValueError("Require mm_emb_size>=2 and dropout probabilities in [0,1).")
        super().__init__(h, input_size, source_dir, **kwargs)
        if self.hist_exog_size < 2 or self.hist_exog_size % 2:
            raise ValueError("SpecTF requires an even-sized historical text embedding (at least two columns).")
        config = SimpleNamespace(task_name="long_term_forecast", seq_len=input_size,
            pred_len=h, n_ts_features=1, enc_in=1, mm_emb_size=mm_emb_size,
            mm_hidden_size=mm_hidden_size, text_emb=text_emb, llm_emb_size=self.hist_exog_size,
            text_dropout=text_dropout, dropout=dropout, embed="timeF", freq="h",
            channel_independence="1", proj_per_freq=False, freq_cut_off_rate=1.0,
            only_text_input=False, fuse_history=True, use_product=False, sum_fusion=False)
        module = self._module("models.SpecTF")
        self.text_encoder = module.TextEncoder(config)
        self.model = module.FreqModelHistPred(config)

    def forward(self, windows_batch):
        y, hist, _ = self._observed(windows_batch)
        text = self.text_encoder(hist, None)
        return self._point_output(self.model(y, None, None, None, text), y)
