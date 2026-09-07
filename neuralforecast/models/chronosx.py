"""Official ChronosX injection-block training with NeuralForecast windows.

Requires the pinned official ChronosX checkout, chronos-forecasting==1.5.0,
transformers==4.47.1 and accelerate==0.34.2 in a dedicated environment.
The default initializes from Chronos-T5-small and trains new injection blocks;
it is NOT a zero-shot covariate model before fitting.
"""

from pathlib import Path
import threading

import torch

from .research import _SourceModel, _positive
from ._official_source import source_module

__all__ = ["ChronosX"]
_BUILD_LOCK = threading.RLock()


class ChronosX(_SourceModel):
    """Official input/output injection blocks around pretrained Chronos-T5.

    Only numerical known-future covariates are accepted. Training minimizes
    the official model's token cross entropy; NF MAE/MSE is used for validation
    of sampled point forecasts. `train_backbone=False` trains injection blocks
    only, as in the upstream fine-tuning path. `model_config` is an explicit
    from-scratch configuration for small/offline experiments, not pretrained.

    NF checkpoints contain all trainable weights. Reconstruction also needs
    the official checkout and the original model/tokenizer config (Hub cache
    or a local model directory). No optional backend imports occur at NF import.
    """
    EXOGENOUS_HIST = False

    def __init__(self, h, input_size, source_dir,
                 model_id="amazon/chronos-t5-small", model_config=None,
                 hidden_dim=256, injection_layers=1, num_samples=20,
                 train_backbone=False, **kwargs):
        _positive(hidden_dim=hidden_dim, injection_layers=injection_layers, num_samples=num_samples)
        super().__init__(h, input_size, source_dir, **kwargs)
        if not self.futr_exog_size:
            raise ValueError("ChronosX requires numerical futr_exog_list.")
        try:
            import transformers
            from chronos import ChronosConfig
            from transformers import AutoConfig, T5Config
        except ImportError as exc:
            raise ImportError("Install the pinned ChronosX optional environment; see docs/research_models.md.") from exc
        if not transformers.__version__.startswith("4.47."):
            raise ImportError("The reviewed ChronosX source requires transformers==4.47.1; use a separate environment.")
        module = source_module(Path(self.source_dir) / "src", "chronosx.chronosx")
        config = (AutoConfig.from_pretrained(model_id) if model_config is None
                  else T5Config(**model_config))
        if config.model_type != "t5" or not hasattr(config, "chronos_config"):
            raise ValueError("ChronosX requires a T5 configuration containing chronos_config.")
        tokenizer_config = dict(config.chronos_config)
        if input_size > tokenizer_config["context_length"]:
            raise ValueError("input_size exceeds the tokenizer context_length.")
        if not tokenizer_config.get("use_eos_token", False):
            raise ValueError("The official covariate preparation requires use_eos_token=True.")
        tokenizer_config["prediction_length"] = h
        self.tokenizer = ChronosConfig(**tokenizer_config).create_tokenizer()
        # The upstream factory stores dimensions on its class. Give each model
        # a private subclass, so constructing another model cannot mutate it.
        with _BUILD_LOCK:
            implementation = type("ConfiguredChronosX", (module.ChronosX,), {})
            implementation.set_state(
                num_covariates=self.futr_exog_size, covariate_injection="IIB+OIB",
                hidden_dim=hidden_dim, num_layers=injection_layers,
                vocab_size=config.vocab_size, model_dim=config.d_model,
            )
            if model_config is None:
                self.model = implementation.from_pretrained(model_id)
            else:
                self.model = implementation(config)
            self.model.initialize_blocks()
            if not train_backbone:
                self.model.freeze("all")
                self.model.unfreeze("injection_block")
        self.num_samples = num_samples

    def _tokens(self, windows_batch):
        y, _, futr = self._observed(windows_batch)
        ids, attention, scale = self.tokenizer.context_input_transform(y[..., 0])
        past, future = futr[:, :self.input_size], futr[:, self.input_size:]
        # Same past-only mean-absolute normalization and EOS padding as the
        # official prepare_covariates function. No cross-window pooling.
        cov_scale = past.abs().mean(dim=1, keepdim=True).clamp_min(1)
        past, future = past / cov_scale, future / cov_scale
        past = torch.cat((past, past.new_zeros(past.shape[0], 1, past.shape[2])), dim=1)
        future = torch.cat((future, future.new_zeros(future.shape[0], 1, future.shape[2])), dim=1)
        if past.shape[1] != ids.shape[1]:
            raise ValueError("Tokenizer and historical covariate lengths do not match.")
        return y, ids, attention, scale, past, future

    def training_step(self, batch, batch_idx):
        windows, target = self._objective_batch(batch)
        y, ids, attention, scale, past, future = self._tokens(windows)
        labels, label_mask = self.tokenizer.label_input_transform(target[..., 0], scale)
        labels = labels.masked_fill(~label_mask, -100)
        result = self.model(input_ids=ids, attention_mask=attention, labels=labels,
                            past_covariates=past, future_covariates=future)
        return self._log_objective(result.loss, "train_token_cross_entropy")

    @torch.no_grad()
    def forward(self, windows_batch):
        from transformers import GenerationConfig
        y, ids, attention, scale, past, future = self._tokens(windows_batch)
        was_training = self.model.training
        self.model.eval()
        try:
            tokens = self.model.generate(
                input_ids=ids, attention_mask=attention,
                past_covariates=past, future_covariates=future,
                generation_config=GenerationConfig(
                    min_new_tokens=self.h, max_new_tokens=self.h, do_sample=True,
                    num_return_sequences=self.num_samples,
                    eos_token_id=self.model.config.eos_token_id,
                    pad_token_id=self.model.config.pad_token_id,
                ),
            )[:, 1:].reshape(y.shape[0], self.num_samples, self.h)
        finally:
            self.model.train(was_training)
            for name in ("input_injection_block", "input_injection_block_decoder", "output_injection_block"):
                block = getattr(self.model, name, None)
                if block is not None:
                    block.generating = False
        samples = self.tokenizer.output_transform(tokens, scale)
        return self._point_output(samples.mean(dim=1), y)

    def predict(self, *args, **kwargs):
        if kwargs.get("explainer_config") is not None:
            raise ValueError("Token sampling does not support gradient explanations.")
        return super().predict(*args, **kwargs)
