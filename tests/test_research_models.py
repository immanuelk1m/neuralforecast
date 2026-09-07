"""Real official-source and real NeuralForecast integration checks, no doubles.

Prepare reviewed sources and set NF_RESEARCH_SOURCES. Optional tests have
separate environments in research-backends.yml. Tiny generated checkpoints
exercise the actual backend code, NOT published checkpoint forecast accuracy.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from neuralforecast import NeuralForecast
from neuralforecast.models import DAG, KITE, APT, GLAFF, TGForecaster, SpecTF, ChronosX, Moirai2
from neuralforecast.models._official_source import source_module

SOURCE_ROOT = os.environ.get("NF_RESEARCH_SOURCES")
CONFIGS = {
    "dag": (DAG, 2, False, dict(d_model=16, d_ff=32, n_heads=2, e_layers=1, patch_len=4, stride=2, dropout=0)),
    "kite": (KITE, 2, False, dict(flow_dim=16, flow_depth=1, flow_head=2, rank=2, num_samples=2, num_sampling_steps=2)),
    "apt": (APT, 2, False, dict(timestamp_dim=8, timestamp_hidden=8, num_prototypes=4, top_k=2)),
    "glaff": (GLAFF, 6, False, dict(d_model=16, d_ff=32, n_heads=2, e_layers=1, dropout=0)),
    "tgtsf": (TGForecaster, 16, False, dict(text_dim=8, d_model=8, n_heads=2, e_layers=1, patch_len=2, dropout=0)),
    "spectf": (SpecTF, 8, True, dict(mm_emb_size=4, mm_hidden_size=8, text_emb=2, dropout=0, text_dropout=0)),
}


def source(key):
    if SOURCE_ROOT is None or not (Path(SOURCE_ROOT) / key).is_dir():
        if os.environ.get("NF_REQUIRE_RESEARCH_TESTS") == "1":
            pytest.fail(f"Required official source missing: {key}")
        pytest.skip("Set NF_RESEARCH_SOURCES after preparing the reviewed checkouts.")
    return str(Path(SOURCE_ROOT) / key)


def make_model(key, **kwargs):
    cls, size, hist, architecture = CONFIGS[key]
    options = dict(h=4, input_size=16, source_dir=source(key), max_steps=2,
                   val_check_steps=2, windows_batch_size=4, random_seed=17,
                   logger=False, enable_progress_bar=False, enable_model_summary=False,
                   accelerator="cpu", devices=1)
    options[("hist" if hist else "futr") + "_exog_list"] = [f"x{i}" for i in range(size)]
    options.update(architecture)
    options.update(kwargs)
    return cls(**options)


def features(key, length=20):
    size = CONFIGS[key][1]
    rng = np.random.default_rng(12)
    values = rng.normal(size=(length, size)).astype("float32")
    if key == "apt":
        values = np.stack((np.arange(length) % 24 / 24 - 0.5,
                           np.arange(length) % 7 / 7 - 0.5), axis=-1).astype("float32")
    if key == "glaff":
        dates = pd.date_range("2025-01-01", periods=length, freq="h")
        values = np.stack((dates.month/12-.5, dates.day/31-.5, dates.dayofweek/6-.5,
                           dates.hour/23-.5, dates.minute/59-.5, dates.second/59-.5), axis=-1).astype("float32")
    return values


def batch(key):
    generator = torch.Generator().manual_seed(32)
    y = torch.randn(2, 16, 1, generator=generator)
    exog = torch.from_numpy(features(key)).unsqueeze(0).repeat(2, 1, 1)
    return dict(insample_y=y, insample_mask=torch.ones_like(y), stat_exog=None,
                hist_exog=exog[:, :16] if CONFIGS[key][2] else None,
                futr_exog=None if CONFIGS[key][2] else exog)


@pytest.mark.parametrize("key", CONFIGS)
def test_official_forward_gradients_and_no_future_target(key):
    torch.set_num_threads(1)
    model = make_model(key).eval()
    windows = batch(key)
    torch.manual_seed(1)
    before = model(windows)
    assert before.shape == (2, 4, 1) and torch.isfinite(before).all()
    if key == "kite":
        model.train()
        y, past, future = model._condition(windows)
        objective = model.model.train_function(y, past, torch.randn(2, 4, 1), future)
    else:
        objective = before.square().mean()
    objective.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0
    model.eval()
    windows["outsample_y"] = torch.full((2, 4, 1), float("nan"))
    torch.manual_seed(1)
    torch.testing.assert_close(before, model(windows))


@pytest.mark.parametrize("key", CONFIGS)
def test_exogenous_values_affect_official_forecast(key):
    model = make_model(key).eval()
    windows = batch(key)
    torch.manual_seed(2)
    initial = model(windows)
    exog_key = "hist_exog" if CONFIGS[key][2] else "futr_exog"
    windows[exog_key] = windows[exog_key].flip(1).contiguous()
    torch.manual_seed(2)
    changed = model(windows)
    assert not torch.allclose(initial, changed, atol=1e-7, rtol=1e-7), key


@pytest.mark.parametrize("key", CONFIGS)
def test_rejects_incomplete_and_nonfinite_conditions(key):
    model = make_model(key).eval()
    windows = batch(key)
    windows["insample_mask"][:, 0] = 0
    with pytest.raises(ValueError, match="complete history"):
        model(windows)
    windows = batch(key)
    windows["hist_exog" if CONFIGS[key][2] else "futr_exog"][0, 1, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        model(windows)


def data(key):
    x = features(key, 48)
    df = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
    df["ds"] = pd.date_range("2025-01-01", periods=len(df), freq="h")
    df["unique_id"] = "series"
    df["y"] = np.sin(np.arange(len(df)) / 3).astype("float32") + x[:, 0]
    return df.iloc[:-4].copy(), df.iloc[-4:].drop(columns="y").copy()


@pytest.mark.parametrize("key", CONFIGS)
def test_neuralforecast_fit_predict_save_load_real_sources(key, tmp_path):
    torch.set_num_threads(1)
    model = make_model(key)
    nf = NeuralForecast(models=[model], freq="h")
    train, future = data(key)
    nf.fit(df=train, val_size=4)
    result = nf.predict(futr_df=future if not CONFIGS[key][2] else None)
    assert len(result) == 4 and np.isfinite(result[str(nf.models[0])]).all()
    path = tmp_path / key
    nf.save(path=str(path), save_dataset=True, overwrite=False)
    restored = NeuralForecast.load(path=str(path))
    again = restored.predict(futr_df=future if not CONFIGS[key][2] else None)
    np.testing.assert_allclose(result[str(nf.models[0])], again[str(nf.models[0])], atol=1e-5, rtol=1e-5)
    if key in {"apt", "dag"}:
        assert "train_auxiliary_loss" in nf.models[0].metrics
    if key == "kite":
        assert "train_flow_loss" in nf.models[0].metrics


def test_calendar_validation_and_tgtsf_dim_guard():
    windows = batch("apt")
    windows["futr_exog"][..., 0] = 100
    with pytest.raises(ValueError, match="calendar"):
        make_model("apt")(windows)
    with pytest.raises(ValueError, match="equal"):
        make_model("tgtsf", d_model=16)
    with pytest.raises(ValueError, match="h>=2"):
        make_model("kite", h=1)


def test_source_imports_do_not_shadow_unrelated_packages():
    old = {name: sys.modules.get(name) for name in ("models", "layers", "utils", "ts_benchmark")}
    left = source_module(source("tgtsf"), "layers.RevIN")
    right = source_module(source("apt"), "baselines.Normalization.normalization.Revin")
    assert left.RevIN is not right.RevIN
    assert all(sys.modules.get(name) is value for name, value in old.items())
    with pytest.raises(ValueError, match="dotted"):
        source_module(source("apt"), "../outside")


def test_kite_historical_only_objective_with_unconditional_dropout(monkeypatch):
    model = make_model("kite", futr_exog_list=None, hist_exog_list=["a", "b"], p_uncond=0.99)
    windows = batch("kite")
    windows["hist_exog"] = windows["futr_exog"][:, :16]
    windows["futr_exog"] = None
    y, past, future = model._condition(windows)
    assert future is None
    import random
    monkeypatch.setattr(random, "random", lambda: 0.0)
    objective = model.model.train_function(y, past, torch.randn(2, 4, 1), y.new_zeros(2, 4, 2))
    objective.backward()
    assert torch.isfinite(objective)
    model.eval()
    assert torch.isfinite(model(windows)).all()


def tiny_chronos_config():
    return dict(vocab_size=64, d_model=32, d_ff=64, d_kv=8, num_heads=4,
                num_layers=1, num_decoder_layers=1, dropout_rate=0.0,
                pad_token_id=0, eos_token_id=1, decoder_start_token_id=0,
                chronos_config=dict(tokenizer_class="MeanScaleUniformBins",
                    tokenizer_kwargs=dict(low_limit=-15.0, high_limit=15.0),
                    context_length=16, prediction_length=4, n_tokens=64,
                    n_special_tokens=2, pad_token_id=0, eos_token_id=1,
                    use_eos_token=True, model_type="seq2seq", num_samples=2,
                    temperature=1.0, top_k=20, top_p=1.0))


def test_chronosx_real_token_objective_and_nf_checkpoint(tmp_path):
    if os.environ.get("NF_TEST_CHRONOSX") != "1":
        pytest.skip("Dedicated pinned ChronosX environment required.")
    torch.set_num_threads(1)
    model = ChronosX(h=4, input_size=16, source_dir=source("chronosx"),
                    model_config=tiny_chronos_config(), hidden_dim=16, num_samples=2,
                    futr_exog_list=["x0", "x1"], max_steps=2, val_check_steps=2,
                    windows_batch_size=4, logger=False, enable_progress_bar=False,
                    enable_model_summary=False, accelerator="cpu", devices=1)
    nf = NeuralForecast(models=[model], freq="h")
    train, future = data("dag")
    nf.fit(df=train, val_size=4)
    assert "train_token_cross_entropy" in nf.models[0].metrics
    result = nf.predict(futr_df=future)
    assert np.isfinite(result["ChronosX"]).all()
    nf.save(path=str(tmp_path / "chronosx"), save_dataset=True)
    again = NeuralForecast.load(path=str(tmp_path / "chronosx")).predict(futr_df=future)
    np.testing.assert_allclose(result["ChronosX"], again["ChronosX"], atol=1e-6)
    frozen = [p for name, p in nf.models[0].model.named_parameters() if "injection_block" not in name]
    assert frozen and not any(p.requires_grad for p in frozen)
    windows = batch("dag")
    y, ids, attention, scale, past, future_values = nf.models[0]._tokens(windows)
    assert past.shape == (2, 17, 2) and future_values.shape == (2, 5, 2)
    assert past[:, -1].eq(0).all() and future_values[:, -1].eq(0).all()
    windows["outsample_y"] = torch.full((2, 4, 1), float("nan"))
    assert torch.isfinite(nf.models[0](windows)).all()


def test_moirai2_real_isolated_worker_and_nf_checkpoint(tmp_path):
    python = os.environ.get("NF_UNI2TS_PYTHON")
    if not python:
        pytest.skip("Dedicated uni2ts interpreter required.")
    checkpoint = tmp_path / "moirai2-tiny"
    script = '''
from uni2ts.model.moirai2 import Moirai2Module
import torch, sys
torch.manual_seed(4)
model = Moirai2Module(d_model=64, d_ff=128, num_layers=1, patch_size=4,
    max_seq_len=128, attn_dropout_p=0., dropout_p=0., num_predict_token=2)
model.save_pretrained(sys.argv[1])
'''
    subprocess.run([python, "-c", script, str(checkpoint)], check=True, timeout=180)
    model = Moirai2(h=4, input_size=16, backend_python=python, model_id=str(checkpoint),
                    futr_exog_list=["x0", "x1"], logger=False, enable_progress_bar=False,
                    accelerator="cpu", devices=1)
    nf = NeuralForecast(models=[model], freq="h")
    train, future = data("dag")
    nf.fit(df=train)
    result = nf.predict(futr_df=future)
    assert np.isfinite(result["Moirai2"]).all()
    nf.save(path=str(tmp_path / "nf"), save_dataset=True)
    again = NeuralForecast.load(path=str(tmp_path / "nf")).predict(futr_df=future)
    np.testing.assert_allclose(result["Moirai2"], again["Moirai2"], atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="checkpoint"):
        Moirai2(h=4, input_size=16, patch_size=4)
