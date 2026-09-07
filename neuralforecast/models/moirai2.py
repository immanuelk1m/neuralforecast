"""Moirai 2.0 median forecasts using the existing isolated uni2ts transport."""

from .foundation import Moirai

__all__ = ["Moirai2"]


class Moirai2(Moirai):
    """Official Moirai 2.0 median forecasts through the isolated uni2ts worker.

    Uses the architecture/checkpoint's own patch size and quantile grid, not
    Moirai-1's sample averaging. Historical and known-future numerical covariates
    are both supported. This adapter is inference-only (max_steps=0).
    """
    DEFAULT_MODEL_ID = "Salesforce/moirai-2.0-R-small"
    BACKEND_KIND = "moirai2"

    def __init__(self, h, input_size, backend_python=None, backend_timeout=600, **kwargs):
        if "patch_size" in kwargs or "num_samples" in kwargs:
            raise ValueError("Moirai2 uses its checkpoint's patch size and median quantile, not num_samples.")
        super().__init__(h=h, input_size=input_size, backend_python=backend_python,
                         backend_timeout=backend_timeout, **kwargs)
        # BaseModel saves intermediate constructor defaults; they are not
        # public Moirai2 parameters and must not be replayed during load().
        self.hparams.pop("patch_size", None)
        self.hparams.pop("num_samples", None)
