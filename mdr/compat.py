from __future__ import annotations

from typing import Any, Callable

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        """No-op fallback when tensorboard is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def __getattr__(self, _name: str) -> Callable[..., None]:
            def _noop(*args: Any, **kwargs: Any) -> None:
                del args, kwargs
                return None

            return _noop
else:
    SummaryWriter = _SummaryWriter


try:
    from transformers import AdamW as _AdamW
except ImportError:
    try:
        from transformers.optimization import AdamW as _AdamW
    except ImportError:
        from torch.optim import AdamW as _AdamW

AdamW = _AdamW
