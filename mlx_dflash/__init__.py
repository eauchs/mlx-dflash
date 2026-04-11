try:
    from .models.qwen3_dflash import DFlashDraftModel
    from .engine import DFlashEngine
except ImportError:
    pass  # MLX not available (non-macOS)
from .convert import convert

__version__ = "0.1.0"
