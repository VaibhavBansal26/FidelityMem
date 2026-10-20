"""FidelityMem: recall-constrained adaptive hybrid memory for LLM agents."""
from .compression import ExtractiveCompressor, build_probes, compress_with_gate
from .controller import Config, Controller
from .memory import FidelityMem
from .policy import MemoryPolicy, Mode
from .types import ContextBundle, Memory, MemoryType, Message, QueryType, RetrievedMemory
from .window import AdaptiveWindow

__all__ = [
    "FidelityMem", "MemoryPolicy", "Mode", "Message", "Memory", "MemoryType", "QueryType",
    "ContextBundle", "RetrievedMemory", "AdaptiveWindow", "Controller", "Config",
    "ExtractiveCompressor", "build_probes", "compress_with_gate",
]
__version__ = "0.1.0"
