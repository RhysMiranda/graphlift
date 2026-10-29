"""graphlift: does your graph actually help?

    from graphlift import load_dir, audit
    result = audit.run(load_dir("my_dataset/"))
"""
from . import audit, report
from .data import GraphData, load_dir

__version__ = "1.0.1"
__all__ = ["GraphData", "load_dir", "audit", "report", "__version__"]
