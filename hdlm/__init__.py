try:
    from hsdd.pipeline import GiddPipeline
except ImportError:  # hsdd not installed (public release case)
    GiddPipeline = None

__all__ = ["GiddPipeline"]
