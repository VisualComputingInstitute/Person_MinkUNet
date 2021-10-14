from .mot import MOTAccumulator

# Handles loading the tools library from the server and as a standalone script.
try:
    from . import lap_util
    from . import metrics
    from . import distances
    from . import io
    from . import utils
except:
    import evaluation.lap_util
    import evaluation.metrics
    import evaluation.distances
    import evaluation.io
    import evaluation.utils


# Needs to be last line
__version__ = '1.1.3'