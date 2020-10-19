try:
    from .AnchorGenerator import AnchorGenerator
except ImportError:
    pass

try:
    from .Head import RPNHead
except ImportError:
    pass

try:
    from .RPN import RegionProposalNetwork
except ImportError:
    pass
