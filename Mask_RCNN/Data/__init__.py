try:
    from .COCO_Evaluator import CocoEvaluator
except ImportError:
    pass

try:
    from .Evaluator import prepare_for_coco
except ImportError:
    pass
