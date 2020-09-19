try:
    from .coco_eval import CocoEvaluator
except ImportError:
    pass

try:
    from .Evaluator import prepare_for_coco
except ImportError:
    pass
