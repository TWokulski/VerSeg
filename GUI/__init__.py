try:
    from .main_menu_widget import MainMenuWidget
except ImportError:
    pass
try:
    from .segmentation_widget import SegmentationWidget
except ImportError:
    pass
try:
    from .training_widget import TrainingWidget
except ImportError:
    pass
try:
    from .prediction_widget import PredictionWidget
except ImportError:
    pass
