from models.multimodal import *
from models.text import *

SUPPORTED_MODELS = {
    'EfficientPunct': EfficientPunct,
    'EfficientPunctBERT': EfficientPunctBERT,
    'EfficientPunctTDNN': EfficientPunctTDNN,
}