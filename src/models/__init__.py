"""Models package for deepfake detection GAN."""

from .dct_extractor import DCT2D, DCTFeatureExtractor
from .discriminator import DCTDiscriminator
from .generator import UNetGenerator, FrequencyAwareBottleneck

__all__ = [
    'DCT2D',
    'DCTFeatureExtractor',
    'DCTDiscriminator',
    'UNetGenerator',
    'FrequencyAwareBottleneck',
]
