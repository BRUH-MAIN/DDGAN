"""
Models for Deepfake Detection GAN
"""
from .dct_extractor import DCT2D
from .discriminator import Discriminator, DualStreamDiscriminator, create_discriminator
from .generator import Generator

__all__ = [
    'DCT2D', 
    'Discriminator', 
    'DualStreamDiscriminator', 
    'create_discriminator',
    'Generator'
]
