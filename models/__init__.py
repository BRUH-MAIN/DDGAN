"""
Models for Deepfake Detection GAN
"""
from .dct_extractor import DCT2D
from .discriminator import Discriminator
from .generator import Generator

__all__ = ['DCT2D', 'Discriminator', 'Generator']
