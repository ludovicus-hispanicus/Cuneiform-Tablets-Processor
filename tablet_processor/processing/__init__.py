"""
Processing module for the Cuneiform Tablet Processor.
"""

from .processor import ImageProcessor

# Define all available modules
__all__ = ['ImageProcessor']

# Try to import RawProcessor, but don't fail if rawpy isn't installed
try:
    from .raw_processor import RawProcessor
    __all__.append('RawProcessor')
except ImportError:
    pass

# Try to import BackgroundRemover, but don't fail if OpenCV isn't installed
try:
    from .background_remover import BackgroundRemover
    __all__.append('BackgroundRemover')
except ImportError:
    pass