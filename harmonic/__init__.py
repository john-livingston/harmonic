from __future__ import annotations

import logging
import sys
import os

def setup_logging(console_level: int = logging.INFO, file_level: int = logging.INFO,
                  format_string: str | None = None, log_file: str | None = None):
    """Setup logging configuration for harmonic package
    
    Args:
        console_level: Log level for console output (default: INFO)
        file_level: Log level for file output (default: INFO) 
        format_string: Format for console output (default: clean message only)
        log_file: Path to log file (default: None, no file logging)
    """
    if format_string is None:
        # Clean format for scientific output
        format_string = '%(message)s'
    
    # Get the root logger for harmonic package
    harmonic_logger = logging.getLogger('harmonic')
    
    # Clear any existing handlers to avoid duplicates
    harmonic_logger.handlers.clear()
    
    # Always add console handler for immediate feedback
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(format_string))
    harmonic_logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            # File format includes timestamp for better record keeping
            file_format = '%(asctime)s - %(levelname)s - %(message)s'
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setLevel(file_level)
            file_handler.setFormatter(logging.Formatter(file_format))
            harmonic_logger.addHandler(file_handler)
        except Exception as e:
            # If file logging fails, just continue with console logging
            print(f"Warning: Could not create log file {log_file}: {e}")
    
    # Set logger to the most permissive level to allow both handlers to work
    harmonic_logger.setLevel(min(console_level, file_level))
    harmonic_logger.propagate = False
    
    # Suppress verbose libraries after they're imported
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('astropy').setLevel(logging.WARNING)

from .exceptions import HarmonicError, ConfigurationError, DataError, PredictionError
from .harmonic import Harmonic

__version__ = '0.4.0'
