# mkyz/utils/logging.py
"""Logging configuration and utilities for MKYZ library."""

import logging
import sys
from enum import Enum
from typing import Optional
from datetime import datetime


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


# MKYZ custom logger name
LOGGER_NAME = 'mkyz'

# Color codes for terminal output
COLORS = {
    'DEBUG': '\033[36m',     # Cyan
    'INFO': '\033[32m',      # Green
    'WARNING': '\033[33m',   # Yellow
    'ERROR': '\033[31m',     # Red
    'CRITICAL': '\033[35m',  # Magenta
    'RESET': '\033[0m'       # Reset
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for terminal."""
    
    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        # Format level name
        level_name = record.levelname
        
        if self.use_colors:
            color = COLORS.get(level_name, COLORS['RESET'])
            reset = COLORS['RESET']
            formatted = f"{color}[{timestamp}] [{level_name}]{reset} {record.getMessage()}"
        else:
            formatted = f"[{timestamp}] [{level_name}] {record.getMessage()}"
        
        return formatted


def setup_logging(level: str = 'INFO',
                 use_colors: bool = True,
                 log_file: Optional[str] = None) -> logging.Logger:
    """Set up MKYZ logging configuration.
    
    Args:
        level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        use_colors: Whether to use colored terminal output
        log_file: Optional file path to write logs
        
    Returns:
        Configured logger instance
    
    Examples:
        >>> from mkyz.utils import setup_logging
        >>> logger = setup_logging(level='DEBUG')
        >>> logger.info("Training started")
        [14:30:25] [INFO] Training started
    """
    # Get or create MKYZ logger
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter(use_colors=use_colors))
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger() -> logging.Logger:
    """Get the MKYZ logger instance.
    
    Returns:
        MKYZ logger instance
    
    Examples:
        >>> from mkyz.utils import get_logger
        >>> logger = get_logger()
        >>> logger.debug("Preprocessing data...")
    """
    logger = logging.getLogger(LOGGER_NAME)
    
    # Set up default configuration if no handlers exist
    if not logger.handlers:
        setup_logging()
    
    return logger


def log_training_start(model_name: str, task: str) -> None:
    """Log training start message."""
    logger = get_logger()
    logger.info(f"Starting {task} training with {model_name}")


def log_training_complete(model_name: str, 
                         duration: float, 
                         score: Optional[float] = None) -> None:
    """Log training completion message."""
    logger = get_logger()
    msg = f"Training {model_name} completed in {duration:.2f}s"
    if score is not None:
        msg += f" (score: {score:.4f})"
    logger.info(msg)


def log_error(message: str, exc: Optional[Exception] = None) -> None:
    """Log error message."""
    logger = get_logger()
    if exc:
        logger.error(f"{message}: {exc}")
    else:
        logger.error(message)
