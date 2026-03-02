import logging
import os
from app.config import config

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
        
        # Console handler
        handler = logging.StreamHandler()
        handler.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger