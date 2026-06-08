"""Minimal config module for wav2vec2_depression_detection.py"""
import logging
import sys
from pathlib import Path

logger = logging.getLogger("depression_detection")


def setup_logging(log_file: str = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )
