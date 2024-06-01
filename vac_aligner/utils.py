import os
from typing import Dict


from .dto import ASRConfig, ALIGNERConfig


def create_nested_folders(path: str):
    base = os.path.dirname(path)
    os.makedirs(base, exist_ok=True)


def _validate_aligner_info(asr_info: Dict) -> ALIGNERConfig:
    try:
        validated_asr_info = ALIGNERConfig(**asr_info)
        return validated_asr_info
    except TypeError as e:
        raise ValueError(f"Invalid ASR info provided: {e}")


def _validate_asr_info(asr_info: Dict) -> ASRConfig:
    try:
        validated_asr_info = ASRConfig(**asr_info)
        return validated_asr_info
    except TypeError as e:
        raise ValueError(f"Invalid ASR info provided: {e}")
