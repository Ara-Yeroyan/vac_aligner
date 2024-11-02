import os
import json
from typing import Dict

import wave

from .dto import ASRConfig, ALIGNERConfig


def add_durations(manifest_path: str):
    new_path = manifest_path.replace(".json", "_with_durs.json")
    with open(manifest_path, "r", encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if "duration" not in item:
                with open(new_path, "r", encoding='utf-8') as w:
                    audio_path = item['audio_filepath']
                    duration = get_wav_duration(audio_path)
                    item['duration'] = duration
                    w.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                return
    print(f"Successfully created a new vad_manifest with durations: {new_path}")


def get_wav_duration(file_path):
    with wave.open(file_path, 'r') as wav_file:
        frames = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
        duration = frames / float(frame_rate)

    return duration


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
