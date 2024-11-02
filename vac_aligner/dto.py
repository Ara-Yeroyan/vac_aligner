from typing import Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkData:
    save_path: str
    audio_path: str
    duration: float
    asr_prediction: str
    text: Optional[str] = None
    source_text: Optional[str] = None
    source_audio: Optional[str] = None


@dataclass(frozen=True)
class ASRConfig:
    """
    Configuration for Automatic Speech Recognition (ASR).

    Attributes:
        hf_token (str): Hugging Face API token for authentication.
        gpu_id (Optional[int]): ID of the GPU to use (None if no GPU).
        batch_size (int): The number of samples to process in a batch.
    """
    hf_token: str
    batch_size: int = 15
    gpu_id: Optional[int] = 0


@dataclass(frozen=True)
class ALIGNERConfig:
    """
    Configuration for the Aligner.

    Attributes:
        manifest_file (str):
            The path to the NEMO manifest file. Each line in this file should be a valid JSON string representing
            individual audio data with keys [audio_filepath, pred_text, duration].

        transcript_path (Optional[str]):
            The path to the file containing the combined transcript of all audio files.
            If None, it will be calculated (combined) based on the "text" key in the manifest_file.

        save_manifest_path (str):
            The path where the processed manifest data will be saved or utilized.

        use_id (bool):
            Determines whether to sort the audio chunks based on their ID. Useful when IDs are used to denote the
            order of audio files (chunks). Default is False.

        target_base (Optional[str]):
            The base path that could be prefixed to each audio file path in the manifest, useful for cases where
            the manifest paths are relative. Default is None.

        ending_punctuations (Optional[str]):
            The punctuation symbols accumulated in a single string (with the most common sentence-ending punctuation
            placed at the end of the string, such as "...,;:."). Only necessary if transcript_path is None, so that
            the texts can be combined into a single transcript. Default is an example for Armenian Language - '․,։'.
    """
    manifest_file: str
    save_manifest_path: str

    use_id: bool = False
    target_base: Optional[str] = None
    transcript_path: Optional[str] = None
    ending_punctuations: Optional[str] = '․,։'
