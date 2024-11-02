import os
import json
import wave
import contextlib
from typing import List, Tuple, Union, Optional

import pandas as pd

from vac_aligner.dto import ASRConfig, ChunkData
from vac_aligner.utils import create_nested_folders

from glob import glob


class ASR:
    """
    Class which  perform ASR inference in a loop and writes to NeMo manifest.
    NeMo and torch related imports are made local as we have `pip install vac_aligner['full']` if asr is needed
    """
    def __init__(self, model_name: str, asr_config: ASRConfig, skip_existing: bool = True):
        from nemo.collections.asr.models import EncDecCTCModelBPE

        self.asr_config = asr_config
        self.skip_existing = skip_existing
        self.check_hf_credential(model_name)
        self.asr_model_bpe: EncDecCTCModelBPE = self.load_model(model_name)

    @staticmethod
    def check_hf_credential(model_name: str, hf_token: Optional[str] = None):
        if not hf_token:
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token is None:
                raise ValueError("Please, provide hf_token or directly set the hugging face token as ENV - HT_TOKEN."
                                 f"You might also need to request access at: https://huggingface.co/{model_name}")

    def load_model(self, model_name: str):
        """Load the model (currently support only HF) and map to correct device"""
        import torch
        from nemo.collections.asr.models import ASRModel

        if not os.environ.get("DISABLE_HF", False):
            model = ASRModel.from_pretrained(model_name)
            if self.asr_config.gpu_id is None:
                device = torch.device('cpu')
            else:
                device = torch.device(f'cuda:{self.asr_config.gpu_id}' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return model
        else:
            raise NotImplemented

    @staticmethod
    def extract_wav_files(wav_files: Union[str, List[str]], test_manifest: Optional[str] = None,
                          audio_sentence_map: Optional[pd.DataFrame] = None) -> \
            Tuple[List[str], List[Optional[str]], List[str]]:
        """Extract audios and corresponding texts from various input formats (dataframe, NeMo json, source wav_dir)"""
        sources = []
        if test_manifest:
            texts = []
            wav_files = []
            with open(test_manifest, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    texts.append(item.get("text", None))
                    wav_files.append(item["audio_filepath"])
                    sources.append(item.get("source_filepath"))
            print("Loaded only Test audios given in the optional manifest!")
        else:
            if isinstance(wav_files, str):
                wav_files = glob(wav_files+"/*.wav")
            try:
                texts = [audio_sentence_map.loc[os.path.basename(wav_path).replace(".wav", ".mp3"), 'sentence'] for wav_path in wav_files]
            except Exception as e:
                print(e.__str__())
                texts = [None for _ in range(len(wav_files))]
        return wav_files, texts, sources

    def run(self, wav_files: Union[str, List[str]], save_dir: Optional[str] = None,
            save_manifest: Optional[str] = None, test_manifest: Optional[str] = None,
            audio_sentence_map: Optional[pd.DataFrame] = None) -> List[ChunkData]:
        """

        Inference with while (batched) as loading too many audios decreases the performance

        Parameters:
            wav_files : Union[str, List[str]]
                If str then have to be the path to the directory where the audios (.wav) where stored or a list
                    containing the paths (audio_filepath)

            save_manifest : str
                The path to the nemo manifest file to save the chunk info.

            save_dir : Optional[str]
                The path to the folder where to ASR predictions (single .txt per audio_filepath). If not provided,
                    will store in the same place as wav_files are stored. Defaults to None.

            test_manifest : Optional[str]
                If there is a nemo manifest available, then it can also be provided to replace wav_files.
                The format of manifest should just contain: [audio_filepath]. If not provided, will only refer to
                    wav_files. Default is None.

        Returns:
             A list of chunks with corresponding info (audio_filepath, chunk_text, chunk_text_path, audio_duration, ...)
        """
        create_nested_folders(save_dir)
        create_nested_folders(save_manifest)
        print(f'test_manifest: {test_manifest}')
        wav_files, texts, sources = self.extract_wav_files(wav_files,
                                                           test_manifest,
                                                           audio_sentence_map=audio_sentence_map)
        wav2text = {wav_files[i]: texts[i] for i in range(len(wav_files))}
        wav2source = {wav_files[i]: sources[i] if sources else None for i in range(len(wav_files))}
        print(f"#of Audios to Process! {len(wav_files)}")
        b_size = self.asr_config.batch_size
        batch_end = b_size * os.environ.get("BATCH_MULTIPLIER", 1)
        full_preds = []

        while batch_end <= len(wav_files):
            batch = wav_files[batch_end - b_size:batch_end]
            if self.skip_existing and all([os.path.exists(self.wav2text_path(x, save_dir)) for x in batch]):
                batch_end += b_size
                continue
            preds = self.asr_model_bpe.transcribe(batch)
            if isinstance(preds[0], list):  # some NeMo models may return [[asr_preds], [asr_preds]]. shape: [2, n_batch]
                preds = preds[0]
            for i, wav in enumerate(batch):
                duration = self.get_wav_duration(wav)
                chunk_txt = self.wav2text_path(wav, save_dir)
                with open(chunk_txt, "w", encoding='utf-8') as f:
                    f.write(preds[i])
                full_preds.append(
                    ChunkData(
                        audio_path=wav,
                        duration=duration,
                        save_path=chunk_txt,
                        asr_prediction=preds[i],
                        text=wav2text[wav],
                        source_audio=wav2source[wav]
                    )
                )
            batch_end += b_size

            if batch_end % 80 == 0:
                print(chunk_txt)
                print(f"Batch reached: {batch_end}")

        not_processed = list(filter(lambda x: not os.path.exists(self.wav2text_path(x, save_dir)), wav_files))
        if not_processed:
            print("Running on the last (partial) batch!")
            text_batch = texts[batch_end - b_size:batch_end]
            preds = self.asr_model_bpe.transcribe(not_processed)
            if isinstance(preds[0], list): # some NeMo models may return [[asr_preds], [asr_preds]]. shape: [2, n_batch]
                preds = preds[0]
            for i, wav in enumerate(not_processed):
                chunk_txt = wav.replace(".wav", ".txt")
                if os.path.exists(chunk_txt):
                    continue
                with open(chunk_txt, "w", encoding='utf-8') as f:
                    f.write(preds[i])
                duration = self.get_wav_duration(wav)
                full_preds.append(
                    ChunkData(
                        audio_path=wav,
                        duration=duration,
                        save_path=chunk_txt,
                        asr_prediction=preds[i],
                        text=wav2text[wav],
                        source_audio=wav2source[wav]
                    )
                )

        print(f"#of Audios Successfully Processed! [{len(full_preds)}/{len(wav_files)}]")
        if save_manifest and full_preds:  # maybe all the files were skipped
            self.write_to_manifest(full_preds, save_manifest)
        return full_preds

    @staticmethod
    def write_to_manifest(items: List[ChunkData], save_manifest: str):
        """Saving chunks into NeMo manifest with corresponding format (each raw is a json representing one chunk)"""
        with open(save_manifest, "w", encoding='utf-8') as f:
            for item in items:
                raw = {
                    'audio_filepath': item.audio_path,
                    'pred_text': item.asr_prediction,
                    'duration': item.duration,
                }
                text = item.text
                if text:
                    raw['text'] = text

                source = item.source_audio
                if source:
                    raw['source_audio'] = source
                f.write(json.dumps(raw, ensure_ascii=False)+"\n")

    @staticmethod
    def wav2text_path(wav_filepath: str, save_dir: Optional[str] = None):
        if save_dir is None:
            return wav_filepath.replace(".wav", ".txt")
        audio_name = wav_filepath.split("\\")[-1].split("/")[-1]
        txt_filename = audio_name.replace(".wav", ".txt")
        txt_filepath = os.path.join(save_dir, txt_filename)
        return txt_filepath

    @staticmethod
    def get_wav_duration(wav_path: str) -> float:
        with contextlib.closing(wave.open(wav_path, 'rb')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return duration


if __name__ == "__main__":
    model_name = "Yeroyan/stt_arm_conformer_ctc_large"
    asr_config = ASRConfig(hf_token=os.environ['HF_TOKEN'], batch_size=24)
    asr = ASR(model_name, asr_config)
    asr.run(
        save_dir=r"D:\CapstoneThesisArmASR\vac_aligner\experiments\asr_preds",
        save_manifest=r"D:\CapstoneThesisArmASR\vac_aligner\experiments\asr.json",
        wav_files=r"D:\Capstone_Thesis\cv-corpus-17.0-2024-03-15\hy-AM\resampled_clips",
        test_manifest=r"D:\CapstoneThesisArmASR\vac_aligner\vac_aligner\data\matching\test_with_predictions17_replaced.json"
    )
