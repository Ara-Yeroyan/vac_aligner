import os
from dataclasses import asdict
from typing import Optional, Tuple, Union, List, Dict

import pandas as pd
from loguru import logger

from .dto import ASRConfig, ALIGNERConfig
from vac_aligner.asr import ASR, ASR_MAPPING
from .utils import _validate_asr_info, _validate_aligner_info
from vac_aligner.matching import ALIGNER_MAPPING, BaseAlignerVAC, Benchmark


class VAC:
    def __init__(self, language, asr_info: Optional[Dict], aligner_matching_info: Optional[Dict],
                 init_aligner_after_asr: bool = False):
        self.language = language
        self.asr_info = _validate_asr_info(asr_info) if asr_info else None
        self.aligner_matching_info = _validate_aligner_info(aligner_matching_info) if aligner_matching_info else None

        if isinstance(self.asr_info, ASRConfig):
            self.asr_model_name = self._get_asr_model_name(language)
            self.asr_model = ASR(self.asr_model_name, self.asr_info)

        if not init_aligner_after_asr:
            self.init_aligner()

        self.init_aligner_after_asr = init_aligner_after_asr

    def init_aligner(self):
        if isinstance(self.aligner_matching_info, dict):
            self.aligner_matching_info = ALIGNERConfig(**self.aligner_matching_info)
        self.aligner_class = self._get_aligner_class(self.language)
        self.aligner: BaseAlignerVAC = self.aligner_class.from_nemo_manifest(**asdict(self.aligner_matching_info))

    @staticmethod
    def _get_aligner_class(language):
        """Map language to corresponding Aligner Class"""
        aligner = ALIGNER_MAPPING.get(language)
        if not aligner:
            raise ValueError(f"Cannot retrieve aligner. Unsupported language: {language}")
        return aligner

    @staticmethod
    def _get_asr_model_name(language) -> str:
        model_name = ASR_MAPPING.get(language)
        if not model_name:
            raise ValueError(f"Cannot retrieve aligner. Unsupported language: {language}")
        return model_name

    def combine_transcript_and_align(self, predictions_manifest: str, save_manifest_path: str,
                                     output_file: str, target_base: str) -> List[Tuple[str, str, float, int, int]]:
        chunks, combined_transcript = self.aligner_class.combine_transcript(predictions_manifest, output_file)
        logger.info("Combined the MCV files into single transcript")
        self.aligner: BaseAlignerVAC = self.aligner_class(combined_transcript, chunks,
                                                          save_manifest_path, target_base=target_base)
        matches_sorted = self.aligner.align(0.35)
        return [tuple(x[:-1]) for x in matches_sorted]

    def run_asr(self, wav_files: Union[str, List[str]], save_dir: Optional[str] = None,
                save_manifest: Optional[str] = None, test_manifest: Optional[str] = None,
                audio_sentence_map: Optional[pd.DataFrame] = None):

        chunk_data = self.asr_model.run(
            save_dir=save_dir,
            wav_files=wav_files,
            save_manifest=save_manifest,
            test_manifest=test_manifest,
            audio_sentence_map=audio_sentence_map
        )

        if self.init_aligner_after_asr:
            print("Initializing Aligner with the ASR manifest")
            self.init_aligner()

        return chunk_data

    def align_match(self, cer_bound: float = 0.35) -> List:
        return self.aligner.align(cer_bound)

    def __call__(self, wav_files: Union[str, List[str]], save_dir: Optional[str] = None,
                 save_manifest: Optional[str] = None, test_manifest: Optional[str] = None,
                 audio_sentence_map: Optional[pd.DataFrame] = None):

        self.run_asr(
            save_dir=save_dir,
            wav_files=wav_files,
            save_manifest=save_manifest,
            test_manifest=test_manifest,
            audio_sentence_map=audio_sentence_map
        )

        self.aligner.align(0.35)

    @classmethod
    def run_end2end(cls, manifest_file: str, asr_input_file: str, language: str = "hy", init_aligner_after_asr = False,
                    target_base: Optional[str] = None, ending_punctuations: Optional[str] = '․,։', hf_token: str = '',
                    cer_bound: float=0.35,  batch_size: int = 15, gpu_id: Optional[int] = 0,
                    audio_sentence_map: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Initialising VAC and running ASR -> obtained predictions manifest, run Aligner (combine transcript and match)

        Parameters:
            manifest_file : str
                The path to the NEMO manifest file (where to save the ASR predictions and then load for Aligner).
                Each line in this file should be a valid JSON string representing individual audio data with keys:
                    [audio_filepath, pred_text, duration]

            asr_input_file: str
                Either json manifest where audio_filepath indicates the audios to be inferenced on or a path to folder
                    containing .wav files. Used in ASR stage.

            language: str
                The language of the dataset. Currently, Default to "hy" (Armenian).

            target_base : Optional[str], optional
                The base path that could be prefixed to each audio file path in the manifest, indicating where to save
                    both ASR predictions (.txt s) and the matches (_matched.txt s). If not provided, will write the
                    text files in the same directory as the audio files are stored. Default is None.

            ending_punctuations : Optional[str]
                The punctuation symbols accumulated in single string (with most common sentence ending punctuation
                placed in the end of the string - such as: "...,;:.". Only necessary if transcript_path is None so that
                we combine the texts into a single transcript. Default is example for Armenian Language - '․,։'

            hf_token : str
                The hugging face token to fetch an ASR model. To get permissions, you must request access by the link
                    raise in the Permission Error when you first try to run. Each language will have it own ASR moddel.

            cer_bound : float
                The base CER threshold (upper bound) for matching algorithm. Defaults to 0.35

            gpu_id : (Optional[int])
                ID of the GPU to use (None if no GPU). Defaults to 0.

            batch_size : int
                The number of samples to process in a batch.

            audio_sentence_map : pd.DataFrame
                Sometimes one might only have test.tsv from MCV dataset (no NeMo format test_manifest). Thus, to combine
                the original `text` s from the MCV dataset, to then benchmark the vac_aligner, we need the dataframe
                with original sentences (to merge into single long transcript)
        """
        os.makedirs(target_base, exist_ok=True)
        os.makedirs(os.path.dirname(manifest_file), exist_ok=True)

        model_name = ASR_MAPPING[language]
        ASR.check_hf_credential(model_name, hf_token)

        asr_config = {
            'gpu_id': gpu_id,
            'hf_token': hf_token,
            'batch_size': batch_size,

        }

        aligner_config = {
            'use_id': False,
            'transcript_path': None,
            'target_base': target_base,
            'manifest_file': manifest_file,
            'ending_punctuations': ending_punctuations,
            'save_manifest_path': manifest_file.replace(".json", "_vac.json")
        }

        vac = cls(language, asr_config, aligner_config, init_aligner_after_asr=init_aligner_after_asr)
        print("Initialized VAC!")
        asr_payload = dict(
            save_dir=target_base,
            save_manifest=manifest_file,
            audio_sentence_map=audio_sentence_map
        )
        if asr_input_file.endswith(".json"):
            asr_payload['wav_files'] = None
            asr_payload['test_manifest'] = asr_input_file
        else:
            asr_payload['wav_files'] = asr_input_file

        print("ASR Payload")
        vac.run_asr(**asr_payload)
        print("Finished ASR inference!")

        vac.align_match(cer_bound)
        print("Finished Matching the texts. Now, performing the Benchmark")
        benchmark = Benchmark(target_base, manifest_file)
        stats = benchmark.get_benchmark()
        benchmark.analyze_and_save_benchmark(stats, os.path.join(target_base, "benchmark.csv"))
        return stats


if __name__ == '__main__':
    VAC.run_end2end(manifest_file=r"D:\CapstoneThesisArmASR\vac_aligner\experiments\end2end\manifest.json", batch_size=64,
                    asr_input_file=r"D:\CapstoneThesisArmASR\vac_aligner\vac_aligner\data\matching\test_with_predictions17_replaced.json",
                    target_base=r"D:\CapstoneThesisArmASR\vac_aligner\experiments\end2end\texts", init_aligner_after_asr=True)
