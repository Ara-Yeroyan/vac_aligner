import json
import multiprocessing
import os
from collections import defaultdict
from dataclasses import asdict
from typing import Optional, Tuple, Union, List, Dict, TypedDict, DefaultDict

import pandas as pd
from loguru import logger

from .vad import detect_speech, run_vad
from .dto import ASRConfig, ALIGNERConfig
from .utils import _validate_asr_info, _validate_aligner_info, add_durations

from vac_aligner.asr import ASR, ASR_MAPPING
from vac_aligner.matching import ALIGNER_MAPPING, BaseAlignerVAC, Benchmark


def align_single_raw_audio(aligner_copy: BaseAlignerVAC, chunks: List, lock_: multiprocessing.Lock, cer_bound: float):
    aligner_copy.chunks = chunks
    aligner_copy.align(cer_bound, lock_)


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

    def init_aligner(self, vad_manifest: Optional[str] = None):
        if isinstance(self.aligner_matching_info, dict):
            self.aligner_matching_info = ALIGNERConfig(**self.aligner_matching_info)
        self.aligner_class = self._get_aligner_class(self.language)
        self.aligner: BaseAlignerVAC = self.aligner_class.from_nemo_manifest(**asdict(self.aligner_matching_info),
                                                                             vad_manifest=vad_manifest)

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
                audio_sentence_map: Optional[pd.DataFrame] = None, vad_manifest: Optional[str] = None):

        chunk_data = self.asr_model.run(
            save_dir=save_dir,
            wav_files=wav_files,
            save_manifest=save_manifest,
            test_manifest=test_manifest,
            audio_sentence_map=audio_sentence_map
        )

        if self.init_aligner_after_asr:
            print("Initializing Aligner with the ASR manifest")
            self.init_aligner(vad_manifest=vad_manifest)

        return chunk_data

    def align_match(self, cer_bound: float = 0.35, lock: Optional[multiprocessing.Lock] = None) -> List:
        return self.aligner.align(cer_bound, lock)

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
    def run_end2end(cls, manifest_file: Optional[str], asr_input_file: str, language: str = "hy", init_aligner_after_asr = False,
                    target_base: Optional[str] = None, ending_punctuations: Optional[str] = '․,։', hf_token: str = '',
                    cer_bound: float = 0.4,  batch_size: int = 15, gpu_id: Optional[int] = 0,
                    audio_sentence_map: Optional[pd.DataFrame] = None,
                    vad_manifest: Optional[str] = None, use_multiprocessing: bool = False) -> pd.DataFrame:
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
        add_durations(vad_manifest)
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
        vad_manifest_file = None
        if vad_manifest:
            # chunks, combined_transcript = vac.aligner_class.combine_transcript(vad_manifest, "combined_transcript.txt")
            vad_manifest_file = run_vad(vad_manifest, output_manifest_file=asr_input_file)
            logger.success("Finished the VAD")
            logger.info(f"Created {len(os.listdir(asr_input_file))} chunks post VAD")

        print("Initialized VAC!")
        asr_payload = dict(
            save_dir=target_base,
            save_manifest=manifest_file,
            audio_sentence_map=audio_sentence_map
        )

        if asr_input_file.endswith(".json"):
            asr_payload['wav_files'] = None
            asr_payload['test_manifest'] = asr_input_file
        elif vad_manifest_file is not None:
            asr_payload['wav_files'] = None
            asr_payload['vad_manifest'] = vad_manifest
            asr_payload['test_manifest'] = vad_manifest_file
        else:
            asr_payload['wav_files'] = asr_input_file

        logger.info("Running ASR")
        vac.run_asr(**asr_payload)
        logger.success("Finished ASR inference!")

        if use_multiprocessing or os.environ.get("ENABLE_MULTIPROCESSING", False):
            logger.debug("Seting up Multi Processing Environment")
            from copy import deepcopy

            meta_chunks = defaultdict(dict)
            for chunk in vac.aligner.chunks:
                try:
                    original_path, chunk_text, chunk_duration = chunk
                    continue
                except:
                    original_path, chunk_text, chunk_duration, sources = chunk
                meta_chunks[sources[0]]["chunks"] = meta_chunks[sources[0]].get("chunks", []) + [chunk]
                meta_chunks[sources[0]]["source_text"] = sources[1]
            logger.debug(f"Created {len(meta_chunks)} partitions based on original number of audios (sources)")

            aligner_matching_info = _validate_aligner_info(aligner_config) if aligner_config else None
            if isinstance(aligner_matching_info, dict):
                aligner_matching_info = ALIGNERConfig(**aligner_matching_info)
            aligner_class = vac._get_aligner_class(language)

            lock = multiprocessing.Lock()
            processes = []
            for chunk_pair in meta_chunks.values():
                chunks = chunk_pair['chunks']
                source_text = chunk_pair['source_text']

                aligner: BaseAlignerVAC = aligner_class.from_nemo_manifest(**asdict(aligner_matching_info),
                                                                           vad_manifest=vad_manifest)
                aligner.reference_text = source_text
                aligner.cut_the_header()
                aligner_copy = deepcopy(aligner)
                p = multiprocessing.Process(target=align_single_raw_audio, args=(aligner_copy, chunks, lock, cer_bound))
                processes.append(p)
                p.start()

            logger.debug("Final Process end result Join")
            for p in processes:
                p.join()
        else:
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
