import multiprocessing
import os
import sys
import json
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


import regex as re
from tqdm import tqdm
from loguru import logger

from .cutting import cut_extra_tokens_from_match
from vac_aligner.utils import create_nested_folders


data_item = Tuple[str, str, float]
log_format = "{time:HH:mm:ss} | {level} | {name}_{function}:{line} | {message}"
logger.remove()  # Remove default handler
logger.add(sys.stdout, format=log_format)


class BaseAlignerVAC(ABC):
    def __init__(self, reference_text: str, chunks: List[Tuple], predictions_manifest_path: str,
                 target_base: Optional[str] = None, lost_search_cer: float = 0.5,
                 shift_back_minimum_bound: int = 8, shift_back_indicator: int = 4,
                 search_segment_length_upper_bound: int = 40, reduce_long_search_segment: int = 30):
        """
        Initializes the AlignerVAC* instance with necessary parameters to align audio chunks
        with a reference text using predicted ASR transcriptions.

        AlignerVAC* - operates within the VAC (VAD -> ASR -> CER-Matching) framework proposed in <>


        Parameters
        ----------
        reference_text : str
            The original long string to compare against, typically a combined transcript.
        chunks : List[Tuple[str, str, float]]
            A list of tuples in a format similar to NeMo, where each tuple contains:
                [audio_filepath, pred_text (ASR inference), audio_duration].
        predictions_manifest_path : str
            Path to a NeMo format manifest (json) where predictions are stored.
            Each file's row (dict) has the format:
                [audio_filepath, text, pred_text (ASR inference), audio_duration].
        target_base : Optional[str], optional
            The directory where matched texts will be stored as a single .txt per audio file.
            If not specified, defaults to a subdirectory 'chunk_texts_mcv' under predictions_manifest_path.
        lost_search_cer : float, optional
            Character Error Rate (CER) threshold indicating that the alignment is lost
            if sequentially exceeded. Default is 0.5.
        shift_back_minimum_bound : int, optional
            The minimum number of characters to backtrack when handling sequentially problematic chunks
            to resume search for the next chunk. Default is 8.
        shift_back_indicator : int, optional
            Number of consecutive problematic search runs required before initiating a shift back. Default is 4.
        search_segment_length_upper_bound : int, optional
            The maximum allowable search segment length, beyond which the cut_from_both_ends method
            might fail to handle extra tokens. Default is 40.
        reduce_long_search_segment : int, optional
            The number of characters to trim from the ends of a long search segment during processing. Default is 30.

        """
        self._aca_var = 0
        self.target_base = target_base
        self.lost_search_cer = lost_search_cer
        self.shift_back_minimum_bound = shift_back_minimum_bound
        self.shift_back_indicator = shift_back_indicator
        self.search_segment_length_upper_bound = search_segment_length_upper_bound
        self.reduce_long_search_segment = reduce_long_search_segment

        self.output_matches = []
        self.memory = [chunks, reference_text, predictions_manifest_path]
        self.restore_memory()

    @property
    @abstractmethod
    def language_id(self) -> str:
        """Identifier for the language (e.g. Armenian, Georgian, etc.)"""
        ...

    def restore_memory(self, idx_for_ref_text: int = 0):
        """Used to restore original Texts and Chunks to (re)run the algorithm"""
        self.chunks, self.reference_text, self.manifest_path = self.memory
        self.cut_the_header()

        self.durs = []
        self.matches = []

        self.shift_back = 0
        self.current_time = 0
        self.search_start_pos = 0  # Start searching from the beginning of the original text

        self.previous_pack = None
        self.try_combinations = False
        self.sequential_big_mismatches = []  # store metadata (chunks) of sequential mismatches (CER > 0.5)
        self.cer_cum = [0 for _ in range(8)]  # we terminate the algorithm if 15times in a row CER is 0.5+

    @property
    def default_ellipsis(self):
        """A punctuation harming the overall CER which can be ignored if ASR failed to predict them"""
        return "..."

    @default_ellipsis.setter
    def default_ellipsis(self, value):
        """
        Sets the default ellipsis value with the provided string.
        Ensures that the input is a string and exactly three periods (language specific), else raises a ValueError.

        Parameters:
        value (str): A string to set as the new default ellipsis.
        """
        if isinstance(value, str) and len(value) == 3:
            self._default_ellipsis = value
        else:
            raise ValueError("The default ellipsis must be a string containing exactly three periods.")

    def cut_the_header(self):
        if len(self.reference_text.split(" ")[0]) == 1 or len(self.reference_text.split("\n")[0]) == 1:
            self.reference_text = self.reference_text[2:]
            logger.info(f'Cut the Header: {self.chunks[0]}')
            self.chunks = self.chunks[1:]

    def get_target_path(self, original_path: str) -> str:
        if "\\" in original_path:
            audio_name = original_path.split("\\")[-1]
        else:
            audio_name = original_path.split("/")[-1]

        if self.target_base:
            target_path = os.path.join(self.target_base, audio_name.replace(".wav", "_matched.txt"))
        else:
            target_path = os.path.join(os.path.dirname(self.manifest_path), "chunk_texts_mcv",
                                       audio_name.replace(".wav", "_matched.txt"))
        return target_path

    def dump_match_info(self, target_path: str, best_match_text: str, match_info: dict,
                        lock: Optional[multiprocessing.Lock] = None):
        """Writing the matched text as .txt and adding the record line to manifest"""
        create_nested_folders(target_path)
        if lock is None:
            class Lock:
                def acquire(self):
                    ...

                def release(self):
                    ...
            lock = Lock()
        lock.acquire()
        try:
            with open(target_path, "w", encoding='utf-8') as f:
                f.write(best_match_text.strip())

            if not os.path.exists(self.manifest_path):
                os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
                with open(self.manifest_path, "w", encoding='utf-8') as f:
                    f.write(json.dumps(match_info, ensure_ascii=False) + "\n")
            else:
                with open(self.manifest_path, "a", encoding='utf-8') as f:
                    f.write(json.dumps(match_info, ensure_ascii=False) + "\n")
        finally:
            lock.release()

    def handle_mismatches(self) -> Tuple[Tuple[int, int], str, str, bool]:
        """Handle sequential big mismatches by coming to the past, and taking the first best output"""
        sequential_big_mismatches = sorted(self.sequential_big_mismatches, key=lambda x: x[2])
        window_text, chunk_text, cer, start_pos, end_pos, _ = sequential_big_mismatches[0]
        best_cer = cer
        best_match_range = (start_pos, end_pos)
        best_match_text = window_text
        found_acceptable_match = True  # after handling this problematic chunk need to proceed to the next chunk
        self.shift_back = min(len(window_text) // 2, self.shift_back_minimum_bound)
        tokens_ = re.findall(r'\S+|\s+', window_text)
        if not tokens_:
            logger.warning(sequential_big_mismatches)
            raise ValueError

        if len(tokens_[-1]) <= 2:  # punctuation
            if len(" ".join(tokens_[-2:])) > self.shift_back:  # get back from whole token, not in the middle
                self.shift_back = len(" ".join(tokens_[-2:])) + 1
        else:
            if len(" ".join(tokens_[-1])) > self.shift_back:  # get back from whole token, not in the middle
                self.shift_back = len(" ".join(tokens_[-1])) + 1

        self.search_start_pos = end_pos - self.shift_back
        self.previous_pack = [len(window_text), best_cer]
        self.try_combinations = True
        # self.shift_back = 0
        window_text, chunk_text, cer, start_pos, end_pos, sources = sequential_big_mismatches[0]
        logger.warning(f"Potentially problematic: {[cer, window_text, chunk_text, start_pos, end_pos, sources]}")
        return best_match_range, chunk_text, best_match_text, found_acceptable_match


    @property
    @abstractmethod
    def ending_punctuations(self) -> str:
        """Language Specific sentence ending punctuations such as ։․,"""
        pass

    @staticmethod
    @abstractmethod
    def language_specific_cleaning(chunk: str):
        """Language Specific text cleaning should be implemented here"""
        return chunk

    @abstractmethod
    def language_specific_postprocessing(self, best_match_text: str, best_match_range: Tuple[int, int]) -> Tuple[str, Tuple[int, int]]:
        """Based on the language, some specific grammatical rules might be needed"""
        return best_match_text, best_match_range

    def update_window(self, window_text: str, chunk_text: str, start_pos: int, end_pos: int):
        """Need to Update the Rolling Window over Source Transcript (if it is >> chunk_text)"""
        new_window_text, cer, to_shift = cut_extra_tokens_from_match(window_text,
                                                                     chunk_text,
                                                                     self.try_combinations)

        clen = len(chunk_text)
        wlen = len(window_text)
        if (clen * 6 < len(window_text)) or (clen < self.search_segment_length_upper_bound < wlen):
            longer_window = self.reference_text[start_pos:end_pos+10]

            logger.debug("Extra length difference between chunk text and source text")
            c_text = chunk_text
            w_text = window_text
            smaller_window, cer2, shift2 = cut_extra_tokens_from_match(window_text[self.reduce_long_search_segment:],
                                                                       chunk_text,
                                                                       self.try_combinations)

            smaller_window3, cer3, shift3 = cut_extra_tokens_from_match(longer_window,
                                                                       chunk_text,
                                                                       self.try_combinations)

            if cer3 <= cer2:
                smaller_window, cer2, shift2 = smaller_window3, cer3, shift3

            if cer2 <= cer:
                start_pos += wlen - self.reduce_long_search_segment  # adjust starting position as cut first 30chars
                new_window_text, cer, to_shift = smaller_window, cer2, shift2

            logger.debug(f"Extra with solved  CER: {cer}!\nchunk_text: {c_text}\n window_text: {w_text}")

        return new_window_text, cer, to_shift, start_pos

    def clean_chunk_text(self, chunk_text: str) -> str:
        chunk_text = chunk_text.replace(" - ", " ").replace(" -", " ").replace("- ", " ")
        chunk_text = self.language_specific_cleaning(chunk_text)
        chunk_text = re.sub("\s+", " ", chunk_text)
        return chunk_text

    def check_lost_scenario(self):
        if all([x > self.lost_search_cer for x in self.cer_cum]):
            logger.warning("transcripts got lost the matching, no point to continue")
            return self.matches

    def parse_chunk_and_prepare_iter(self, chunk: data_item) -> Tuple[Tuple, Tuple]:
        try:
            sources = None
            original_path, chunk_text, chunk_duration = chunk
        except:
            original_path, chunk_text, chunk_duration, sources = chunk
        chunk_text = self.clean_chunk_text(chunk_text)
        target_path = self.get_target_path(original_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        best_cer = float('inf')
        best_match_range = (0, 0)
        best_match_text = ""
        found_acceptable_match = False
        self.sequential_big_mismatches = []

        # if self.previous_pack:  # previous chunk was added if len(sequential_big_mismatches) > 4:
        #     if self.previous_pack[-1] > 0.3:
        #         self.shift_back += self.previous_pack[0]
        #         self.search_start_pos -= self.previous_pack[0]
        #         self.search_start_pos = max(self.search_start_pos, 0)
        #         self.try_combinations = True

        return ((original_path, target_path, chunk_text, chunk_duration, sources),
                (best_cer, best_match_range, best_match_text, found_acceptable_match))

    def align(self, cer_threshold=0.35, lock: Optional[multiprocessing.Lock] = None):

        if lock is not None:
            import contextvars
            from loguru import logger

            logger_source_name = contextvars.ContextVar("logger_source_name", default="default_id")
            log_format = "{time:HH:mm:ss} | {level} | {message} | ID: {extra[id]}"
            logger.remove()  # Remove default handler
            logger.add(sys.stdout, format=log_format)
            logger = logger.patch(lambda record: record["extra"].update(id=logger_source_name.get()))

            def set_log_id(new_id):
                logger_source_name.set(new_id)
        try:
            prev_source_name = None
            for ind, chunk in tqdm(enumerate(self.chunks)):
                lost = self.check_lost_scenario()
                if lost:
                    return lost

                chunk_metadata, init_states = self.parse_chunk_and_prepare_iter(chunk)
                original_path, target_path, chunk_text, chunk_duration, sources = chunk_metadata
                best_cer, best_match_range, best_match_text, found_acceptable_match = init_states

                if sources:
                    if lock is not None:
                        logger_id = sources[0].split("/")[-1].split("\\")[-1]
                        set_log_id(logger_id)

                    if prev_source_name is None:
                        prev_source_name = sources[0]
                    else:
                        if sources[0] != prev_source_name:
                            logger.info(f"The source audio changed from {prev_source_name} -- to -- {sources[0]}")
                            self.memory = (self.chunks[ind:], sources[1], self.manifest_path)
                            self.restore_memory()
                            res = self.align()
                            if not res:
                                logger.warning(f"Got empty alignment for source: {prev_source_name} -> {sources[0]}")
                                return
                            self.output_matches.extend(res)
                            return res

                for start_pos in range(self.search_start_pos, self.search_start_pos+len(chunk_text)*4):
                    if len(self.sequential_big_mismatches) > self.shift_back_indicator:
                        best_match_range, chunk_text, best_match_text, found_acceptable_match = self.handle_mismatches()
                        break

                    to_add = 0
                    end_pos = start_pos + len(chunk_text) + 6 + self.shift_back
                    window_text = self.reference_text[start_pos:end_pos]
                    if self.default_ellipsis in window_text and self.default_ellipsis not in chunk_text:
                        window_text = window_text.replace(self.default_ellipsis, "")
                        to_add = 3  # as ignored 3chars from source segment

                    new_window_text, cer, to_shift, start_pos = self.update_window(window_text, chunk_text, start_pos, end_pos)
                    start_pos += to_shift
                    window_text = new_window_text
                    # end_pos - (len(window_text) - len(new_window_text))
                    end_pos = start_pos + len(window_text) + to_add

                    if cer < best_cer:
                        best_cer = cer
                        best_match_range = (start_pos, end_pos)
                        best_match_text = window_text

                    if cer <= cer_threshold:  # Break if CER is below the threshold, indicating a good enough match
                        self.previous_pack = None
                        found_acceptable_match = True
                        self.search_start_pos = end_pos
                        break

                    else:
                        self.sequential_big_mismatches.append(
                            [new_window_text, chunk_text, cer, start_pos, end_pos, ind - 1])
                        self.try_combinations = False
                        self.shift_back = 0

                else:
                    logger.info(f"Failed to find a match for chunk: {chunk_text}")
                    self.search_start_pos += len(chunk_text)//4  # adjust search start position conservatively

                best_match_text, best_match_range = self.language_specific_postprocessing(best_match_text,
                                                                                          best_match_range)

                match_info = {
                    'audio_filepath': original_path,
                    'matched_text_path': target_path,
                    'text': best_match_text.strip(),
                    'chunk_text': chunk_text,
                    'start_time': self.current_time,
                    'end_time': self.current_time + chunk_duration,
                    'best_match_range': best_match_range,
                    'cer': best_cer,
                    'duration': chunk_duration,
                }
                if sources:
                    match_info.update({'source_audio': sources[0], 'source_text': sources[1]})

                self.cer_cum = self.cer_cum[1:] + [best_cer]
                self.matches.append(match_info)
                self.durs.append(len(self.sequential_big_mismatches))
                self.current_time += chunk_duration
                self.dump_match_info(target_path, best_match_text, match_info, lock)

        except KeyboardInterrupt:
            logger.error(self.matches)

        self.output_matches.extend(self.matches)
        return self.output_matches

    @classmethod
    def from_nemo_manifest(cls, manifest_file: str,  save_manifest_path: str, transcript_path: Optional[str] = None,
                           use_id: bool = False, target_base: Optional[str] = None, ending_punctuations: str = '․,։',
                           vad_manifest: Optional[str] = None):
        """
        Create an instance by reading and processing a NEMO manifest and a transcript file.

        Parameters:
            manifest_file : str
                The path to the NEMO manifest file. Each line in this file should be a valid JSON string representing
                    individual audio data with keys [audio_filepath, pred_text, duration]

            transcript_path : Optional[str]
                The path to the file containing the combined transcript of all audio files.
                If None, will calculate (combine) based on the "text" key in the manifest_file.

            save_manifest_path : str
                The path where the processed manifest data will be saved or utilized.

            use_id : bool, optional
                Determines whether to sort the audio chunks based on their ID. Useful when IDs are used to denote the
                    order of audio files/chunks. If provided, then manifest_file should have id field. Default is False.

            target_base : Optional[str], optional
                The base path that could be prefixed to each audio file path in the manifest, useful for cases where
                    the manifest paths are relative. Default is None.

            ending_punctuations : Optional[str]
                The punctuation symbols accumulated in single string (with most common sentence ending punctuation
                placed in the end of the string - such as: "...,;:.". Only necessary if transcript_path is None so that
                we combine the texts into a single transcript. Default is example for Armenian Language - '․,։'

        Returns:
            An instance of the class populated with the combined transcript, sorted (if specified) and filtered chunks

        Example:
            # Create a new instance using a manifest file, a transcript, and specify the use of IDs
            instance = ClassName.from_nemo_manifest('path/to/manifest.json', 'path/to/transcript.txt',
                                                    'path/to/save/manifest', target_base="abs_path_cwc", use_id=True)
        """
        create_nested_folders(target_base)
        create_nested_folders(save_manifest_path)

        get_source_text = lambda x: None
        if vad_manifest is not None:
            combined_transcript = ""
            path2text = {}
            with open(vad_manifest, "r", encoding='utf-8') as f:
               for line in f:
                   item = json.loads(line)
                   path2text[item['audio_filepath']] = item['text']
                   combined_transcript += item['text']

            def get_source_text(source_audio_path: str):
                return path2text[source_audio_path]

            print("Successfully created a combined transcript from the VAD manifest!")
            with open("combined_transcript.txt", "w", encoding='utf-8') as f:
                f.write(combined_transcript)
        elif transcript_path is None:
            combined_transcript = ""
        else:
            with open(transcript_path, "r", encoding='utf-8') as f:
                combined_transcript = f.read()

        chunks = []
        with open(manifest_file, "r", encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                sources = (item['source_audio'], get_source_text(item['source_audio']))
                chunk = (item["audio_filepath"], item["pred_text"], item["duration"], sources, item.get("id"))
                chunks.append(tuple(chunk))
                if transcript_path is None:
                    if not vad_manifest:
                        text = item["text"]
                        if text[-1] not in ":,." + ending_punctuations:
                            text = text + (ending_punctuations[-1] or "։")
                        else:
                            text = text + " "
                        combined_transcript += text

        if use_id:
            chunks = sorted(chunks, key=lambda x: x[-1])
        chunks = [tuple(chunk[:-1]) for chunk in chunks]
        print(f"Found #{len(chunks)}chunks")
        return cls(combined_transcript, chunks, save_manifest_path, target_base=target_base)

    @staticmethod
    def combine_transcript(predictions_manifest: str, output_file: str, ending_punctuations: str) \
            -> Tuple[List[data_item], str]:
        chunks = []
        combined_transcript = ""
        with open(fr"{predictions_manifest}", "r", encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                chunks.append((item["audio_filepath"], item["pred_text"], item["duration"]))
                text = item["text"]
                if text[-1] not in ":,." + ending_punctuations:
                    text = text + (ending_punctuations[-1] or "։")
                else:
                    text = text + " "
                combined_transcript += text

        with open(fr"{output_file}", "w", encoding='utf-8') as t:
            t.write(combined_transcript)

        return chunks, combined_transcript


class ArmenianAlignerVAC(BaseAlignerVAC):
    """Aligner for Armenian Language"""

    @property
    def language_id(self):
        return "hy"

    @property
    def ending_punctuations(self) -> str:
        return '․,։'

    @staticmethod
    def language_specific_cleaning(chunk_text: str):
        """Language Specific text cleaning should be implemented here"""
        return chunk_text.replace("Ե ՛վ", "Եվ")

    def language_specific_postprocessing(self, best_match_text: str, best_match_range: Tuple[int, int]) -> Tuple[str, Tuple[int, int]]:
        """Based on the language, some specific grammatical rules might be needed"""
        if best_match_text.startswith("ւ"):  # specific case for Armenian letter "ու"
            if self.reference_text[best_match_range[0] - 1] == "ո":
                best_match_text = "ո" + best_match_text
                best_match_range = (best_match_range[0] - 1, best_match_range[1])

        return best_match_text, best_match_range


class GeorgianAlignerVAC(BaseAlignerVAC):
    """Aligner for Armenian Language"""

    @property
    def language_id(self):
        return "ka"

    @property
    def ending_punctuations(self) -> str:
        return '.,:჻'

    @staticmethod
    def language_specific_cleaning(chunk_text: str):
        """Language Specific text cleaning should be implemented here"""
        return chunk_text

    def language_specific_postprocessing(self, best_match_text: str, best_match_range: Tuple[int, int]) -> Tuple[str, Tuple[int, int]]:
        """Based on the language, some specific grammatical rules might be needed"""
        return best_match_text, best_match_range


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process the predictions manifest and output a combined transcript.")
    parser.add_argument("predictions_manifest", type=str, help="Path to the manifest file storing predictions.",
                        default=r"D:\CapstoneThesisArmASR\data\cv-corpus-17.0-2024-03-15\hy-AM\test\vac_mcv_manifest.json")
    parser.add_argument("combined_transcript", type=str, help="Combined MCV transcript.")
    args = parser.parse_args()

    # matches_sorted = ArmenianAlignerVAC(args.combined_transcript, chunks, args.predictions_manifest).align(0.35)
