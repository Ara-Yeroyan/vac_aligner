import os

from .sdp import (
    Subprocess,
    RenameFields,
    SplitAudioFile,
    GetRttmSegments,
    PreserveByValue,
    KeepOnlySpecifiedFields
)


def detect_speech(input_manifest_file: str, output_manifest_file: str = "vad", **kwargs):
    args = dict(
        cmd="python vad/asr_with_vad.py",
        output_manifest_arg="output_dir",
        input_manifest_arg="manifest_filepath",
        input_manifest_file=input_manifest_file,
        output_manifest_file=output_manifest_file,
    )
    args.update(kwargs)
    processor = Subprocess(**args)
    processor.test()
    print("VAD Processor has successfully passed the tests!")
    processor.process()


def postprocess_vad(
        input_manifest_file: str = "vad/temp_manifest_vad_rttm-onset0.3-offset0.3-pad_onset0.2-pad_offset0.2"
                                   "-min_duration_on0.2-min_duration_off0.2-filter_speech_firstTrue.json",
        output_manifest_file: str = "vad/renamed_manifest.json", **kwargs):

    args = dict(input_manifest_file=input_manifest_file,
                output_manifest_file=output_manifest_file,
                rename_fields={"audio_filepath": "source_filepath"})
    args.update(kwargs)
    processor = RenameFields(**args)
    processor.test()
    print("Rename Processor has successfully passed the tests!")
    processor.process()


def segment(output_manifest_file: str, **kwargs):
    args = dict(
        rttm_key='rttm_file',
        duration_threshold=15,
        duration_key='duration',
        output_file_key='audio_segments',
        output_manifest_file=output_manifest_file
    )
    args.update(kwargs)
    processor = GetRttmSegments(**args)
    processor.test()
    print("Segment Processor has successfully passed the tests!")
    processor.process()


def cut_audios(splited_audio_dir: str, output_manifest_file: str, **kwargs):
    args = dict(
        splited_audio_dir=splited_audio_dir,
        segments_key='audio_segments',
        duration_key='duration',
        input_file_key='source_filepath',
        output_file_key='audio_filepath',
        output_manifest_file=output_manifest_file
    )
    args.update(kwargs)
    print(args)
    processor = SplitAudioFile(**args)
    processor.test()
    print("Splitter Processor has successfully passed the tests!")
    processor.process()


def filter_audios(input_manifest_file: str, output_manifest_file: str, final_manifest: str):
    args_preserve = dict(
        input_manifest_file=input_manifest_file,
        output_manifest_file=output_manifest_file,
        input_value_key='duration',
        operator='gt',
        target_value=0.0
    )
    processor = PreserveByValue(**args_preserve)
    processor.test()
    print("PreserveByValue Processor has successfully passed the tests!")
    processor.process()

    processor = KeepOnlySpecifiedFields(
        input_manifest_file=output_manifest_file,
        output_manifest_file=final_manifest,
        fields_to_keep=["audio_filepath", "duration", "source_filepath"]
    )
    processor.test()
    print("KeepOnlySpecifiedFields Processor has successfully passed the tests!")
    processor.process()


def run_vad(input_manifest_file: str, output_manifest_file: str = "vad",
            input_manifest_file_rename: str = "temp_manifest_vad_rttm-onset0.3-offset0.3-pad_onset0.2-pad_offset0.2"
                                              "-min_duration_on0.2-min_duration_off0.2-filter_speech_firstTrue.json",
            output_manifest_file_rename: str = "renamed_manifest.json", **kwargs) -> str:

    path_to_save_audios = output_manifest_file
    output_manifest_file = os.path.join(output_manifest_file, "vad_artifacts")
    if os.path.exists(output_manifest_file):
        return os.path.join(path_to_save_audios, "final_manifest.json")
    detect_speech(input_manifest_file, output_manifest_file, **kwargs)
    print("Finished VAD stage 1")
    input_manifest_file = os.path.join(output_manifest_file, input_manifest_file_rename)
    output_manifest_file = os.path.join(output_manifest_file, output_manifest_file_rename)
    postprocess_vad(input_manifest_file, output_manifest_file)
    print("Finished VAD stage 2")
    output_manifest_segmented = os.path.join(os.path.dirname(output_manifest_file), "vad_segmented.json")
    segment(output_manifest_segmented, input_manifest_file=output_manifest_file)
    print("Finished VAD stage 3")
    splited_audio_dir = os.path.join(path_to_save_audios, "splited_audios")
    output_manifest_file = os.path.join(path_to_save_audios, "splited_audios.json")
    cut_audios(splited_audio_dir, output_manifest_file, input_manifest_file=output_manifest_segmented)
    print("Finished VAD stage 4")
    output_manifest_file_filter = os.path.join(path_to_save_audios, "filtered_audios.json")
    final_manifest = os.path.join(os.path.dirname(output_manifest_file_filter), "final_manifest.json")
    filter_audios(output_manifest_file, output_manifest_file=output_manifest_file_filter, final_manifest=final_manifest)
    print("Finished the FINAL VAD stage 5")
    return final_manifest
