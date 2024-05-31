import pandas as pd

from .factory import VAC


def run_pipeline(**kwargs) -> pd.DataFrame:
    return VAC.run_end2end(**kwargs)


if __name__ == '__main__':
    run_pipeline(manifest_file= r"D:\CapstoneThesisArmASR\vac_aligner\experiments\end2end\manifest.json", batch_size=64,
    asr_input_file=r"D:\CapstoneThesisArmASR\vac_aligner\vac_aligner\data\matching\test_with_predictions17_replaced.json",
    target_base=r"D:\CapstoneThesisArmASR\vac_aligner\experiments\end2end\texts", init_aligner_after_asr=True)