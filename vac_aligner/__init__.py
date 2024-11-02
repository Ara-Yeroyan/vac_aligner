"""
VAC - VAD-ASR-CER (Matching) Pipeline package

1. VAD: Voice Activity Detection
2. ASR: Automatic Speech Recognition
3. CER: Character Error Rate

**Usage***

VAD splits long audio into many multi-second chunks
ASR runs inference and gets predictions per chunk (chunk texts)
CER-Matching uses our algorithm to match the predicted text (with errors) to ground truth transcript

**Results**

Having multi hour long audio and texts, you can obtain many multi-second audio chunks and corresponding texts (98% acc)

"""

__author__ = """Ara Yeroyan"""
__email__ = 'ar23yeroyan@gmail.com'
__version__ = '0.2.0'

try:
    from .vac_pipeline import run_pipeline
except (ImportError, ModuleNotFoundError) as e:
    import os
    print(e.__str__())
    if not os.environ.get("NOT_IGNORE_IMPORTS"):
        raise e
    from vac_pipeline import run_pipeline
