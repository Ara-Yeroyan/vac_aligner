"""
VAC - VAD-ASR-CER (Matching)

This is the VAC pipeline's last step - CER based matching!
==========================================================

A subpackage for both matching the chunked texts with single, long transcript and benchmarking (if labels are available)

"""

from .align import BaseAlignerVAC, ArmenianAlignerVAC
from .benchmark_on_mcv import Benchmark

ALIGNER_MAPPING = {
    "hy": ArmenianAlignerVAC
}