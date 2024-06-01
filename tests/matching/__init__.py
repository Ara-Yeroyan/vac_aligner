"""
VAC - VAD-ASR-CER (Matching)

This is the VAC pipeline's last step - CER based matching!
==========================================================

A subpackage for both matching the chunked texts with single, long transcript and benchmarking (if labels are available)

"""

from vac_aligner.matching.align import ArmenianAlignerVAC


ALIGNER_MAPPING = {
    "hy": ArmenianAlignerVAC
}