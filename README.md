[![PyPI Version](https://img.shields.io/pypi/v/vac_aligner.svg)](https://pypi.python.org/pypi/vac_aligner) **vac_aligner**
[![Documentation Status](https://readthedocs.org/projects/vac_aligner/badge/?version=latest)](https://vac_aligner.readthedocs.io/en/latest/)


# VAC - VAD-ASR-CER (Matching) Pipeline

Comprehensive pipeline designed for processing long audio recordings through three main stages: Voice Activity Detection (VAD), Automatic Speech Recognition (ASR), and Character Error Rate (CER) Matching. This pipeline is ideal for improving speech recognition models and training Text-to-Speech (TTS) systems with high accuracy.

- Free software: Apache Software License 2.0
- Documentation: [https://vac-aligner.readthedocs.io](https://vac-aligner.readthedocs.io)

## Components

1. **VAD (Voice Activity Detection)**: Splits long audio into manageable multi-second chunks suitable for further processing.
2. **ASR (Automatic Speech Recognition)**: Performs inference on each audio chunk to generate text predictions.
3. **CER (Character Error Rate) Matching**: Aligns the predicted text from ASR with the ground truth transcript to correct errors and ensure high accuracy.


## Installation

To install `vac_aligner`, run this command in your terminal:

```bash
pip install vac_aligner['full']
```

This is the preferred method to install vac_aligner, as it will always install the most recent stable release.

The 2nd part of our pipeline requires  `nemo-toolkit['asr']` which can make the library very heavy,
especially, for GPU compatible torch version. Thus, you can skip this **extra** installations.
```bash
pip install vac_aligner
```

Now you can use VAD or the Matching Part (say you have your own ASR model
or predictions) while skipping the `torch` and `nemo` installations!

## Usage

The pipeline processes multi-hour long audio and texts to produce many multi-second audio chunks and corresponding texts with an accuracy of 97%. This high accuracy is achieved through sophisticated matching algorithms that correct common ASR errors such as repeated characters, incomplete words, and incorrect word predictions.

There are a couple of ways the library can be used. One can use the full functionality by

```python
from vac_aligner import run_pipeline

run_pipeline(
   manifest_file="path/to/save/manifest.json", batch_size=64,
   asr_input_file="path/to/manifest.json", # or path to folder, containing audio chunks (.wav)s
   target_base="path/where/to/save/artifacts", # otherwise, will use `asr_input_file`
   init_aligner_after_asr=True  # If there is no long transcript available and you need to extract it
)
```

There are many scenarios where one might need a partial functionality of the pipeline. Then we can use the classes directly.

### Scenario 1

Need to perform ASR over short chunks and store in `nemo_manifest`

```python
import os

from vac_aligner.dto import ASRConfig
from vac_aligner.asr import ASR_MAPPING, ASR

language = "hy"  #  Armenain
model_name = ASR_MAPPING[language] # "Yeroyan/stt_arm_conformer_ctc_large"
asr_config = ASRConfig(hf_token=os.environ['HF_TOKEN'], batch_size=24)
asr = ASR(model_name, asr_config)
asr.run(
    save_dir="where/to/save/asr/predictions", # .txt(s)
    save_manifest="path/to/save/manifest.json",
    wav_files="path/to/wav/files",
    test_manifest='path/to/manifest/with/predictions.json'
)

```

### Scenario 2

You already have predictions manifest and/or long transcript, and want to run the Matching to obtain the correct
chunk texts to replace ASR predictions. 

```python
from vac_aligner.matching import ArmenianAlignerVAC

output_file = 'path/to/save/combined/transcript.txt'
predictions_manifest = 'path/to/manifest/with/predictions.json'
chunks, combined_transcript = ArmenianAlignerVAC.combine_transcript(predictions_manifest,
                                                                    output_file,
                                                                    ending_punctuations="․,։")
```
and then matching

```python
from vac_aligner.matching import ArmenianAlignerVAC

target_base = 'path/to/save/artifacts'
matches_sorted = ArmenianAlignerVAC(combined_transcript, 
                                    chunks, output_file.replace(".txt", ".json"), 
                                    target_base=target_base).align(0.35)

```

and get the benchmark

```python
from vac_aligner.matching.benchmark_on_mcv import Benchmark

benchmark = Benchmark(target_base, predictions_manifest)
stats = benchmark.get_benchmark()
benchmark.analyze_and_save_benchmark(stats, output_file)
```

## Pipeline Explanation

### 1. Voice Activity Detection (VAD)

Split original audios based on the silences in the audios. This can result in having hundreds
of up to 2second long chunks per 20-minute audio. Algorithm is straightforward signal processing, without any ML models
used.


### 2. ASR inference

We need an ASR model to obtain predictions per chunk, we should  

### 3. CER based Matching

At this stage in the pipeline, we have processed a long audio file and its corresponding
transcript into multiple short audio chunks (each 4-15 seconds long) and their predicted
texts from an Automatic Speech Recognition (ASR) model. These short audio segments are
well-suited for ASR or Text-to-Speech (TTS) training. However, the predicted texts from
the ASR model are not perfect and often contain errors such as repeated letters, incomplete
words (e.g., "daw" instead of "draw"), or incorrect words (e.g., "dug" instead of "dog").
Therefore, it is necessary to align these predicted texts with the original transcript.

### Challenges of Matching Predicted Texts with the Original Transcript
The ASR model’s errors can cause significant misalignments. These errors arise due to various reasons, such as:

1. Repeated characters or words.
2. Incomplete words.
3. Incorrect words that are similar to the correct ones.

These discrepancies mean that a direct match between the original transcript and the predicted texts is often impossible. Even small errors can cause sequential mismatches, resulting in incomplete or redundant matches.

**Example Scenarios Illustrating Matching Challenges:**

1. The erroneous prediction results in incomplete extraction from the source text (missed/confused some tokens spoken in the chunk audio):

    **Predicted Text  (i<sup>th</sup> chunk):** "Come here Doggyyyy!!, dog, gggy, y?"

    **Predicted Text  ((i+1)<sup>th</sup> chunk):** "Are you serious?"

    **Original Transcript (context):** "... Come here doggy, doggy. Johnny, are you serious? Why did you hit the dog? ..."

   Here, if we naively take "***Come here doggy, doggy,***"  from the source text as the
   best **CER-based** match (for the i<sup>th</sup> chunk) => the next iteration will incorrectly include "***Johnny***" in the search, causing high CER and misalignment in the future.

2. The erroneous prediction results in redundant extraction from the source text (mistakenly include some tokens not spoken in the i<sup>th</sup> chunk but in (i+1)<sup>th</sup>:

    **Predicted Text (i<sup>th</sup> chunk):** "Hey Madam, madam. Go mam"

    **Predicted Text ((i+1)<sup>th</sup> chunk):** "Adam and Yeva are"

    **Original Transcript:** "... Hey Madam, madam. Adam and Yeva are ..."

  Here, if we naively take "***Hey Madam, madam. Adam,***" from the source text as the best **CER-based** match (for the i<sup>th</sup> chunk) =>  the next iteration will miss "***Adam***" from the original source segment, causing further misalignments.


### Proposed Solution: Dynamic (Greedy) Matching Algorithm

To address the sequential nature and error-prone aspects of the matching process, we propose an advanced algorithm that tries different combinations of source text segments, dynamically adjusts the search window, and greedily selects the best match at each step. Here is a high-level description of how it works:

**Dynamic Window Adjustment:** The algorithm dynamically adjusts the size and position of the text window from the original transcript, referred to as the Source Segment (see examples from the previous - Challenges Section). This adjustment helps align the predicted texts even when there are misalignments. For example, correctly adjusting the start and end position of the Source Segment can help address errors from previous chunks (by extending search window allows room to include missed tokens, while CER threshold helps to remove redundant tokens).

**Greedy Selection:** At each step, the algorithm selects the best match based on similarity measures between the predicted text and the Source Segment. Similarity is defined using the Character Error Rate (CER), with a threshold of <=0.3. Although the algorithm is greedy in the sense that it does not revisit matched texts once they are decided, it allows re-adjustment of the search window to cope with potential errors from previous matches, without breaking the sequential nature of the matches as the matches are not reconsidered.

**Sequential Error Mitigation:** The algorithm continuously mitigates sequential errors by refining its matches and avoiding the propagation of alignment errors. For example, if a match causes subsequent text segments to lag or overlap, the algorithm readjusts (extends its search window),  allowing for correction in subsequent iterations.

So, in other words, our matching algorithm is dynamic in that it always adjusts the search window (determining where to start and end looking in the source text). It ensures that the segment is long enough to include the required true tokens spoken in the audio but reasonably small to optimize computation speed and inductive bias. As highlighted in the first example, understanding the sequential nature of the problem eliminates the need to consider very long source text segments for a given chunk length. Our algorithm is also greedy in that once a match is determined between the chunk text and the source, it is marked and moved to the next iteration (matched chunks are not revisited) to maintain consistency and order throughout the matching process. However, the algorithm can still readjust the search window for subsequent chunks. This flexibility ensures that even if a chunk is incorrectly matched due to coincidental similarity, the algorithm still has opportunities to succeed with the following chunks.

#### High Level Example:

**Original Transcript:** "Once upon a time, in a faraway land, there lived a king."

**Predicted Texts (chunks):**

   1. "Once upon a tme"
   2. "In a farway land"
   3. "The're livd a kng"


**Dynamic Matching Process:**


*   **Initial Window:** The algorithm starts with a text window from the beginning of  the original transcript. For the first chunk, it looks at a larger segment (e.g., "***Once upon a time, in a faraway land***").

*   **First Match:** The predicted text "***Once upon a tme***" is compared with the  segment. The best match "***Once upon a time***" is found, and the algorithm moves to the next segment.

*   **Adjust Window:** The window is adjusted to start from "a tme in a faraway land" - as there was non-exact match because of the ASR error "tme" instead of "time".

*   **Subsequent Matches:** The process continues with the next predicted texts "***in a farway land***" and "the're livd a kng", adjusting the window each time based on the previous match.

* **Final Match:** The last segment "***the're livd a kng***" is matched with "there lived a king".

The algorithm ensures that each predicted text chunk aligns as closely as possible with the corresponding
part of the original transcript, even if the matches are not exact. This dynamic and greedy approach helps in mitigating
sequential errors and ensures a more accurate alignment overall.

## Authors and Acknowledgements
- **Main Author**: Ara Yeroyan - Responsible for the primary development and maintenance of the project.
- **Contributor (Supervisor)**: Dr. Nikolay Karpov - An esteemed scientist who provided expert oversight and significantly contributed to the conceptual framework, research methodologies, and strategic direction of the project and its publication.

## Citing This Work
If you use this project or its components in your research or software, please consider citing it:

```bibtex
@misc{Enable ASR for low-resource languages,
  author = {Ara Yeroyan and Nikolay Karpov},
  title = {VAC - framework for aligning extremely long audios and texts},
  year = {2024},
  publisher = {https://github.com/Ara-Yeroyan},
  howpublished = {https://arxiv.org/abs/2406.01446}
}
```
