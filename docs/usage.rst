=====
Usage
=====

To use vac_aligner in a project::

    import vac_aligner



The pipeline processes multi-hour long audio and texts to produce many
multi-second audio chunks and corresponding texts with an accuracy of
97%. This high accuracy is achieved through sophisticated matching
algorithms that correct common ASR errors such as repeated characters,
incomplete words, and incorrect word predictions.

There are a couple of ways the library can be used. One can use the full
functionality by

.. code:: python

   from vac_aligner import run_pipeline

   run_pipeline(
      manifest_file="path/to/save/manifest.json", batch_size=64,
      asr_input_file="path/to/manifest.json", # or path to folder, containing audio chunks (.wav)s
      target_base="path/where/to/save/artifacts", # otherwise, will use `asr_input_file`
      init_aligner_after_asr=True  # If there is no long transcript available and you need to extract it
   )

There are many scenarios where one might need a partial functionality
of the pipeline. Then we can use the classes directly.

Scenario 1
~~~~~~~~~~

Need to perform ASR over short chunks and store in ``nemo_manifest``

.. code:: python

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

Scenario 2
~~~~~~~~~~

You already have predictions manifest and/or long transcript, and want
to run the Matching to obtain the correct chunk texts to replace ASR
predictions.

.. code:: python

   from vac_aligner.matching import ArmenianAlignerVAC

   output_file = 'path/to/save/combined/transcript.txt'
   predictions_manifest = 'path/to/manifest/with/predictions.json'
   chunks, combined_transcript = ArmenianAlignerVAC.combine_transcript(predictions_manifest,
                                                                       output_file,
                                                                       ending_punctuations="․,։")

and then matching

.. code:: python

   from vac_aligner.matching import ArmenianAlignerVAC

   target_base = 'path/to/save/artifacts'
   matches_sorted = ArmenianAlignerVAC(combined_transcript,
                                       chunks, output_file.replace(".txt", ".json"),
                                       target_base=target_base).align(0.35)

and get the benchmark

.. code:: python

   from vac_aligner.matching.benchmark_on_mcv import Benchmark

   benchmark = Benchmark(target_base, predictions_manifest)
   stats = benchmark.get_benchmark()
   benchmark.analyze_and_save_benchmark(stats, output_file)
