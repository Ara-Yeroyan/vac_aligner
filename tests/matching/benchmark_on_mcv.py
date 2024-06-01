import os
import pytest
import pandas as pd


from vac_aligner.matching.benchmark_on_mcv import Benchmark

data_dir = r"vac_aligner/data/matching/matches"
benchmark = Benchmark(data_dir, r"vac_aligner/data/matching/test_with_predictions17_replaced.json")
stats = benchmark.get_benchmark()
original_df = pd.read_csv(os.path.join(os.path.dirname(data_dir), 'combined.csv'))
benchmark_df = benchmark.analyze_and_save_benchmark(stats, os.path.join(data_dir, 'test_benchmark.csv'))


def test_benchmark_metric_consistency():
    assert round(100 * benchmark_df.VAC_WER.sum() / len(benchmark_df), 3) == 0.565
    assert original_df.ASR_WER.sum() == benchmark_df.ASR_WER.sum()
    assert original_df.VAC_WER.sum() == benchmark_df.VAC_WER.sum()