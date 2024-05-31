#!/usr/bin/env python

import pytest
from vac_aligner.matching.metrics import calculate_cer, calculate_wer, calculate_cer_full


def test_calculate_cer_completely_different():
    "Assuming completely different strings of the same length"
    assert calculate_cer_full("hello", "abcde") == 1, "CER for completely different strings should be 1"


def test_calculate_cer_substitution():
    "One substitution"
    assert 1 > calculate_cer("hello", "jello") > 0, "CER should be between 0 and 1 for one substitution"


def test_calculate_cer_identical():
    assert calculate_cer("hello", "hello") == 0, "CER for identical strings should be 0"


def test_calculate_cer_insertion():
    assert 1 > calculate_cer("hello", "helloo") > 0, "CER should be between 0 and 1 for one insertion"


def test_calculate_cer_deletion():
    assert 1 > calculate_cer("hello", "hell") > 0, "CER should be between 0 and 1 for one deletion"


def test_calculate_cer_empty_strings():
    assert calculate_cer("", "") == 0, "CER for two empty strings should be 0"
    assert calculate_cer("hello", "") == 1, "CER should be 1 when comparing any string to an empty string"
    assert calculate_cer("", "hello") == 1, "CER should be 1 when comparing empty string to any string"

# Assuming calculate_cer_full is implemented correctly
# Here are basic tests for it, mirroring the tests for calculate_cer but potentially with different expectations


def test_calculate_cer_full_identical():
    assert calculate_cer_full("hello", "hello") == 0, "Full CER for identical strings should be 0"


def test_calculate_wer_completely_different():
    """Assuming completely different strings with the same number of words"""
    assert calculate_wer("hello world", "goodbye there") == 1, "WER for completely different strings should be 1"


def test_calculate_wer_substitution():
    """One substitution in a two-word sentence"""
    assert calculate_wer("hello world", "hello there") == 0.5, "WER should be 0.5 for one substitution out of two words"


def test_calculate_wer_insertion():
    """One insertion"""
    assert calculate_wer("hello world", "hello big world") == 1/2, "WER should be 1/2 for one insertion in a three-word sentence"


def test_calculate_wer_deletion():
    """One deletion"""
    assert calculate_wer("hello big world", "hello world") == 1/3, "WER should be 1/3 for one deletion in a three-word sentence"


def test_calculate_wer_empty_strings():
    assert calculate_wer("", "") == 0, "WER for two empty strings should be 0"
    assert calculate_wer("hello world", "") == 1, "WER should be 1 when comparing any string to an empty string"
    assert calculate_wer("", "hello world") == 1, "WER should be 1 when comparing empty string to any string"


def test_calculate_wer_multiple_errors():
    "Multiple errors including substitution, insertion, and deletion"
    original = "hello world here"
    modified = "hi world there"
    # Expected WER: (1 substitution: hello->hi, 1 substitution: here->there) / 3 words = 2/3
    assert calculate_wer(original, modified) == 2/3, "WER should account for multiple types of errors"
