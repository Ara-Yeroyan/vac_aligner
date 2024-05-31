import regex as re
import numpy as np
from difflib import SequenceMatcher

pattern = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")  # GPT4 pre tokenize


def calculate_cer(s1, s2):
    """
    Character Error Rate (CER) is typically (substitutions + deletions + insertions) / len(original)
    We'll use a simplistic version where we just count how many characters differ
    This uses SequenceMatcher which gives a similarity ratio; we convert it to error rate
    """
    matcher = SequenceMatcher(None, s1, s2)
    return 1 - matcher.ratio()


def calculate_wer(reference, hypothesis):
    """
    Calculates the Word Error Rate (WER) between a reference and a hypothesis sentence.
    """
    if reference == hypothesis == "":
        return 0

    elif reference == "" or hypothesis == "":
        return 1

    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    matrix = np.zeros((len(reference_words) + 1, len(hypothesis_words) + 1))

    for i in range(len(reference_words) + 1):
        matrix[i, 0] = i
    for j in range(len(hypothesis_words) + 1):
        matrix[0, j] = j
    for i in range(1, len(reference_words) + 1):
        for j in range(1, len(hypothesis_words) + 1):
            if reference_words[i - 1] == hypothesis_words[j - 1]:
                matrix[i, j] = matrix[i - 1, j - 1]
            else:
                substitution = matrix[i - 1, j - 1] + 1
                insertion = matrix[i, j - 1] + 1
                deletion = matrix[i - 1, j] + 1
                matrix[i, j] = min(substitution, insertion, deletion)

    distance = matrix[len(reference_words), len(hypothesis_words)]
    wer_result = distance / len(reference_words)
    return wer_result


def calculate_cer_full(reference: str, hypothesis: str):
    """
    Calculate the Character Error Rate (CER) between reference and hypothesis strings.
    """
    ref = reference.replace(" ", "")
    hyp = hypothesis.replace(" ", "")
    n, m = len(ref), len(hyp)

    d = np.zeros((n+1)*(m+1), dtype=np.uint8).reshape((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, n+1):
        for j in range(1, m+1):
            if ref[i-1] == hyp[j-1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i-1][j] + 1,      # Deletion
                          d[i][j-1] + 1,      # Insertion
                          d[i-1][j-1] + cost) # Substitution

    cer = d[n][m] / float(n)
    return cer
