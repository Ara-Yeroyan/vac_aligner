from typing import Literal


import regex as re


def clean_corpus(text: str, source: Literal['prediction', 'original'] = 'original') -> str:
    """Corpus specific (source or predicted) text processing to better align based on CER threshold"""
    if source == "original":
        verjaket_origin = ":"
        verjaket_nemo = "։"
        replace_pairs = [("\n", " "), (" — ", " "), ("— ", " "), (" —", " "), (verjaket_origin, verjaket_nemo)]
        for pair in replace_pairs:
            text = text.replace(pair[0], pair[1])
        text = re.sub("\s+", " ", text)
        text = text.strip().replace(" և ", " եւ ")
    else:
        text = text.replace(" - ", " ").replace(" -", " ").replace("- ", " ")
        text = re.sub("\s+", " ", text)
    return text


def lowercase(token):
    """SPECIAL Name such as DAVID -> Special Name such as David"""
    if all([char.isupper() for char in token]):
        return token.lower().title()
    return token