import regex as re

from .utils import lowercase
from .metrics import calculate_cer_full


def cut_extra_tokens_from_match(original: str, predicted: str, try_combinations: bool = False, verbose: bool = False):
    """
    Removes redundant parts from the Source Segment to find the best match with the ASR-predicted chunk text.
    This function can optionally try different combinations (cut from both start and the end) to find the best cut-offs
    in the predicted string that matches the original.

    Parameters
    ----------
    original : str
        The original (long) string to compare against - Source Segment Text from the long transcript
    predicted : str
        The predicted chunk text which is potentially full of ASR errors and needs to be replaced from the corresponding
        part from the source label (original)
    try_combinations : bool, optional
        If True, the function will attempt various combinations simultaneously to determine the
        best point to cut the predicted string. Defaults to False which means, only will cut from the end (right)
    verbose : bool, optional
        If True, the function will print detailed information about the operations
        being performed. Defaults to False.

    Returns
    -------
    str
        The predicted string after removing the redundant parts.

    Examples
    --------
    >>> original = "Arfifacts from left; This is a Source Segment that needs to be matched with the ASR preds (chunks)."
    >>> predicted =                     "This is an source segment that leads to bemach with the a s r brides."
    >>> cut_extra_tokens_from_match(original, predicted)
    'This is a Source Segment that needs to be matched with the ASR preds'  # the best possible match
    The latter is most probably (see the benchmark with 97% accuracy) the true text spoken in the chunk !
    """

    if verbose:
        print(f"Predicted chunk text (subject to replaced with corresponding source text segment): {predicted}")
        print("Note, this is much smaller than the Source Segment below, to ensure, it contains the spoken chunk text.")
        print(f"Source Text Segment  (subject to be searched over): {original}")
        print()

    to_shift = 0  # index from the left
    best_cut = original  # currently the full "lengthy" source text
    base_cer = calculate_cer_full(original, predicted)
    best_cer = base_cer

    tokens = re.findall(r'\S+|\s+', original)  # Tokenizes the string including whitespace
    tokens = list(map(lowercase, tokens))
    if not try_combinations:
        for i in range(len(tokens), 0, -1):
            cutted = "".join(tokens[:i])
            curr_cer = calculate_cer_full(cutted, predicted)
            if curr_cer <= best_cer:  # Consider '<=' to handle the case where shorter substrings might also have the same CER
                best_cer = curr_cer
                best_cut = cutted
        if verbose:
            print(f"Cut only from the right (end): {best_cut}")
    else:
        for st_idx in range(1, min(len(tokens) - 4, 7)):  # First for is to remove extra tokens from left
            for i in range(len(tokens), 0, -1):  # second for is to remove extra tokens from the right
                cutted = "".join(tokens[st_idx:i])
                curr_cer = calculate_cer_full(cutted, predicted)
                if curr_cer <= best_cer:  # Consider '<=' to handle the case where shorter substrings might also have the same CER
                    best_cer = curr_cer
                    best_cut = cutted
                    to_shift = len("".join(tokens[:st_idx]))
                    if verbose:
                        print(
                            f"Inner Loop. Cutted from the start (left): {tokens[:st_idx]}; cutting from the right (end); best cut so far: {tokens[i:]}; CER: {round(best_cer, 2)}")

            if verbose:
                print("------------Outer Loop------------")
                print(f"to_shift (from left) {to_shift}; best cut so far: {best_cut}; CER: {round(best_cer, 2)}")

        init_shift = to_shift
        tokens = re.findall(r'\S+|\s+', best_cut)
        cut_counter = 0
        for i in range(0, len(tokens)):
            cutted = "".join(tokens[i:])
            if cut_counter >= 3 and len(cutted) < len(predicted):
                break

            curr_cer = calculate_cer_full(cutted, predicted)
            if curr_cer <= best_cer:  # Consider '<=' to handle the case where shorter substrings might also have the same CER
                cut_counter = 0
                best_cer = curr_cer
                best_cut = cutted
                to_shift = init_shift + len("".join(tokens[:i]))
                if verbose:
                    print(
                        f"cut_counter: {cut_counter} to_shift: {to_shift}; init_shift: {init_shift}; best cut: {best_cut}; CER: {round(best_cer, 2)}")
            else:
                cut_counter += 1
    if verbose:
        print()
        print(f"Matched Pairs: {(predicted, best_cut)}")  # First text is not subject to change, as well as the
        # Source text, just need the best possible match
    return best_cut, best_cer, to_shift


def demo(original, predicted):
    new_b, new_cer, to_shift = cut_extra_tokens_from_match(original, predicted, try_combinations=True)
    print()
    print("Original CER: ",  round(calculate_cer_full(original, predicted), 2))
    print("Source         text:", original)
    print("Predicted      text:", predicted)
    print("Optimized substring:", new_b)
    print("Minimized CER:", round(new_cer, 2))
    print()


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('ignore')

    reference_text = "EXTRA1 EXTRA2, Text to Be Mached EXTRA3 but mistakenly shifted too much"
    asr_bad_pred = "Txt  tobe, mtchd"

    match_ = cut_extra_tokens_from_match(reference_text, asr_bad_pred, try_combinations=True, verbose=True)

    # ------------------
    demo_texts = [
        ("I amm old boy: mistakenly shifted window too much", "I am old boy"),
        ('Իր նիզակը ներսուք տալով մարմնին, կանգնեց պատասխանելով։',
         'իզակը նեցուկ տալով մարմնին, կանգնեց, պատասխանելով. Վանից։'),
        ('Իր նիզակը ներսուք տալով մարմնին, կանգնեց պատասխանելով։',
         'իզակը նեցուկ տալով մարմնին, կանգնեց, պատասխանելով. Վանից։'),
        ('ավաղ, տեսարանն երկար չտևեց։ Կինը համր քայլերով հեռա', 'Կինը համր քայլելով հեռացավ լուսամուտից։'),
        (' ՍԱՂԱԹԵԼ – Բան', 'Սաղացթել։'), (' ԿԱՐԻՆՅԱՆ – Մնա', 'Կարինյան։')
    ]

    for ex in demo_texts:
        reference_text, asr_errorours_pred = ex
        demo(reference_text, asr_errorours_pred)