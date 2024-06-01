import pytest
import warnings

from vac_aligner.matching.cutting import cut_extra_tokens_from_match

warnings.filterwarnings('ignore')


@pytest.mark.parametrize("reference_text, asr_bad_pred, expected", [
    (
    "EXTRA1 EXTRA2, Text to Be Mached EXTRA3 but mistakenly shifted too much", "Txt  tobe, mtchd", 'Text to Be Mached'),
    ("I amm old boy: mistakenly shifted window too much", "I am old boy", 'amm old boy:'),
    ('Իր նիզակը ներսուք տալով մարմնին, կանգնեց պատասխանելով։',
     'իզակը նեցուկ տալով մարմնին, կանգնեց, պատասխանելով. Վանից։',
     'նիզակը ներսուք տալով մարմնին, կանգնեց պատասխանելով։'),
    ('ավաղ, տեսարանն երկար չտևեց։ Կինը համր քայլերով հեռա', 'Կինը համր քայլելով հեռացավ լուսամուտից։',
     'Կինը համր քայլերով հեռա')
])


def test_cut_extra_tokens_from_match(reference_text, asr_bad_pred, expected):
    assert cut_extra_tokens_from_match(reference_text, asr_bad_pred, try_combinations=True, verbose=False)[
               0] == expected

