import numpy as np
import pandas as pd
import pytest

from missing_data.run_e2 import mcar_mask, mean_cpt_kl
from parameter_learning import DiscreteBN


def test_mask_only_hides_observed_cells_and_is_reproducible():
    data = pd.DataFrame({"A": [0, 1, None, 0] * 30, "B": [1, 1, 0, 0] * 30}, dtype="Int64")
    a, mask = mcar_mask(data, ["A"], .5, 0)
    b, second_mask = mcar_mask(data, ["A"], .5, 0)
    assert a.equals(b) and mask.equals(second_mask)
    assert a.isna().equals(data.isna() | mask)
    assert not mask.B.any()


@pytest.mark.parametrize("rate", [-0.1, 1.5, np.nan])
def test_mask_rejects_invalid_rates(rate):
    with pytest.raises(ValueError):
        mcar_mask(pd.DataFrame({"A": [0, 1]}, dtype="Int64"), ["A"], rate, 0)


def test_cpt_kl_is_zero_for_identical_models_and_positive_otherwise():
    def model(p):
        return DiscreteBN({"A": (), "B": ("A",)}, {"A": 2, "B": 2},
                          {"A": [[0.5, 0.5]], "B": [[p, 1 - p], [0.3, 0.7]]})

    reference = model(0.9)
    assert mean_cpt_kl(reference, model(0.9)) == 0
    expected = (0.9 * np.log(0.9 / 0.6) + 0.1 * np.log(0.1 / 0.4)) / 2 / 2
    assert mean_cpt_kl(reference, model(0.6)) == pytest.approx(expected)
