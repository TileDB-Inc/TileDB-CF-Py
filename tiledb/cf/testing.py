import numpy as np


def assert_dict_arrays_equal(d1, d2, ordered=True):
    assert d1.keys() == d2.keys(), "Keys not equal"

    if ordered:
        for k in d1.keys():
            np.testing.assert_array_equal(d1[k], d2[k])
    else:
        d1_dtypes = [tuple((name, value.dtype)) for name, value in d1.items()]
        d2_dtypes = [tuple((name, value.dtype)) for name, value in d2.items()]

        assert d1_dtypes == d2_dtypes

        d1_records = [tuple(values) for values in zip(*d1.values())]
        array1 = np.sort(np.array(d1_records, dtype=d1_dtypes))

        d2_records = [tuple(values) for values in zip(*d2.values())]
        array2 = np.sort(np.array(d2_records, dtype=d2_dtypes))

        np.testing.assert_array_equal(array1, array2)
