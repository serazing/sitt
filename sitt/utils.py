import os
import difflib

def _not_empty(x):
    return x.shape and 0 not in x.shape


def _maybe_check_dtype(a, dtype=None):
    # Only check dtype matches for non-empty
    if _not_empty(a):
        assert a.dtype == dtype


# Copied from dask.array.utils
def assert_eq(a, b, **kwargs):
    if isinstance(a, Array):
        adt = a.dtype
        a = a.compute(get=get_sync)
        _maybe_check_dtype(a, adt)
    else:
        adt = getattr(a, 'dtype', None)
    if isinstance(b, Array):
        bdt = b.dtype
        assert bdt is not None
        b = b.compute(get=get_sync)
        _maybe_check_dtype(b, bdt)
    else:
        bdt = getattr(b, 'dtype', None)

    if str(adt) != str(bdt):
        diff = difflib.ndiff(str(adt).splitlines(), str(bdt).splitlines())
        raise AssertionError('string repr are different' + os.linesep +
                             os.linesep.join(diff))

    try:
        if _not_empty(a) and _not_empty(b):
            # Treat all empty arrays as equivalent
            assert a.shape == b.shape
            assert allclose(a, b, **kwargs)
        return
    except TypeError:
        pass

    c = a == b

    if isinstance(c, np.ndarray):
        assert c.all()
    else:
        assert c

    return True