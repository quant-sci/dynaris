"""Smoke tests for the dynaris package."""


def test_version():
    import dynaris

    assert dynaris.__version__ == "0.1.0"


def test_subpackages_importable():
    import dynaris.core
    import dynaris.filters
    import dynaris.models
    import dynaris.plotting
    import dynaris.smoothers
    import dynaris.utils

    assert dynaris.core is not None
    assert dynaris.filters is not None
    assert dynaris.models is not None
    assert dynaris.plotting is not None
    assert dynaris.smoothers is not None
    assert dynaris.utils is not None
