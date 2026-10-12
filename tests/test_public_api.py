import spatialproteomics


def test_all_exports_exist():
    missing = [name for name in spatialproteomics.__all__ if not hasattr(spatialproteomics, name)]
    assert not missing, f"Names listed in spatialproteomics.__all__ but not defined: {missing}"
