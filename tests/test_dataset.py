from app.dataset import CATEGORICAL_FEATURES, DATA_PATH, NUMERIC_FEATURES, load_dataset


def test_german_dataset_shape_and_labels():
    X, y = load_dataset()
    assert DATA_PATH.exists()
    assert X.shape == (1000, 20)
    assert len(CATEGORICAL_FEATURES) == 13
    assert len(NUMERIC_FEATURES) == 7
    assert y.value_counts().to_dict() == {0: 700, 1: 300}


def test_categorical_codes_are_read_as_strings():
    X, _ = load_dataset()
    assert X.loc[0, "checking_status"] == "A11"
    assert X.loc[0, "personal_status_sex"] == "A93"
