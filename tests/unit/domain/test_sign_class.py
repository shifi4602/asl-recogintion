import pytest

from asl.domain.entities.sign_class import SignClass


def test_label_list_has_29_entries():
    assert len(SignClass.label_list()) == 29


def test_from_index_returns_correct_member():
    assert SignClass.from_index(0) == SignClass.A
    assert SignClass.from_index(25) == SignClass.Z
    assert SignClass.from_index(26) == SignClass.SPACE


def test_from_index_out_of_range_raises():
    with pytest.raises(ValueError):
        SignClass.from_index(99)


def test_num_classes():
    assert SignClass.num_classes() == 29
