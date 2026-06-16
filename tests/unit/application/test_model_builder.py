"""
Unit tests for ModelConfigBuilder — no ML framework imports needed.
"""
import pytest

from asl.infrastructure.ml.model_builder import ModelConfigBuilder


def test_defaults():
    config = ModelConfigBuilder().build()
    assert config.input_size == 224
    assert config.num_classes == 29
    assert config.dropout_rate == 0.5


def test_fluent_chain():
    config = (
        ModelConfigBuilder()
        .with_input_size(128)
        .with_num_classes(10)
        .with_dropout(0.3)
        .with_fine_tune_layers(20)
        .build()
    )
    assert config.input_size == 128
    assert config.num_classes == 10
    assert config.dropout_rate == 0.3
    assert config.fine_tune_layers == 20


def test_invalid_dropout_raises():
    with pytest.raises(ValueError):
        ModelConfigBuilder().with_dropout(1.5).build()
