"""
Unit tests for TrainingEventBus (Observer DP).
"""
from asl.application.events.training_events import (
    Phase1CompleteEvent,
    TrainingCompleteEvent,
    TrainingEventBus,
)


def test_subscribe_and_publish():
    bus = TrainingEventBus()
    received = []
    bus.subscribe(Phase1CompleteEvent, lambda e: received.append(e))
    bus.publish(Phase1CompleteEvent(history={"accuracy": [0.9]}))
    assert len(received) == 1
    assert received[0].history["accuracy"] == [0.9]


def test_no_handler_for_event_type_is_silent():
    bus = TrainingEventBus()
    bus.publish(TrainingCompleteEvent())  # no subscribers — should not raise


def test_faulty_handler_does_not_crash_bus():
    bus = TrainingEventBus()

    def bad_handler(_):
        raise RuntimeError("Handler failed")

    bus.subscribe(TrainingCompleteEvent, bad_handler)
    bus.publish(TrainingCompleteEvent())  # must not raise


def test_unsubscribe():
    bus = TrainingEventBus()
    calls = []
    handler = lambda e: calls.append(e)
    bus.subscribe(TrainingCompleteEvent, handler)
    bus.unsubscribe(TrainingCompleteEvent, handler)
    bus.publish(TrainingCompleteEvent())
    assert calls == []
