"""
CLI — Command pattern via Typer.
Commands: asl train | asl evaluate | asl webcam | asl prepare
"""
from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger

app = typer.Typer(name="asl", help="ASL Sign Language Recognition CLI")


@app.command()
def prepare(
    train_ratio: float = typer.Option(0.80, help="Fraction of images for training"),
    val_ratio: float = typer.Option(0.10, help="Fraction of images for validation"),
) -> None:
    """Download and split the ASL dataset."""
    from asl.config.container import container

    container.prepare_data_use_case().execute(train_ratio=train_ratio, val_ratio=val_ratio)
    typer.echo("Dataset preparation complete.")


@app.command()
def train(
    phase1_epochs: int = typer.Option(10, help="Phase-1 training epochs (frozen backbone)"),
    phase2_epochs: int = typer.Option(5, help="Phase-2 fine-tuning epochs (0 = skip)"),
    batch_size: int = typer.Option(32),
    model_name: str = typer.Option("asl_model"),
    backbone: str = typer.Option("mobilenet_v2", help="mobilenet_v2 | efficientnet_b0"),
    subset: float = typer.Option(1.0, help="Fraction of data to use (0.01 for smoke-tests)"),
) -> None:
    """Train the ASL classifier."""
    from asl.application.dto.train_request import TrainRequest
    from asl.config.container import container

    request = TrainRequest(
        phase1_epochs=phase1_epochs,
        phase2_epochs=phase2_epochs,
        batch_size=batch_size,
        model_name=model_name,
        backbone=backbone,
        subset_fraction=subset,
    )
    result = container.training_service().train(request)
    typer.echo(result.summary)


@app.command()
def evaluate(
    model_name: str = typer.Argument(..., help="Saved model name (without .keras extension)"),
    batch_size: int = typer.Option(32),
) -> None:
    """Evaluate a trained model on the held-out test set."""
    from asl.config.container import container

    result = container.evaluate_model_use_case().execute(model_name=model_name, batch_size=batch_size)
    typer.echo(f"Test accuracy: {result.test_accuracy:.2%}")
    if result.confusion_matrix_path:
        typer.echo(f"Confusion matrix saved → {result.confusion_matrix_path}")


@app.command()
def webcam(
    model_name: str = typer.Option(
        None,
        help="Model name (default: latest). Use 'fallback' for local heuristic mode.",
    ),
) -> None:
    """Run real-time ASL recognition via webcam."""
    from asl.presentation.webcam.realtime_ui import RealtimeUI

    ui = RealtimeUI(model_name=model_name)
    typer.echo("Starting webcam — press Q to quit.")
    ui.run()


if __name__ == "__main__":
    app()
