# setup MLflow logger and callbacks
import os

from dotenv import load_dotenv
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Timer
from pytorch_lightning.loggers import MLFlowLogger

load_dotenv(dotenv_path="../.env")
MLFLOW_SERVICE_URI = os.getenv("MLFLOW_SERVICE_URI", "")


def get_mlflow_logger(experiment_name: str, run_name: str) -> MLFlowLogger:
    """
    Set up MLflow logger for PyTorch Lightning.
    Args:
        experiment_name (str): Name of the MLflow experiment.
        run_name (str): Name of the MLflow run.
    Returns:
        MLFlowLogger: Configured MLflow logger.
    """
    return MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=MLFLOW_SERVICE_URI,  # can also use http://... for remote
    )


def get_callbacks(exp_name: str, run_name: str, patience: int, dirpath: str = "test_checkpoints/") -> tuple:
    """
    Set up callbacks for PyTorch Lightning.
    Args:
        exp_name (str): Name of the MLflow experiment.
        run_name (str): Name of the MLflow run.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        dirpath (str): Directory path to save checkpoints.
    Returns:
        tuple: Tuple containing ModelCheckpoint and EarlyStopping callbacks.
    """
    # Define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1",  # or "val_loss" if you're monitoring loss
        mode="max",  # or "max" if you're monitoring accuracy/F1
        save_top_k=1,
        save_weights_only=True,
        dirpath=dirpath,
        filename=f"{exp_name}-{run_name}-best-checkpoint-{{epoch:02d}}-{{val_f1:.2f}}",
        verbose=True,
    )

    # Define the early stopping callback
    early_stopping = EarlyStopping(monitor="val_f1", patience=patience, mode="max", verbose=True)

    # Define the timer callback
    timer = Timer()

    return [checkpoint_callback, early_stopping, timer]
