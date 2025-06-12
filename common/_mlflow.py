# setup MLflow logger and callbacks
import os
from typing import List, Dict

from dotenv import load_dotenv
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Timer
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.loggers import MLFlowLogger

load_dotenv(dotenv_path="../.env")
MLFLOW_SERVICE_URI = os.getenv("MLFLOW_SERVICE_URI", "")


def get_mlflow_logger(experiment_name: str, run_name: str, tags: Dict[str, str] = None) -> MLFlowLogger:
    """
    Set up MLflow logger for PyTorch Lightning.
    Args:
        experiment_name (str): Name of the MLflow experiment.
        run_name (str): Name of the MLflow run.
    Returns:
        MLFlowLogger: Configured MLflow logger.
    """
    logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=MLFLOW_SERVICE_URI,  # can also use http://... for remote
    )

    if tags is not None:
        for key, value in tags.items():
            logger.experiment.set_tag(logger.run_id, key, value)
    return logger


def get_callbacks(
    run_name: str,
    patience: int,
    hyper_param_str: str = "",
    exp_name: str = "",
    monitor_metric: str = "val_ndcg10",
    monitor_mode: str = "max",
    min_delta: float = 0.0,
) -> List[Callback]:
    """
    Set up callbacks for PyTorch Lightning.
    Args:
        exp_name (str): Name of the MLflow experiment.
        run_name (str): Name of the MLflow run.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        dirpath (str): Directory path to save checkpoints.
        monitor_metric (str): Metric to monitor for early stopping and checkpointing. ('val_ndcg@5', 'val_loss' or 'val_f1')
        monitor_mode (str): Mode for monitoring ('min' for loss, 'max' for accuracy/F1/NDCG).
    Returns:
        list: list containing ModelCheckpoint and EarlyStopping callbacks.
    """
    # Define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        mode=monitor_mode,
        save_top_k=1,
        save_weights_only=True,
        dirpath=f"test_checkpoints/{exp_name}",
        filename=f"{run_name}-{hyper_param_str}-best-checkpoint-{{epoch:02d}}-{{{monitor_metric}:.2f}}",
        verbose=True,
    )

    # Define the early stopping callback
    early_stopping = EarlyStopping(monitor=monitor_metric, patience=patience, mode=monitor_mode, min_delta=min_delta, verbose=True)

    # Define the timer callback
    timer = Timer()

    return [checkpoint_callback, early_stopping, timer]
