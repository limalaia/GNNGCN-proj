#!/usr/bin/env python
"""Configura e inicia os experimentos da Weighted GNN."""

from pathlib import Path

from src.experiment_pipeline import execute_experiments


# Parâmetros do experimento: cada valor pode ser escalar ou uma lista.
TARGET_DATE = "2024-01-17"
K = [15]
HIDDEN_DIM = [256]
N_LAYERS = [2]
EPOCHS = [1000]
LR = 1e-3
WEIGHT_DECAY = 0.0
LOSS_FN = 'mse'
SEED = 42
PATIENCE = 2000
MIN_DELTA = 1e-4
TARGET_NODE = 32
STANDARDIZE_FEATURES = True
DEVICE = "auto"

# Parâmetros de caminhos e relatório.
DATA_DIR = Path("Datasets")
OUTPUT_ROOT = Path("outputs")
COMPARISON_REPORT = True
PIVOT_PARAMETER = "K"


if __name__ == "__main__":
    execute_experiments(
        globals(),
        project_dir=Path(__file__).resolve().parent,
        data_dir=DATA_DIR,
        output_root=OUTPUT_ROOT,
        comparison_report=COMPARISON_REPORT,
        pivot_parameter=PIVOT_PARAMETER,
    )
