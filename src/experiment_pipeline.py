#!/usr/bin/env python
"""Executa uma Weighted GNN para uma data e uma configuração específicas.

O pipeline foi extraído da ideia central de ``Notebooks/WeightedEdges.ipynb``:
cada nó representa uma estação do INMET, o grafo é construído por KNN e as
arestas recebem peso exponencialmente decrescente com a distância.
"""

from __future__ import annotations

import copy
import itertools
import json
import logging
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, Normalize
from sklearn.neighbors import NearestNeighbors


LOGGER = logging.getLogger("gnn_experiment")

LOSS_ALIASES = {
    "mse": "mse",
    "mse_loss": "mse",
    "mae": "mae",
    "l1": "mae",
    "l1_loss": "mae",
    "huber": "huber",
    "smooth_l1": "huber",
    "weighted_mse": "weighted_mse",
    "wmse": "weighted_mse",
}


@dataclass(frozen=True)
class ExperimentConfig:
    TARGET_DATE: str
    K: int
    HIDDEN_DIM: int
    N_LAYERS: int
    EPOCHS: int
    LR: float
    WEIGHT_DECAY: float
    LOSS_FN: str
    SEED: int
    PATIENCE: int
    MIN_DELTA: float
    TARGET_NODE: int
    STANDARDIZE_FEATURES: bool
    DEVICE: str


RUN_PARAMETER_NAMES = tuple(ExperimentConfig.__dataclass_fields__)


class WeightedGNN(nn.Module):
    """GNN densa com propagação por uma adjacência ponderada."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        if n_hidden_layers < 1:
            raise ValueError("N_LAYERS deve ser maior ou igual a 1.")

        layers = [nn.Linear(in_dim, hidden_dim)]
        layers.extend(
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_hidden_layers - 1)
        )
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        normalized_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        hidden = node_features
        for layer in self.hidden_layers:
            hidden = normalized_adjacency @ hidden
            hidden = F.relu(layer(hidden))
        return self.output_layer(hidden).squeeze(-1)


def parse_loss_name(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in LOSS_ALIASES:
        choices = ", ".join(sorted(set(LOSS_ALIASES.values())))
        raise ValueError(
            f"LOSS_FN desconhecida: {value!r}. Opções: {choices}."
        )
    return LOSS_ALIASES[normalized]


def parse_target_date(value: str) -> str:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "TARGET_DATE deve usar o formato YYYY-MM-DD."
        ) from exc

    if pd.isna(timestamp) or timestamp != timestamp.normalize():
        raise ValueError(
            "TARGET_DATE deve conter somente a data no formato YYYY-MM-DD."
        )
    return timestamp.strftime("%Y-%m-%d")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError("O valor deve ser maior ou igual a 1.")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError("O valor não pode ser negativo.")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError("O valor deve ser finito e maior que zero.")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError("O valor deve ser finito e não negativo.")
    return parsed


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda foi solicitado, mas CUDA não está disponível.")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def resolve_from_project(path: Path, project_dir: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


def create_output_directory(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%d_%m_%Y_%H")
    base_name = f"experiment_{timestamp}"

    for run_number in range(1, 1000):
        suffix = "" if run_number == 1 else f"_{run_number:02d}"
        candidate = output_root / f"{base_name}{suffix}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate

    raise RuntimeError(f"Não foi possível criar uma pasta única em {output_root}.")


def format_batch_directory_name(
    target_date: object,
    execution_time: str | None = None,
) -> str:
    target_dates = [parse_target_date(value) for value in parameter_options(target_date)]
    target_label = (
        target_dates[0]
        if len(target_dates) == 1
        else f"{target_dates[0]}_to_{target_dates[-1]}"
    )
    if execution_time is None:
        execution_time = datetime.now().astimezone().strftime("%Hh%M")
    return f"run_{target_label}_{execution_time}"


def create_batch_directory(output_root: Path, target_date: object) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    base_name = format_batch_directory_name(target_date)
    for run_number in range(1, 1000):
        suffix = "" if run_number == 1 else f"_{run_number:02d}"
        candidate = output_root / f"{base_name}{suffix}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"NÃ£o foi possÃ­vel criar uma pasta de lote em {output_root}.")


def configure_logging(output_dir: Path) -> None:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(
        output_dir / "execution.log", encoding="utf-8"
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)
    LOGGER.propagate = False


def save_json(path: Path, content: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2, allow_nan=False)


def find_catalog(data_dir: Path) -> Path:
    candidates = sorted((data_dir / "dados_inmet").glob("Catalogo*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Catálogo de estações não encontrado em {data_dir / 'dados_inmet'}."
        )
    return candidates[0]


def load_stations(catalog_path: Path) -> pd.DataFrame:
    catalog = pd.read_csv(catalog_path, sep=";", dtype=str)
    required = {"DC_NOME", "SG_ESTADO", "VL_LATITUDE", "VL_LONGITUDE"}
    missing = required.difference(catalog.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no catálogo: {sorted(missing)}.")

    stations = catalog.loc[
        catalog["SG_ESTADO"].eq("RS"),
        ["DC_NOME", "VL_LATITUDE", "VL_LONGITUDE"],
    ].copy()
    stations["latitude"] = pd.to_numeric(
        stations["VL_LATITUDE"].str.replace(",", ".", regex=False),
        errors="raise",
    )
    stations["longitude"] = pd.to_numeric(
        stations["VL_LONGITUDE"].str.replace(",", ".", regex=False),
        errors="raise",
    )
    stations = (
        stations.rename(columns={"DC_NOME": "station"})
        .loc[:, ["station", "latitude", "longitude"]]
        .reset_index(drop=True)
    )
    stations.index.name = "node"

    if stations.empty:
        raise ValueError("O catálogo não contém estações do RS.")
    if stations[["latitude", "longitude"]].isna().any().any():
        raise ValueError("O catálogo contém coordenadas ausentes.")
    return stations


def station_indexers(stations: pd.DataFrame) -> tuple[xr.DataArray, xr.DataArray]:
    station_names = stations["station"].to_numpy()
    latitudes = xr.DataArray(
        stations["latitude"].to_numpy(),
        dims="station",
        coords={"station": station_names},
    )
    longitudes = xr.DataArray(
        stations["longitude"].to_numpy(),
        dims="station",
        coords={"station": station_names},
    )
    return latitudes, longitudes


def select_one_day(
    data_array: xr.DataArray,
    target_date: pd.Timestamp,
) -> xr.DataArray:
    day_end = target_date + pd.Timedelta(days=1)
    inclusive_end = day_end - pd.Timedelta(nanoseconds=1)
    selected = data_array.sel(time=slice(target_date, inclusive_end))
    if selected.sizes.get("time", 0) == 0:
        raise ValueError(f"Sem dados horários para {target_date.date()}.")
    return selected


def load_temperature_feature(
    path: Path,
    target_date: pd.Timestamp,
    latitudes: xr.DataArray,
    longitudes: xr.DataArray,
) -> np.ndarray:
    with xr.open_dataset(path) as dataset:
        if "t2m" not in dataset:
            raise KeyError(f"Variável t2m ausente em {path}.")
        hourly = select_one_day(dataset["t2m"], target_date)
        values = (
            hourly.sel(
                latitude=latitudes,
                longitude=longitudes,
                method="nearest",
            )
            .mean("time")
            .load()
            .values
        )
    return np.asarray(values, dtype=np.float32) - np.float32(273.15)


def load_vertical_velocity_feature(
    path: Path,
    target_date: pd.Timestamp,
    latitudes: xr.DataArray,
    longitudes: xr.DataArray,
) -> np.ndarray:
    with xr.open_dataset(path) as dataset:
        if "w" not in dataset:
            raise KeyError(f"Variável w ausente em {path}.")

        hourly = select_one_day(dataset["w"], target_date)
        level_coord = next(
            (
                name
                for name in ("isobaricInhPa", "level", "pressure_level")
                if name in hourly.coords or name in hourly.dims
            ),
            None,
        )
        if level_coord is None:
            raise ValueError(f"Nível de pressão não encontrado em {path}.")

        at_850_hpa = hourly.sel({level_coord: 850}, method="nearest")
        values = (
            at_850_hpa.sel(
                latitude=latitudes,
                longitude=longitudes,
                method="nearest",
            )
            .mean("time")
            .load()
            .values
        )
    return np.asarray(values, dtype=np.float32)


def load_daily_precipitation(
    path: Path,
    target_date: pd.Timestamp,
    latitudes: xr.DataArray,
    longitudes: xr.DataArray,
) -> np.ndarray:
    day_end = target_date + pd.Timedelta(days=1)

    with xr.open_dataset(path) as dataset:
        if "tp" not in dataset:
            raise KeyError(f"Variável tp ausente em {path}.")
        if "valid_time" not in dataset.coords:
            raise KeyError(f"Coordenada valid_time ausente em {path}.")
        if "step" not in dataset["tp"].dims:
            raise ValueError("A precipitação precisa ter a dimensão step.")

        # Uma previsão válida no começo do dia pode ter sido inicializada no
        # dia anterior. Doze horas cobrem o maior step deste conjunto ERA5.
        reference_start = target_date - pd.Timedelta(hours=12)
        subset = dataset.sel(time=slice(reference_start, day_end))
        points = (
            subset["tp"]
            .sel(
                latitude=latitudes,
                longitude=longitudes,
                method="nearest",
            )
            .transpose("time", "step", "station")
            .load()
        )
        valid_times = np.asarray(subset["valid_time"].values).reshape(-1)
        values = np.asarray(points.values).reshape(-1, len(latitudes))

    in_target_day = (valid_times >= np.datetime64(target_date)) & (
        valid_times < np.datetime64(day_end)
    )
    if not in_target_day.any():
        raise ValueError(f"Sem precipitação válida para {target_date.date()}.")

    target_times = valid_times[in_target_day]
    target_values = values[in_target_day]

    # Reproduz o groupby(...).last() do notebook antes da soma diária.
    order = np.argsort(target_times, kind="stable")
    target_times = target_times[order]
    target_values = target_values[order]
    keep_last_duplicate = np.r_[
        target_times[1:] != target_times[:-1],
        True,
    ]
    target_times = target_times[keep_last_duplicate]
    target_values = target_values[keep_last_duplicate]

    if len(target_times) != 24:
        raise ValueError(
            f"Dia incompleto em {path}: esperadas 24 horas válidas para "
            f"{target_date.date()}, encontradas {len(target_times)}."
        )

    # A unidade original é metro; a saída utilizada no notebook é mm/dia.
    return np.asarray(target_values.sum(axis=0) * 1000.0, dtype=np.float32)


def validate_finite(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        nan_count = int(np.isnan(values).sum())
        inf_count = int(np.isinf(values).sum())
        raise ValueError(
            f"{name} contém valores inválidos (nan={nan_count}, inf={inf_count})."
        )


def load_day_data(
    data_dir: Path,
    target_date: str,
    stations: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    nc_dir = data_dir / "nc_files"
    paths = {
        "precipitation": nc_dir / "era5_precipitation_80-26.nc",
        "temperature": nc_dir / "temp_99-25.nc",
        "vertical_velocity": nc_dir / "vv_94-25.nc",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Arquivos NetCDF ausentes:\n- " + "\n- ".join(missing)
        )

    date = pd.Timestamp(target_date)
    latitudes, longitudes = station_indexers(stations)

    LOGGER.info("Carregando temperatura média diária...")
    temperature = load_temperature_feature(
        paths["temperature"], date, latitudes, longitudes
    )
    LOGGER.info("Carregando velocidade vertical média em 850 hPa...")
    vertical_velocity = load_vertical_velocity_feature(
        paths["vertical_velocity"], date, latitudes, longitudes
    )
    LOGGER.info("Carregando precipitação diária...")
    precipitation = load_daily_precipitation(
        paths["precipitation"], date, latitudes, longitudes
    )

    validate_finite("temperatura", temperature)
    validate_finite("velocidade vertical", vertical_velocity)
    validate_finite("precipitação", precipitation)

    features = np.column_stack((vertical_velocity, temperature)).astype(np.float32)
    return features, precipitation


def split_nodes(
    n_nodes: int,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n_nodes, generator=generator).numpy()

    n_train = round(train_ratio * n_nodes)
    n_val = round(val_ratio * n_nodes)
    n_test = n_nodes - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError("O split por nós produziu algum conjunto vazio.")

    split = np.full(n_nodes, "test", dtype=object)
    split[permutation[:n_train]] = "train"
    split[permutation[n_train : n_train + n_val]] = "validation"

    masks = tuple(split == name for name in ("train", "validation", "test"))
    return split, masks[0], masks[1], masks[2]


def build_weighted_graph(
    stations: pd.DataFrame,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = stations[["latitude", "longitude"]].to_numpy(dtype=np.float32)
    n_nodes = len(coordinates)
    if k >= n_nodes:
        raise ValueError(f"K={k} deve ser menor que o número de nós ({n_nodes}).")

    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(coordinates)
    _, indices = neighbors.kneighbors(coordinates)
    indices = indices[:, 1:]

    binary = np.zeros((n_nodes, n_nodes), dtype=bool)
    for node, node_neighbors in enumerate(indices):
        binary[node, node_neighbors] = True
    binary = binary | binary.T
    np.fill_diagonal(binary, False)

    coordinate_tensor = torch.as_tensor(coordinates)
    distance_matrix = torch.cdist(coordinate_tensor, coordinate_tensor, p=2)
    distance_weights = torch.exp(-distance_matrix).numpy()

    weighted = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    weighted[binary] = distance_weights[binary]
    np.fill_diagonal(weighted, 1.0)

    degree = weighted.sum(axis=1).clip(min=1e-12)
    inv_sqrt_degree = 1.0 / np.sqrt(degree)
    normalized = (
        weighted * inv_sqrt_degree[:, np.newaxis] * inv_sqrt_degree[np.newaxis, :]
    ).astype(np.float32)
    return binary, weighted, normalized


def standardize_features(
    features: np.ndarray,
    train_mask: np.ndarray,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if enabled:
        mean = features[train_mask].mean(axis=0)
        std = features[train_mask].std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
    else:
        mean = np.zeros(features.shape[1], dtype=np.float32)
        std = np.ones(features.shape[1], dtype=np.float32)

    standardized = (features - mean) / std
    return (
        standardized.astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def make_loss(
    loss_name: str,
    train_targets: torch.Tensor,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "mae":
        return nn.L1Loss()
    if loss_name == "huber":
        return nn.HuberLoss(delta=1.0)
    if loss_name == "weighted_mse":
        threshold = torch.quantile(train_targets.detach(), 0.90)

        def weighted_mse(
            prediction: torch.Tensor,
            target: torch.Tensor,
        ) -> torch.Tensor:
            weights = torch.where(target > threshold, 10.0, 1.0)
            return torch.mean(weights * (prediction - target).square())

        return weighted_mse
    raise ValueError(f"LOSS_FN não implementada: {loss_name}.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    config: ExperimentConfig,
    features: np.ndarray,
    targets: np.ndarray,
    normalized_adjacency: np.ndarray,
    masks: tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
) -> tuple[WeightedGNN, dict[str, list[float]], dict]:
    seed_everything(config.SEED)
    train_mask_np, val_mask_np, test_mask_np = masks

    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    adjacency = torch.as_tensor(
        normalized_adjacency, dtype=torch.float32, device=device
    )
    train_mask = torch.as_tensor(train_mask_np, dtype=torch.bool, device=device)
    val_mask = torch.as_tensor(val_mask_np, dtype=torch.bool, device=device)
    test_mask = torch.as_tensor(test_mask_np, dtype=torch.bool, device=device)

    model = WeightedGNN(
        in_dim=features.shape[1],
        hidden_dim=config.HIDDEN_DIM,
        n_hidden_layers=config.N_LAYERS,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
    )
    criterion = make_loss(config.LOSS_FN, y[train_mask])

    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "validation_loss": [],
        "test_loss": [],
    }
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    started_at = time.perf_counter()
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, adjacency)
        train_loss = criterion(prediction[train_mask], y[train_mask])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            prediction = model(x, adjacency)
            validation_loss = criterion(prediction[val_mask], y[val_mask])
            test_loss = criterion(prediction[test_mask], y[test_mask])

        train_value = float(train_loss.detach())
        validation_value = float(validation_loss)
        test_value = float(test_loss)
        if not all(
            math.isfinite(value)
            for value in (train_value, validation_value, test_value)
        ):
            raise RuntimeError(
                f"Loss não finita na época {epoch}: "
                f"train={train_value}, validation={validation_value}, "
                f"test={test_value}."
            )
        history["epoch"].append(epoch)
        history["train_loss"].append(train_value)
        history["validation_loss"].append(validation_value)
        history["test_loss"].append(test_value)

        if validation_value < best_validation_loss - config.MIN_DELTA:
            best_validation_loss = validation_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            config.PATIENCE > 0
            and epochs_without_improvement >= config.PATIENCE
        ):
            LOGGER.info(
                "Early stopping na época %d (melhor época: %d).",
                epoch,
                best_epoch,
            )
            break

    training_seconds = time.perf_counter() - started_at
    model.load_state_dict(best_state)
    training_info = {
        "best_epoch": best_epoch,
        "stop_epoch": history["epoch"][-1],
        "epochs_ran": len(history["epoch"]),
        "best_validation_loss": best_validation_loss,
        "training_seconds": training_seconds,
    }
    return model, history, training_info


def regression_metrics(
    target: torch.Tensor,
    prediction: torch.Tensor,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> dict[str, float | None]:
    loss_value = float(criterion(prediction, target).detach())
    target = target.detach().float().cpu()
    prediction = prediction.detach().float().cpu()
    mse = float(torch.mean((prediction - target).square()))
    mae = float(torch.mean(torch.abs(prediction - target)))
    denominator = float(torch.sum((target - target.mean()).square()))
    r2 = (
        None
        if denominator <= 1e-12
        else 1.0 - float(torch.sum((prediction - target).square())) / denominator
    )
    return {
        "loss": loss_value,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mae,
        "r2": r2,
    }


def evaluate_model(
    model: WeightedGNN,
    features: np.ndarray,
    targets: np.ndarray,
    normalized_adjacency: np.ndarray,
    masks: tuple[np.ndarray, np.ndarray, np.ndarray],
    loss_name: str,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, dict[str, float | None]]]:
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    adjacency = torch.as_tensor(
        normalized_adjacency, dtype=torch.float32, device=device
    )
    train_mask = torch.as_tensor(masks[0], dtype=torch.bool, device=device)
    criterion = make_loss(loss_name, y[train_mask])

    model.eval()
    with torch.no_grad():
        prediction = model(x, adjacency)

    split_names = ("train", "validation", "test")
    metrics = {}
    for split_name, mask_np in zip(split_names, masks):
        mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
        metrics[split_name] = regression_metrics(
            y[mask], prediction[mask], criterion
        )
    return prediction.detach().cpu().numpy(), metrics


def save_edge_table(
    path: Path,
    stations: pd.DataFrame,
    binary: np.ndarray,
    weighted: np.ndarray,
    normalized: np.ndarray,
) -> None:
    rows = []
    for node_i in range(len(stations)):
        for node_j in range(node_i + 1, len(stations)):
            if not binary[node_i, node_j]:
                continue
            rows.append(
                {
                    "node_i": node_i,
                    "station_i": stations.loc[node_i, "station"],
                    "node_j": node_j,
                    "station_j": stations.loc[node_j, "station"],
                    "raw_weight": weighted[node_i, node_j],
                    "normalized_weight_i_to_j": normalized[node_i, node_j],
                    "normalized_weight_j_to_i": normalized[node_j, node_i],
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_loss_curves(
    history: dict[str, list[float]],
    config: ExperimentConfig,
    training_info: dict,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    epochs = history["epoch"]
    ax.plot(epochs, history["train_loss"], label="Treino", linewidth=2)
    ax.plot(epochs, history["validation_loss"], label="Validação", linewidth=2)
    ax.plot(epochs, history["test_loss"], label="Teste", linewidth=2, alpha=0.8)
    ax.axvline(
        training_info["best_epoch"],
        color="black",
        linestyle="--",
        linewidth=1.3,
        label=f"Melhor época: {training_info['best_epoch']}",
    )

    positive_losses = [
        value
        for key in ("train_loss", "validation_loss", "test_loss")
        for value in history[key]
        if value > 0
    ]
    if positive_losses and max(positive_losses) / min(positive_losses) > 100:
        ax.set_yscale("log")

    ax.set_title(
        f"Curvas de loss | K={config.K}, hidden={config.HIDDEN_DIM}, "
        f"layers={config.N_LAYERS}, loss={config.LOSS_FN}"
    )
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_adjacency_matrices(
    weighted: np.ndarray,
    normalized: np.ndarray,
    config: ExperimentConfig,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, matrix, title in (
        (axes[0], weighted, "Adjacência ponderada"),
        (axes[1], normalized, "Adjacência normalizada"),
    ):
        image = ax.imshow(matrix, cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Nó j")
        ax.set_ylabel("Nó i")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Weighted GNN | K={config.K}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_knn_graph(
    stations: pd.DataFrame,
    binary: np.ndarray,
    weighted: np.ndarray,
    split: np.ndarray,
    config: ExperimentConfig,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    latitudes = stations["latitude"].to_numpy()
    longitudes = stations["longitude"].to_numpy()

    edge_weights = weighted[binary]
    max_edge_weight = float(edge_weights.max()) if edge_weights.size else 1.0
    edge_alpha = 0.16 if config.K > 10 else 0.42
    for node_i in range(len(stations)):
        for node_j in range(node_i + 1, len(stations)):
            if not binary[node_i, node_j]:
                continue
            relative_weight = weighted[node_i, node_j] / max_edge_weight
            ax.plot(
                [longitudes[node_i], longitudes[node_j]],
                [latitudes[node_i], latitudes[node_j]],
                color="0.35",
                linewidth=0.35 + 1.2 * relative_weight,
                alpha=edge_alpha,
                zorder=1,
            )

    split_style = {
        "train": ("tab:blue", "Treino"),
        "validation": ("tab:orange", "Validação"),
        "test": ("tab:red", "Teste"),
    }
    for split_name, (color, label) in split_style.items():
        mask = split == split_name
        ax.scatter(
            longitudes[mask],
            latitudes[mask],
            s=52,
            color=color,
            label=label,
            zorder=3,
        )

    target_node = config.TARGET_NODE
    ax.scatter(
        longitudes[target_node],
        latitudes[target_node],
        s=300,
        marker="*",
        color="black",
        edgecolors="white",
        linewidths=0.8,
        label=f"Nó destacado: {target_node}",
        zorder=5,
    )
    for node, (longitude, latitude) in enumerate(zip(longitudes, latitudes)):
        ax.annotate(
            str(node),
            (longitude, latitude),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
            alpha=0.8,
            zorder=6,
        )

    ax.set_title(f"Grafo KNN ponderado das estações | K={config.K}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_residual_graph(
    stations: pd.DataFrame,
    binary: np.ndarray,
    weighted: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    split: np.ndarray,
    target_date: str,
    run_description: str,
    output_path: Path,
) -> None:
    """Plota o grafo com a magnitude e o sinal dos resíduos por estação."""
    residuals = np.asarray(targets) - np.asarray(predictions)
    absolute_residuals = np.abs(residuals)
    color_limit = max(float(absolute_residuals.max()), 1e-6)
    color_map = LinearSegmentedColormap.from_list(
        "residual_white_to_red",
        ("#ffffff", "#ff0000"),
    )
    normalization = Normalize(vmin=0.0, vmax=color_limit)

    latitudes = stations["latitude"].to_numpy(dtype=float)
    longitudes = stations["longitude"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(15, 12))

    edge_weights = weighted[binary]
    max_edge_weight = float(edge_weights.max()) if edge_weights.size else 1.0
    possible_directed_edges = max(len(stations) * (len(stations) - 1), 1)
    edge_density = edge_weights.size / possible_directed_edges
    edge_alpha = max(0.025, 0.28 / (1.0 + 9.0 * edge_density))
    for node_i in range(len(stations)):
        for node_j in range(node_i + 1, len(stations)):
            if not binary[node_i, node_j]:
                continue
            relative_weight = weighted[node_i, node_j] / max_edge_weight
            ax.plot(
                [longitudes[node_i], longitudes[node_j]],
                [latitudes[node_i], latitudes[node_j]],
                color="0.55",
                linewidth=0.18 + 0.55 * relative_weight,
                alpha=edge_alpha,
                zorder=1,
            )

    split_markers = {
        "train": ("o", "Treino"),
        "validation": ("s", "Validação"),
        "test": ("^", "Teste"),
    }
    for split_name, (marker, label) in split_markers.items():
        mask = split == split_name
        ax.scatter(
            longitudes[mask],
            latitudes[mask],
            c=absolute_residuals[mask],
            cmap=color_map,
            norm=normalization,
            marker=marker,
            s=115,
            edgecolors="black",
            linewidths=0.75,
            label=label,
            zorder=3,
        )

    for node, (longitude, latitude, residual) in enumerate(
        zip(longitudes, latitudes, residuals)
    ):
        ax.annotate(
            f"{node}: {residual:+.1f}",
            (longitude, latitude),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.4,
            color="black",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.62,
            },
            zorder=4,
        )

    scalar_map = plt.cm.ScalarMappable(norm=normalization, cmap=color_map)
    scalar_map.set_array([])
    colorbar = fig.colorbar(scalar_map, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("|Resíduo| (mm)")

    ax.set_title(
        f"Resíduos por estação | {target_date} | {run_description}\n"
        "Rótulos: nó: resíduo assinado (real - predito), em mm"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def format_parameter_value(value: object) -> str:
    if isinstance(value, float):
        if value != 0.0 and abs(value) < 1e-3:
            return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        return f"{value:g}"
    return str(value)


def format_pivot_description(
    config: ExperimentConfig,
    pivot_parameter: str,
) -> str:
    parameter_name = pivot_parameter.strip().upper()
    if parameter_name not in RUN_PARAMETER_NAMES:
        raise ValueError(f"PIVOT_PARAMETER desconhecido: {pivot_parameter!r}.")
    display_name = "WD" if parameter_name == "WEIGHT_DECAY" else parameter_name
    value = format_parameter_value(getattr(config, parameter_name))
    return f"{display_name}={value}"


def plot_prediction_scatter(
    targets: np.ndarray,
    predictions: np.ndarray,
    split: np.ndarray,
    metrics: dict,
    config: ExperimentConfig,
    pivot_parameter: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    split_style = {
        "train": ("tab:blue", "Treino"),
        "validation": ("tab:orange", "Validação"),
        "test": ("tab:red", "Teste"),
    }
    for split_name, (color, label) in split_style.items():
        mask = split == split_name
        ax.scatter(
            targets[mask],
            predictions[mask],
            color=color,
            label=label,
            alpha=0.82,
            s=58,
        )

    lower = float(min(targets.min(), predictions.min()))
    upper = float(max(targets.max(), predictions.max()))
    margin = max((upper - lower) * 0.05, 0.5)
    limits = (lower - margin, upper + margin)
    ax.plot(limits, limits, "k--", linewidth=1.3, label="Predição perfeita")
    ax.set_xlim(limits)
    ax.set_ylim(limits)

    test_metrics = metrics["test"]
    r2_text = "n/a" if test_metrics["r2"] is None else f"{test_metrics['r2']:.3f}"
    ax.text(
        0.04,
        0.96,
        f"Teste: RMSE={test_metrics['rmse']:.3f} | "
        f"MAE={test_metrics['mae']:.3f} | R²={r2_text}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    ax.set_title(
        f"Precipitação real vs. predita | {config.TARGET_DATE} | "
        f"{format_pivot_description(config, pivot_parameter)}"
    )
    ax.set_xlabel("Real (mm/dia)")
    ax.set_ylabel("Predito (mm/dia)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_predictions_by_node(
    targets: np.ndarray,
    predictions: np.ndarray,
    split: np.ndarray,
    config: ExperimentConfig,
    pivot_parameter: str,
    output_path: Path,
) -> None:
    node_ids = np.arange(len(targets))
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(node_ids, targets, color="black", linewidth=1.7, label="Real")
    ax.plot(
        node_ids,
        predictions,
        color="tab:green",
        linewidth=1.7,
        label="Predito",
    )

    split_colors = {
        "train": "tab:blue",
        "validation": "tab:orange",
        "test": "tab:red",
    }
    for split_name, color in split_colors.items():
        mask = split == split_name
        ax.scatter(
            node_ids[mask],
            predictions[mask],
            color=color,
            s=36,
            alpha=0.85,
            label=f"Predito — {split_name}",
            zorder=3,
        )

    ax.axvline(
        config.TARGET_NODE,
        color="0.35",
        linestyle="--",
        linewidth=1.2,
        label=f"Nó destacado: {config.TARGET_NODE}",
    )
    ax.set_title(
        f"Precipitação por estação | {config.TARGET_DATE} | "
        f"{format_pivot_description(config, pivot_parameter)}"
    )
    ax.set_xlabel("Índice do nó/estação")
    ax.set_ylabel("Precipitação (mm/dia)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_artifacts(
    output_dir: Path,
    config: ExperimentConfig,
    pivot_parameter: str,
    stations: pd.DataFrame,
    raw_features: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    split: np.ndarray,
    masks: tuple[np.ndarray, np.ndarray, np.ndarray],
    binary: np.ndarray,
    weighted: np.ndarray,
    normalized: np.ndarray,
    model: WeightedGNN,
    history: dict[str, list[float]],
    training_info: dict,
    metrics: dict,
    experiment_started_at: float,
) -> dict:
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "history.csv", index=False)

    predictions_frame = stations.copy()
    predictions_frame.insert(0, "node", np.arange(len(stations)))
    predictions_frame["split"] = split
    predictions_frame["vertical_velocity_mean_850hpa"] = raw_features[:, 0]
    predictions_frame["temperature_mean_celsius"] = raw_features[:, 1]
    predictions_frame["target_precipitation_mm"] = targets
    predictions_frame["predicted_precipitation_mm"] = predictions
    predictions_frame["residual_mm"] = targets - predictions
    predictions_frame["absolute_error_mm"] = np.abs(targets - predictions)
    predictions_frame.to_csv(output_dir / "predictions.csv", index=False)

    save_edge_table(
        output_dir / "edge_weights.csv",
        stations,
        binary,
        weighted,
        normalized,
    )

    checkpoint = {
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "model_config": {
            "in_dim": raw_features.shape[1],
            "hidden_dim": config.HIDDEN_DIM,
            "n_hidden_layers": config.N_LAYERS,
            "out_dim": 1,
        },
        "feature_names": [
            "vertical_velocity_mean_850hpa",
            "temperature_mean_celsius",
        ],
        "feature_mean": torch.as_tensor(feature_mean),
        "feature_std": torch.as_tensor(feature_std),
        "normalized_adjacency": torch.as_tensor(normalized),
        "config": asdict(config),
    }
    torch.save(checkpoint, output_dir / "model.pt")

    plot_loss_curves(
        history,
        config,
        training_info,
        output_dir / "loss_curves.png",
    )
    plot_adjacency_matrices(
        weighted,
        normalized,
        config,
        output_dir / "adjacency_heatmaps.png",
    )
    plot_knn_graph(
        stations,
        binary,
        weighted,
        split,
        config,
        output_dir / "knn_graph.png",
    )
    plot_residual_graph(
        stations,
        binary,
        weighted,
        targets,
        predictions,
        split,
        config.TARGET_DATE,
        format_pivot_description(config, pivot_parameter),
        output_dir / "residual_graph.png",
    )
    plot_prediction_scatter(
        targets,
        predictions,
        split,
        metrics,
        config,
        pivot_parameter,
        output_dir / "prediction_scatter.png",
    )
    plot_predictions_by_node(
        targets,
        predictions,
        split,
        config,
        pivot_parameter,
        output_dir / "predictions_by_node.png",
    )

    target_node = config.TARGET_NODE
    total_seconds = time.perf_counter() - experiment_started_at
    summary = {
        "status": "completed",
        "completed_at": datetime.now().astimezone().isoformat(),
        "config": asdict(config),
        "data": {
            "number_of_nodes": len(stations),
            "number_of_undirected_edges": int(np.triu(binary, k=1).sum()),
            "split_counts": {
                "train": int(masks[0].sum()),
                "validation": int(masks[1].sum()),
                "test": int(masks[2].sum()),
            },
            "feature_mean_on_train": {
                "vertical_velocity_mean_850hpa": float(feature_mean[0]),
                "temperature_mean_celsius": float(feature_mean[1]),
            },
            "feature_std_on_train": {
                "vertical_velocity_mean_850hpa": float(feature_std[0]),
                "temperature_mean_celsius": float(feature_std[1]),
            },
        },
        "training": training_info,
        "metrics": metrics,
        "target_node": {
            "node": target_node,
            "station": stations.loc[target_node, "station"],
            "split": split[target_node],
            "target_precipitation_mm": float(targets[target_node]),
            "predicted_precipitation_mm": float(predictions[target_node]),
            "absolute_error_mm": float(
                abs(targets[target_node] - predictions[target_node])
            ),
        },
        "total_seconds": total_seconds,
        "artifacts": [
            "config.json",
            "summary.json",
            "execution.log",
            "history.csv",
            "predictions.csv",
            "edge_weights.csv",
            "model.pt",
            "loss_curves.png",
            "adjacency_heatmaps.png",
            "knn_graph.png",
            "residual_graph.png",
            "prediction_scatter.png",
            "predictions_by_node.png",
        ],
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def latex_escape(value: object) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def comparison_value(value: object) -> object:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "n/a"
    return value


def parameter_options(value: object) -> list[object]:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Uma lista de parâmetros não pode estar vazia.")
        return list(value)
    return [value]


def build_parameter_combinations(
    parameters: dict[str, object],
) -> list[dict[str, object]]:
    missing = [name for name in RUN_PARAMETER_NAMES if name not in parameters]
    if missing:
        raise ValueError(f"Parâmetros obrigatórios ausentes: {', '.join(missing)}.")
    values = [parameter_options(parameters[name]) for name in RUN_PARAMETER_NAMES]
    return [
        dict(zip(RUN_PARAMETER_NAMES, combination))
        for combination in itertools.product(*values)
    ]


def ensure_residual_graph_from_artifacts(
    run_dir: Path,
    config: dict,
    pivot_parameter: str,
) -> Path | None:
    """Cria o grafo residual de runs antigas a partir dos CSVs salvos."""
    output_path = run_dir / "residual_graph.png"
    source_modified_at = Path(__file__).stat().st_mtime
    if output_path.exists() and output_path.stat().st_mtime >= source_modified_at:
        return output_path

    predictions_path = run_dir / "predictions.csv"
    edges_path = run_dir / "edge_weights.csv"
    if not predictions_path.exists() or not edges_path.exists():
        return None

    predictions = pd.read_csv(predictions_path).sort_values("node")
    required_prediction_columns = {
        "node",
        "station",
        "latitude",
        "longitude",
        "split",
        "target_precipitation_mm",
        "predicted_precipitation_mm",
    }
    missing_prediction_columns = required_prediction_columns.difference(
        predictions.columns
    )
    if missing_prediction_columns:
        raise ValueError(
            f"Colunas ausentes em {predictions_path}: "
            f"{sorted(missing_prediction_columns)}."
        )

    node_ids = predictions["node"].to_numpy(dtype=int)
    expected_node_ids = np.arange(len(predictions))
    if not np.array_equal(node_ids, expected_node_ids):
        raise ValueError(
            f"A coluna node de {predictions_path} deve conter 0..N-1 sem lacunas."
        )

    edge_table = pd.read_csv(edges_path)
    required_edge_columns = {"node_i", "node_j", "raw_weight"}
    missing_edge_columns = required_edge_columns.difference(edge_table.columns)
    if missing_edge_columns:
        raise ValueError(
            f"Colunas ausentes em {edges_path}: {sorted(missing_edge_columns)}."
        )

    n_nodes = len(predictions)
    binary = np.zeros((n_nodes, n_nodes), dtype=bool)
    weighted = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for edge in edge_table.itertuples(index=False):
        node_i = int(edge.node_i)
        node_j = int(edge.node_j)
        raw_weight = float(edge.raw_weight)
        binary[node_i, node_j] = True
        binary[node_j, node_i] = True
        weighted[node_i, node_j] = raw_weight
        weighted[node_j, node_i] = raw_weight
    np.fill_diagonal(weighted, 1.0)

    display_name = "WD" if pivot_parameter == "WEIGHT_DECAY" else pivot_parameter
    run_description = (
        f"{display_name}={format_parameter_value(config[pivot_parameter])}"
    )
    stations = predictions.loc[:, ["station", "latitude", "longitude"]].copy()
    plot_residual_graph(
        stations=stations,
        binary=binary,
        weighted=weighted,
        targets=predictions["target_precipitation_mm"].to_numpy(dtype=float),
        predictions=predictions["predicted_precipitation_mm"].to_numpy(dtype=float),
        split=predictions["split"].to_numpy(dtype=object),
        target_date=parse_target_date(config["TARGET_DATE"]),
        run_description=run_description,
        output_path=output_path,
    )
    return output_path


def generate_comparison_report(output_root: Path, pivot_parameter: str) -> Path:
    pivot_parameter = pivot_parameter.strip().upper()
    valid_parameters = set(RUN_PARAMETER_NAMES)
    if pivot_parameter not in valid_parameters:
        choices = ", ".join(sorted(valid_parameters))
        raise ValueError(
            f"PIVOT_PARAMETER={pivot_parameter!r} não existe na configuração. "
            f"Opções: {choices}."
        )

    runs = []
    for run_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        config_path = run_dir / "config.json"
        summary_path = run_dir / "summary.json"
        if not config_path.exists() or not summary_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if summary.get("status") != "completed" or pivot_parameter not in config:
            continue
        try:
            residual_graph_path = ensure_residual_graph_from_artifacts(
                run_dir,
                config,
                pivot_parameter,
            )
        except (OSError, TypeError, ValueError) as exc:
            LOGGER.warning(
                "Não foi possível criar o grafo residual de %s: %s",
                run_dir.name,
                exc,
            )
            residual_graph_path = None
        test_metrics = summary.get("metrics", {}).get("test", {})
        runs.append(
            {
                "RUN_DIR": run_dir.name,
                "PIVOT": comparison_value(config[pivot_parameter]),
                "RMSE": test_metrics.get("rmse"),
                "MAE": test_metrics.get("mae"),
                "R2": test_metrics.get("r2"),
                "RESIDUAL_GRAPH": residual_graph_path is not None,
            }
        )

    if not runs:
        raise ValueError("Nenhuma run concluída foi encontrada para o relatório.")

    def sort_key(row: dict) -> tuple[int, object]:
        value = row["PIVOT"]
        return (0, value) if isinstance(value, (int, float)) else (1, str(value))

    runs.sort(key=sort_key)
    best_values = {}
    for metric in ("RMSE", "MAE", "R2"):
        values = [float(row[metric]) for row in runs if row[metric] is not None]
        best_values[metric] = (max(values) if metric == "R2" else min(values))

    table_rows = []
    for row in runs:
        cells = [latex_escape(row["PIVOT"])]
        for metric in ("RMSE", "MAE", "R2"):
            value = row[metric]
            if value is None:
                cells.append("n/a")
            else:
                text_value = f"{float(value):.6f}"
                if np.isclose(float(value), best_values[metric]):
                    text_value = rf"\textbf{{{text_value}}}"
                cells.append(text_value)
        table_rows.append(" & ".join(cells) + r" \\")

    x_ticks = ",".join(str(index) for index in range(len(runs)))
    x_labels = ",".join(latex_escape(row["PIVOT"]) for row in runs)
    plot_texts = []
    for metric in ("RMSE", "MAE", "R2"):
        coordinates = " ".join(
            f"({index},{float(row[metric]):.8g})"
            for index, row in enumerate(runs)
            if row[metric] is not None
        )
        plot_texts.append(
            rf"""\nextgroupplot[title={{{metric}}}, xlabel={{{latex_escape(pivot_parameter)}}}, ylabel={{{metric}}}, xtick={{{x_ticks}}}, xticklabels={{{x_labels}}}, x tick label style={{rotate=45, anchor=east}}]
\addplot+[mark=*, thick] coordinates {{{coordinates}}};"""
        )

    table_text = "\n".join(table_rows)
    plot_text = "\n".join(plot_texts)
    visual_blocks = []
    for row in runs:
        run_label = latex_escape(row["RUN_DIR"])
        scatter_path = latex_escape(f"{row['RUN_DIR']}/prediction_scatter.png")
        xy_path = latex_escape(f"{row['RUN_DIR']}/predictions_by_node.png")
        visual_blocks.append(
            rf"""\begin{{minipage}}[t]{{0.48\textwidth}}
\centering
\includegraphics[width=\textwidth]{{{scatter_path}}}
\captionof{{figure}}{{{run_label} --- real vs. predicted}}
\end{{minipage}}
\hfill
\begin{{minipage}}[t]{{0.48\textwidth}}
\centering
\includegraphics[width=\textwidth]{{{xy_path}}}
\captionof{{figure}}{{{run_label} --- plot XY por estação}}
\end{{minipage}}\par\vspace{{0.4cm}}"""
        )
    visual_text = "\n".join(visual_blocks)
    residual_blocks = []
    for row in runs:
        if not row["RESIDUAL_GRAPH"]:
            continue
        run_label = latex_escape(row["RUN_DIR"])
        residual_path = latex_escape(f"{row['RUN_DIR']}/residual_graph.png")
        residual_blocks.append(
            rf"""\begin{{figure}}[H]
\centering
\includegraphics[width=0.94\textwidth]{{{residual_path}}}
\caption{{{run_label} --- resíduos por estação. A cor representa $|\mathrm{{real}}-\mathrm{{predito}}|$; os rótulos mostram nó e resíduo assinado, em mm.}}
\end{{figure}}"""
        )
    residual_text = "\n".join(residual_blocks)
    loss_plot_parts = []
    loss_legend_parts = []
    loss_colors = (
        "blue", "red", "teal", "orange", "violet", "brown", "magenta", "cyan"
    )
    for index, row in enumerate(runs):
        history_path = output_root / row["RUN_DIR"] / "history.csv"
        if not history_path.exists():
            continue
        history = pd.read_csv(history_path)
        max_loss_points = 200
        if len(history) > max_loss_points:
            sampled_indices = np.linspace(
                0, len(history) - 1, max_loss_points, dtype=int
            )
            history = history.iloc[np.unique(sampled_indices)]
        color = loss_colors[index % len(loss_colors)]
        pivot_display_name = "WD" if pivot_parameter == "WEIGHT_DECAY" else pivot_parameter
        pivot_value = format_parameter_value(row["PIVOT"])
        run_description = f"{pivot_display_name}={pivot_value}"
        run_label = latex_escape(run_description)
        train_coordinates = " ".join(
            f"({int(epoch)},{float(loss):.8g})"
            for epoch, loss in zip(history["epoch"], history["train_loss"])
        )
        validation_coordinates = " ".join(
            f"({int(epoch)},{float(loss):.8g})"
            for epoch, loss in zip(history["epoch"], history["validation_loss"])
        )
        loss_plot_parts.append(
            rf"""\addplot+[solid, color={color}, line width=0.8pt, mark=none] coordinates {{{train_coordinates}}};
\addplot+[densely dotted, color={color}, line width=0.8pt, mark=none] coordinates {{{validation_coordinates}}};"""
        )
        loss_legend_parts.append(
            rf"""\begin{{minipage}}[t]{{0.48\textwidth}}
\begin{{tikzpicture}}[baseline=-0.5ex]
\draw[color={color}, line width=0.8pt] (0,0.10) -- (0.9,0.10);
\draw[color={color}, densely dotted, line width=0.8pt] (0,-0.10) -- (0.9,-0.10);
\end{{tikzpicture}}\hspace{{0.15cm}}\scriptsize {run_label}
\end{{minipage}}"""
        )
    loss_plot_text = "\n".join(loss_plot_parts)
    loss_legend_text = "\n".join(loss_legend_parts)
    report = rf"""\documentclass{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{graphicx}}
\usepackage{{caption}}
\usepackage{{tikz}}
\usepackage{{pgfplots}}
\usepgfplotslibrary{{groupplots}}
\pgfplotsset{{compat=1.18}}
\title{{Comparison Report}}
\begin{{document}}
\maketitle

Pivot parameter: \texttt{{{latex_escape(pivot_parameter)}}}

\begin{{table}}[H]
\centering
\caption{{Test-set metrics for each run. The best value in each metric is bold.}}
\begin{{tabular}}{{rrrr}}
\toprule
{latex_escape(pivot_parameter)} & RMSE & MAE & R2 \\
\midrule
{table_text}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=3 by 1, horizontal sep=1.2cm}}, width=0.31\textwidth, height=6cm]
{plot_text}
\end{{groupplot}}
\end{{tikzpicture}}
\caption{{Test-set metrics as a function of {latex_escape(pivot_parameter)}.}}
\end{{figure}}

\newpage
\section*{{Evolução da loss por época}}
\begin{{figure}}[H]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[width=0.95\textwidth, height=9cm, xlabel={{Época}}, ylabel={{Loss}}, grid=major, major grid style={{draw=gray!20, line width=0.2pt}}, axis line style={{draw=gray!60, line width=0.4pt}}, tick style={{draw=gray!60, line width=0.4pt}}]
{loss_plot_text}
\end{{axis}}
\end{{tikzpicture}}
\par\vspace{{0.25cm}}
\noindent
{loss_legend_text}
\caption{{Linhas sólidas: treino. Linhas pontilhadas: validação. Cada run usa uma cor.}}
\end{{figure}}

\newpage
\section*{{Resíduos por estação}}
Escala de cores de branco (menor erro) a vermelho (maior erro), em mm.
O resíduo assinado é definido como valor real menos valor predito.
{residual_text}

\newpage
\section*{{Plots de cada run}}
{visual_text}
\end{{document}}
"""
    report_path = output_root / "comparison_report.tex"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def run(
    parameter_values: dict[str, object],
    batch_output_root: Path,
    project_dir: Path,
    data_dir: Path,
    pivot_parameter: str,
) -> Path:
    parameters = parameter_values

    target_date = parameters["TARGET_DATE"]
    k = parameters["K"]
    hidden_dim = parameters["HIDDEN_DIM"]
    n_layers = parameters["N_LAYERS"]
    epochs = parameters["EPOCHS"]
    lr = parameters["LR"]
    weight_decay = parameters["WEIGHT_DECAY"]
    loss_fn = parameters["LOSS_FN"]
    seed = parameters["SEED"]
    patience = parameters["PATIENCE"]
    min_delta = parameters["MIN_DELTA"]
    target_node = parameters["TARGET_NODE"]
    standardize_features_value = parameters["STANDARDIZE_FEATURES"]
    requested_device = parameters["DEVICE"]

    device = resolve_device(requested_device)

    stations = load_stations(find_catalog(data_dir))
    if k >= len(stations):
        raise ValueError(
            f"K={k} deve ser menor que o número de estações ({len(stations)})."
        )
    if target_node >= len(stations):
        raise ValueError(
            f"TARGET_NODE={target_node} está fora do intervalo "
            f"0..{len(stations) - 1}."
        )

    config = ExperimentConfig(
        TARGET_DATE=parse_target_date(target_date),
        K=positive_int(k),
        HIDDEN_DIM=positive_int(hidden_dim),
        N_LAYERS=positive_int(n_layers),
        EPOCHS=positive_int(epochs),
        LR=positive_float(lr),
        WEIGHT_DECAY=nonnegative_float(weight_decay),
        LOSS_FN=parse_loss_name(loss_fn),
        SEED=int(seed),
        PATIENCE=nonnegative_int(patience),
        MIN_DELTA=nonnegative_float(min_delta),
        TARGET_NODE=nonnegative_int(target_node),
        STANDARDIZE_FEATURES=bool(standardize_features_value),
        DEVICE=str(device),
    )

    output_dir = create_output_directory(batch_output_root)
    configure_logging(output_dir)
    save_json(
        output_dir / "config.json",
        {
            **asdict(config),
            "PROJECT_DIR": str(project_dir),
            "DATA_DIR": str(data_dir),
            "OUTPUT_DIR": str(output_dir),
            "CREATED_AT": datetime.now().astimezone().isoformat(),
        },
    )

    started_at = time.perf_counter()
    LOGGER.info("Experimento: %s", output_dir.name)
    LOGGER.info("Configuração: %s", asdict(config))
    LOGGER.info("Dispositivo: %s", device)

    try:
        raw_features, targets = load_day_data(
            data_dir,
            config.TARGET_DATE,
            stations,
        )
        split, train_mask, val_mask, test_mask = split_nodes(
            len(stations), config.SEED
        )
        masks = (train_mask, val_mask, test_mask)
        LOGGER.info(
            "Split por nós: treino=%d, validação=%d, teste=%d.",
            int(train_mask.sum()),
            int(val_mask.sum()),
            int(test_mask.sum()),
        )

        binary, weighted, normalized = build_weighted_graph(stations, config.K)
        LOGGER.info(
            "Grafo criado: %d nós e %d arestas não direcionadas.",
            len(stations),
            int(np.triu(binary, k=1).sum()),
        )
        features, feature_mean, feature_std = standardize_features(
            raw_features,
            train_mask,
            config.STANDARDIZE_FEATURES,
        )

        model, history, training_info = train_model(
            config,
            features,
            targets,
            normalized,
            masks,
            device,
        )
        predictions, metrics = evaluate_model(
            model,
            features,
            targets,
            normalized,
            masks,
            config.LOSS_FN,
            device,
        )
        summary = save_artifacts(
            output_dir=output_dir,
            config=config,
            pivot_parameter=pivot_parameter,
            stations=stations,
            raw_features=raw_features,
            feature_mean=feature_mean,
            feature_std=feature_std,
            targets=targets,
            predictions=predictions,
            split=split,
            masks=masks,
            binary=binary,
            weighted=weighted,
            normalized=normalized,
            model=model,
            history=history,
            training_info=training_info,
            metrics=metrics,
            experiment_started_at=started_at,
        )

        test_metrics = summary["metrics"]["test"]
        LOGGER.info(
            "Concluído em %.2fs | teste RMSE=%.4f, MAE=%.4f, R²=%s.",
            summary["total_seconds"],
            test_metrics["rmse"],
            test_metrics["mae"],
            (
                "n/a"
                if test_metrics["r2"] is None
                else f"{test_metrics['r2']:.4f}"
            ),
        )
        LOGGER.info("Outputs salvos em: %s", output_dir)
    except Exception as exc:
        LOGGER.exception("O experimento falhou.")
        save_json(
            output_dir / "failure.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_at": datetime.now().astimezone().isoformat(),
            },
        )
        raise

    return output_dir


def execute_experiments(
    parameters: dict[str, object],
    *,
    project_dir: Path,
    data_dir: Path,
    output_root: Path,
    comparison_report: bool,
    pivot_parameter: str,
) -> Path:
    """Executa todas as combinações e retorna a pasta do lote criado."""
    project_dir = project_dir.resolve()
    resolved_data_dir = resolve_from_project(data_dir, project_dir)
    resolved_output_root = resolve_from_project(output_root, project_dir)
    combinations = build_parameter_combinations(parameters)
    batch_output_root = create_batch_directory(
        resolved_output_root,
        parameters["TARGET_DATE"],
    )

    for parameter_values in combinations:
        run(
            parameter_values,
            batch_output_root=batch_output_root,
            project_dir=project_dir,
            data_dir=resolved_data_dir,
            pivot_parameter=pivot_parameter,
        )

    if comparison_report:
        report_path = generate_comparison_report(
            batch_output_root,
            pivot_parameter,
        )
        LOGGER.info("Relatório de comparação salvo em: %s", report_path)

    return batch_output_root
