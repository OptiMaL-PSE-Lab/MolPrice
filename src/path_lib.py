from pathlib import Path

__all__ = [
    "path",
    "gin_path_model",
    "gin_path_dataloader",
    "gin_path_tuning",
    "data_path",
    "database_path",
    "checkpoint_path",
]

# Set up all paths
path = Path(__file__).parent.parent
gin_path_model = str(path / "configs" / "model_configs.gin")
gin_path_dataloader = str(path / "configs" / "dataloader.gin")
gin_path_tuning = str(path / "configs" / "hp_tuning.gin")
data_path = path / "data"
database_path = data_path / "databases"
checkpoint_path = path / "models"
