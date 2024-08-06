from pathlib import Path

__all__ = [
    "path",
    "GIN_PATH_MODEL",
    "GIN_PATH_DATALOADER",
    "GIN_PATH_TUNING",
    "DATA_PATH",
    "DATABASE_PATH",
    "CHECKPOINT_PATH",
    "TEST_PATH",
]

# Set up all paths
path = Path(__file__).parent.parent
GIN_PATH_MODEL = str(path / "configs" / "model_configs.gin")
GIN_PATH_DATALOADER = str(path / "configs" / "dataloader.gin")
GIN_PATH_TUNING = str(path / "configs" / "hp_tuning.gin")
DATA_PATH = path / "data"
DATABASE_PATH = DATA_PATH / "databases"
CHECKPOINT_PATH = path / "models"
TEST_PATH = path / "testing"
