from pathlib import Path 

path = Path(__file__)

ROOT_DIR = path.parent.parent
DATA_DIR = ROOT_DIR / "data"
SOURCE_DIR = ROOT_DIR / "src"
CONFIGS_DIR = ROOT_DIR / "configs"