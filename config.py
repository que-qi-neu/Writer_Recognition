from pathlib import Path

class Settings:
    ROOT_DIR = Path(__file__).resolve().parent
    DATA_DIR = ROOT_DIR / "data"
    TEST_DATA_DIR = ROOT_DIR / "data/test"
    MODEL_DIR = ROOT_DIR / "model"
    CODE_DIR = ROOT_DIR / "src"

settings = Settings()

