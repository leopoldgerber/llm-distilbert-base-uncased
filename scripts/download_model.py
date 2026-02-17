from pathlib import Path
from transformers import AutoModel, AutoTokenizer


def build_model_path(project_root: Path) -> Path:
    """Build a local path for storing model artifacts.
    Args:
        project_root (Path): Project root directory."""
    model_path = project_root / 'artifacts' / 'distilbert-base-uncased'
    model_path.mkdir(parents=True, exist_ok=True)
    return model_path


def save_hf_model(model_name: str, model_path: Path) -> Path:
    """Download and save Hugging Face model and tokenizer.
    Args:
        model_name (str): Model id on Hugging Face Hub.
        model_path (Path): Local directory to store artifacts."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)

    return model_path


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]
    model_path = build_model_path(project_root)
    saved_path = save_hf_model(
        'distilbert/distilbert-base-uncased',
        model_path
    )
    print(f'Saved to: {saved_path}')
