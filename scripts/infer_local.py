import torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer


def build_artifacts_path(project_root: Path) -> Path:
    """Build a local path to model artifacts.
    Args:
        project_root (Path): Project root directory."""
    model_path = project_root / 'artifacts' / 'distilbert-base-uncased'
    return model_path


def load_local_model(model_path: Path) -> tuple[AutoTokenizer, AutoModel]:
    """Load tokenizer and model from a local directory.
    Args:
        model_path (Path): Local directory with saved model artifacts."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def run_text_infer(
        tokenizer: AutoTokenizer,
        model: AutoModel,
        texts: list[str]
) -> torch.Tensor:
    """Run model inference for a list of texts and return pooled embeddings.
    Args:
        tokenizer (AutoTokenizer): Tokenizer for the model.
        model (AutoModel): Loaded Transformer model.
        texts (list[str]): Input texts for inference."""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(**encoded)

    pooled = outputs.last_hidden_state.mean(dim=1)
    return pooled


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]
    model_path = build_artifacts_path(project_root)

    tokenizer, model = load_local_model(model_path)

    texts = [
        'I love this product. It works great!',
        'This is the worst thing I have ever bought.',
        'It is okay, not amazing but not terrible.',
    ]

    embeddings = run_text_infer(tokenizer, model, texts)
    print(f'Embeddings shape: {embeddings.shape}')
