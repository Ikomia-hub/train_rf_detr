import os
import shutil
from contextlib import contextmanager
from pathlib import Path

from rfdetr.detr import RFDETR, RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRBase, RFDETRLarge

try:
    import rfdetr.main as rfdetr_main
except ImportError:
    rfdetr_main = None

try:
    from rfdetr.util.files import download_file
except ImportError:
    download_file = None

MODEL_CLASSES = {
    "rf-detr-nano": RFDETRNano,
    "rf-detr-small": RFDETRSmall,
    "rf-detr-medium": RFDETRMedium,
    "rf-detr-base": RFDETRBase,
    "rf-detr-large": RFDETRLarge,
}


@contextmanager
def _chdir(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _weights_dir():
    weights_dir = Path(__file__).resolve().parent / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    return weights_dir


def _hosted_models():
    if rfdetr_main is None:
        return {}

    return (
        getattr(rfdetr_main, "HOSTED_MODELS", None)
        or getattr(rfdetr_main, "OPEN_SOURCE_MODELS", None)
        or {}
    )


def _download_with_rfdetr_helper(filename, output_path):
    if rfdetr_main is None:
        return False

    download_pretrain_weights = getattr(rfdetr_main, "download_pretrain_weights", None)
    if download_pretrain_weights is None:
        return False

    with _chdir(output_path.parent):
        download_pretrain_weights(filename)

    return output_path.exists()


def _download_with_url(filename, output_path):
    hosted_models = _hosted_models()
    if filename not in hosted_models or download_file is None:
        return False

    download_file(hosted_models[filename], str(output_path))
    return output_path.exists()


def _copy_from_roboflow_cache(filename, output_path):
    cache_path = Path.home() / ".roboflow" / "models" / filename
    if not cache_path.exists():
        return False

    shutil.copy2(cache_path, output_path)
    print(f"Copied RF-DETR weights from {cache_path} to {output_path}")
    return True


def load_model(param) -> RFDETR:
    # Determine which weights file to use
    if param.cfg["model_weight_file"] and os.path.exists(param.cfg["model_weight_file"]):
        # Use custom weights file if provided and exists
        model_weights = param.cfg["model_weight_file"]
        print(f"Using custom weights file: {model_weights}")
    else:
        model_weights = download_pretrain_weights(param.cfg["model_name"])

    # Load the model based on the specified architecture
    model_class = MODEL_CLASSES.get(param.cfg["model_name"])
    if model_class is None:
        raise ValueError(f"Unsupported model architecture: {param.cfg['model_name']}")

    model_kwargs = {
        "resolution": param.cfg["input_size"],
    }
    if model_weights is not None:
        model_kwargs["pretrain_weights"] = model_weights

    model = model_class(**model_kwargs)

    return model


def download_pretrain_weights(model_name: str) -> str:
    """Download the pre-trained weights for the specified model if not already available."""
    weights_dir = _weights_dir()
    candidate_filenames = [
        f"{model_name}.pt",
        f"{model_name}.pth",
    ]

    for filename in candidate_filenames:
        output_path = weights_dir / filename
        if output_path.exists():
            print(f"Using existing weights file: {output_path}")
            return str(output_path)

    for filename in candidate_filenames:
        output_path = weights_dir / filename
        if _download_with_rfdetr_helper(filename, output_path):
            print(f"Downloaded RF-DETR weights to {output_path}")
            return str(output_path)

        if _download_with_url(filename, output_path):
            print(f"Downloaded RF-DETR weights to {output_path}")
            return str(output_path)

        if _copy_from_roboflow_cache(filename, output_path):
            return str(output_path)

    print(
        "Could not pre-cache RF-DETR weights in the plugin weights folder. "
        "Falling back to RF-DETR's default weight resolution."
    )
    return None
