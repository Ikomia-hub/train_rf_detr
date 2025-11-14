import os

from rfdetr.detr import RFDETR, RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRBase, RFDETRLarge
from rfdetr.main import HOSTED_MODELS
from rfdetr.util.files import download_file

MODEL_CLASSES = {
    "rf-detr-nano": RFDETRNano,
    "rf-detr-small": RFDETRSmall,
    "rf-detr-medium": RFDETRMedium,
    "rf-detr-base": RFDETRBase,
    "rf-detr-large": RFDETRLarge,
}


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
        raise ValueError(f"Unsupported model architecture: {param.cfg["model_name"]}")

    model = model_class(
        resolution=param.cfg["input_size"],
        pretrain_weights=model_weights
    )

    return model


def download_pretrain_weights(model_name: str) -> str:
    """Download the pre-trained weights for the specified model if not already available."""
    # Ensure the weights folder exists
    model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")
    os.makedirs(model_folder, exist_ok=True)
    weight_filename = f"{model_name}.pth"
    weight_path = os.path.join(model_folder, weight_filename)

    # Check if the weight file exists
    if not os.path.exists(weight_path):
        if weight_filename in HOSTED_MODELS:
            print(f"Downloading pre-trained weights for {model_name}...")
            download_file(HOSTED_MODELS[weight_filename], weight_path)
            print(f"Download complete: {weight_path}")
            return weight_path
        else:
            raise ValueError(f"No pre-trained weights available for {model_name}")
    else:
        print(f"Using existing weights file: {weight_path}")
        return weight_path
