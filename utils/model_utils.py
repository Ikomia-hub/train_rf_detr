import os
from train_rf_detr.rf_detr.rfdetr.util.files import download_file
from train_rf_detr.rf_detr.rfdetr import RFDETRBase, RFDETRLarge


HOSTED_MODELS = {
    "rf-detr-base": "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
    "rf-detr-base-2": "https://storage.googleapis.com/rfdetr/rf-detr-base-2.pth",
    "rf-detr-large": "https://storage.googleapis.com/rfdetr/rf-detr-large.pth"
}


def load_model(param):
    # Determine which weights file to use
    if param.cfg["model_weight_file"] and os.path.exists(param.cfg["model_weight_file"]):
        # Use custom weights file if provided and exists
        model_weights = param.cfg["model_weight_file"]
        print(f"Using custom weights file: {model_weights}")
    else:
        model_weights = download_pretrain_weights(
            param.cfg["model_name"])

    # Load the model based on the specified architecture
    if param.cfg["model_name"] == "rf-detr-base" or param.cfg["model_name"] == "rf-detr-base-2":
        model = RFDETRBase(
            resolution=param.cfg["input_size"],
            pretrain_weights=model_weights
        )
    elif param.cfg["model_name"] == "rf-detr-large":
        model = RFDETRLarge(
            resolution=param.cfg["input_size"],
            pretrain_weigths=model_weights
        )
    else:
        raise ValueError(f"Unknown model name: {param.cfg['model_name']}")

    return model


def download_pretrain_weights(model_name):
    """Download the pre-trained weights for the specified model if not already available."""
    # Ensure the weights folder exists
    model_folder = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "..", "weights")
    os.makedirs(model_folder, exist_ok=True)
    weight_filename = f"{model_name}.pth"
    weight_path = os.path.join(model_folder, weight_filename)

    # Check if the weight file exists
    if not os.path.exists(weight_path):
        if model_name in HOSTED_MODELS:
            print(f"Downloading pre-trained weights for {model_name}...")
            download_file(
                HOSTED_MODELS[model_name],
                weight_path
            )
            print(f"Download complete: {weight_path}")
            return weight_path
        else:
            raise ValueError(
                f"No pre-trained weights available for {model_name}")
    else:
        print(f"Using existing weights file: {weight_path}")
        return weight_path
