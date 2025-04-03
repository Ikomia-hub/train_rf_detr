import os
if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from train_rf_detr.rf_detr.rfdetr.detr import RFDETRBase, RFDETRLarge
