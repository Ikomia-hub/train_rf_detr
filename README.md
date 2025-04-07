<div align="center">
  <img src="images/icon.png" alt="Algorithm icon">
  <h1 align="center">train_rf_detr</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/train_rf_detr">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/train_rf_detr">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/train_rf_detr/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/train_rf_detr.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Train D-FINE object detection models.

Following best practices recommended by the [official repository](https://github.com/Peterande/D-FINE), this algorithm utilizes the Objects365 pre-trained model as a foundation for fine-tuning, enabling optimized performance for custom object detection tasks.

![Desk object detection](https://raw.githubusercontent.com/Ikomia-hub/train_rf_detr/main/images/output.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow
```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()    

# Add dataset loader
coco = wf.add_task(name="dataset_coco")

coco.set_parameters({
    "json_file": "path/to/json/annotation/file",
    "image_folder": "path/to/image/folder",
    "task": "detection",
}) 

# Add training algorithm
train = wf.add_task(name="train_rf_detr", auto_connect=True)

# Launch your training on your data
wf.run()
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).
- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).


## :pencil: Set algorithm parameters
- `model_name` (str) - default 'rf-detr-base': Name of the RF-DETR pre-trained model. Other model available:
    - rf-detr-large
- `batch_size` (int) - default '8': Number of samples processed before the model is updated.
- `epochs` (int) - default '100': Number of complete passes through the training dataset.
- `dataset_split_ratio` (float) – default '0.9': Divide the dataset into train and evaluation sets ]0, 1[.
- `input_size` (int) - default '560': Size of the input image.
- `weight_decay` (float) - default '0.000125': Amount of weight decay, regularization method.
- `workers` (int) - default '0': Number of worker threads for data loading (per RANK if DDP).
- `lr` (float) - default '0.00025': Initial learning rate. Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated.
- `lr_encoder` (float) - default '1.5e-4': Separate learning rate for the encoder parameters. Allows fine-tuning at a different rate than the rest of the model.
- `output_folder` (str, *optional*): path to where the model will be saved. 
- `early_stopping` (bool) - default 'False': Whether to enable early stopping during training. This stops training if performance stops improving after a certain number of epochs.
- `early_stopping_patience`(int) - default '10': Number of consecutive validation checks with no improvement before early stopping is triggered. Only applicable if `early_stopping=True`.


**Parameters** should be in **strings format**  when added to the dictionary.
```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()    

# Add dataset loader
coco = wf.add_task(name="dataset_coco")

coco.set_parameters({
    "json_file": "path/to/json/annotation/file",
    "image_folder": "path/to/image/folder",
    "task": "detection",
}) 

# Add training algorithm
train = wf.add_task(name="train_rf_detr", auto_connect=True)

train.set_parameters({
    "model_name": "dfine_m",
    "epochs": "100",
    "batch_size": "6",
    "input_size": "560",
    "dataset_split_ratio": "0.9",
    "workers": "0",  # Recommended to set to 0 if you are using Windows
    "weight_decay": "1e-4",
    "lr": " 1e-4",
    "output_folder": "Path/To/Output/Folder", # Default folder : runs 
    "model_weight_file": "", # Optional
    
})

# Launch your training on your data
wf.run()
```
