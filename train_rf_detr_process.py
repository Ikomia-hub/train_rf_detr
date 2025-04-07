import copy
import os
import yaml
import torch
from datetime import datetime
from ikomia.dnn import dnntrain
from ikomia import core, dataprocess, utils
from ikomia.core.task import TaskParam
from train_rf_detr.utils.ikutils import prepare_dataset
from train_rf_detr.utils.model_utils import load_model


# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------


class TrainRfDetrParam(TaskParam):

    def __init__(self):
        TaskParam.__init__(self)
        dataset_folder = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "dataset")
        self.cfg["dataset_folder"] = dataset_folder
        self.cfg["model_name"] = "rf-detr-base"
        self.cfg["model_weight_file"] = ""
        self.cfg["epochs"] = 100
        self.cfg["batch_size"] = 2
        self.cfg["grad_accum_steps"] = 4
        self.cfg["input_size"] = 560
        self.cfg["dataset_split_ratio"] = 0.9
        self.cfg["workers"] = 0
        self.cfg["weight_decay"] = 1e-4
        self.cfg["lr"] = 1e-4
        self.cfg["lr_encoder"] = 1.5e-4
        self.cfg["early_stopping"] = False
        self.cfg["early_stopping_patience"] = 10
        self.cfg["output_folder"] = os.path.dirname(
            os.path.realpath(__file__)) + "/runs/"

    def set_values(self, param_map):
        self.cfg["dataset_folder"] = str(param_map["dataset_folder"])
        self.cfg["model_name"] = str(param_map["model_name"])
        self.cfg["model_weight_file"] = str(param_map["model_weight_file"])
        self.cfg["epochs"] = int(param_map["epochs"])
        self.cfg["batch_size"] = int(param_map["batch_size"])
        self.cfg["grad_accum_steps"] = int(param_map["grad_accum_steps"])
        self.cfg["input_size"] = int(param_map["input_size"])
        self.cfg["workers"] = int(param_map["workers"])
        self.cfg["weight_decay"] = float(param_map["weight_decay"])
        self.cfg["lr"] = float(param_map["lr"])
        self.cfg["lr_encoder"] = float(param_map["lr_encoder"])
        self.cfg["early_stopping"] = utils.strtobool(
            param_map["early_stopping"])
        self.cfg["early_stopping_patience"] = int(
            param_map["early_stopping_patience"])
        self.cfg["dataset_split_ratio"] = float(
            param_map["dataset_split_ratio"])
        self.cfg["output_folder"] = str(param_map["output_folder"])


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class TrainRfDetr(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)
        # Create parameters object
        if param is None:
            self.set_param_object(TrainRfDetrParam())
        else:
            self.set_param_object(copy.deepcopy(param))
        self.device = torch.device("cpu")
        self.model_weights = None
        self.enable_tensorboard(False)
        self.enable_mlflow(False)
        self.model = None

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def run(self):
        # Main function of your algorithm
        # Call begin_task_run() for initialization
        self.begin_task_run()

        param = self.get_param_object()
        dataset_input = self.get_input(0)

        # Conversion from Ikomia dataset to YoloV8 dataset
        print("Preparing dataset...")
        # Prepare dataset
        dataset_yaml_info, class_names = prepare_dataset(dataset_input.data,
                                                         param.cfg["dataset_folder"],
                                                         param.cfg["dataset_split_ratio"]
                                                         )
        print(f"\nFinal dataset info: {dataset_yaml_info}")

        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Create output folder
        experiment_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(param.cfg["output_folder"], exist_ok=True)
        output_folder = os.path.join(
            param.cfg["output_folder"], experiment_name)
        os.makedirs(output_folder, exist_ok=True)

        # Save model name and class names to one YAML file
        # Convert YAML string to dictionary if necessary
        model_info = {
            "model_name": param.cfg["model_name"],
            "classes": class_names
        }
        with open(os.path.join(output_folder, 'class_names.yaml'), 'w', encoding='utf-8') as file:
            yaml.dump(model_info, file, allow_unicode=True)

        # Check input size
        if param.cfg["input_size"] % 56 != 0:
            print("Input size must be a multiple of 56. Adjusting...")
            param.cfg["input_size"] = param.cfg["input_size"] // 56 * 56
            print(f"Adjusted input size: {param.cfg['input_size']}")

        # Load and Train the model
        model = load_model(param)
        model.train(
            dataset_dir=dataset_yaml_info,
            epochs=param.cfg["epochs"],
            batch_size=param.cfg["batch_size"],
            num_workers=param.cfg["workers"],
            grad_accum_steps=param.cfg["grad_accum_steps"],
            lr=param.cfg["lr"],
            lr_encoder=param.cfg["lr_encoder"],
            weight_decay=param.cfg["weight_decay"],
            output_dir=output_folder)

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class TrainRfDetrFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "train_rf_detr"
        self.info.short_description = "Train RF-DETR models"
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.0.0"
        self.info.icon_path = "images/icon.png"
        self.info.authors = "Robinson, Isaac and Robicheaux, Peter and Popov, Matvei"
        self.info.article = ""
        self.info.journal = ""
        self.info.year = 2025
        self.info.license = "Apache-2.0"

        # Ikomia API compatibility
        # self.info.min_ikomia_version = "0.11.1"

        # Python compatibility
        self.info.min_python_version = "3.11.0"
        # self.info.max_python_version = "3.11.0"

        # URL of documentation
        self.info.documentation_link = "https://blog.roboflow.com/rf-detr/"

        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_rf_detr"
        self.info.original_repository = "https://github.com/roboflow/rf-detr"

        # Keywords used for search
        self.info.keywords = "DETR, object, detection, roboflow, real-time"
        self.info.algo_type = core.AlgoType.TRAIN
        self.info.algo_tasks = "OBJECT_DETECTION"

    def create(self, param=None):
        # Create algorithm object
        return TrainRfDetr(self.info.name, param)
