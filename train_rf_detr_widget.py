from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from train_rf_detr.train_rf_detr_process import TrainRfDetrParam

# PyQt GUI framework
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt

# --------------------
# - Class which implements widget associated with the algorithm
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------


class TrainRfDetrWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = TrainRfDetrParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # Model name
        self.combo_model = pyqtutils.append_combo(
            self.grid_layout, "Model name")
        self.combo_model.addItem("rf-detr-base")
        self.combo_model.addItem("rf-detr-base-2")
        self.combo_model.addItem("rf-detr-large")
        self.combo_model.setCurrentText(self.parameters.cfg["model_name"])

        # Dataset folder
        self.browse_dataset_folder = pyqtutils.append_browse_file(
            self.grid_layout, label="Dataset folder",
            path=self.parameters.cfg["dataset_folder"],
            tooltip="Select folder",
            mode=QFileDialog.Directory
        )

        # Epochs
        self.spin_epochs = pyqtutils.append_spin(
            self.grid_layout, "Epochs", self.parameters.cfg["epochs"])

        # Batch size
        self.spin_batch = pyqtutils.append_spin(
            self.grid_layout, "Batch size", self.parameters.cfg["batch_size"])

        # Train/ val image size
        self.spin_input_size = pyqtutils.append_spin(
            self.grid_layout, "Image size", self.parameters.cfg["input_size"])

        # Train test split
        self.spin_train_test_split = pyqtutils.append_double_spin(
            self.grid_layout,
            "Split train/val",
            self.parameters.cfg["dataset_split_ratio"],
            min=0.01, max=1.0,
            step=0.05, decimals=2
        )

        # Early stopping checkbox
        self.check_early_stopping = pyqtutils.append_check(
            self.grid_layout, "Early stopping", self.parameters.cfg["early_stopping"])
        self.check_early_stopping.stateChanged.connect(
            self.on_early_stopping_changed)

        # Early stopping patience
        row = self.grid_layout.rowCount()
        self.spin_early_stopping_patience = pyqtutils.append_spin(
            self.grid_layout,
            "Early stopping patience",
            self.parameters.cfg["early_stopping_patience"]
        )
        # Retrieve the label widget from the grid layout
        self.label_early_stopping_patience = self.grid_layout.itemAtPosition(
            row, 0).widget()
        # Hide initially if early stopping is disabled
        visible = self.check_early_stopping.isChecked()
        self.label_early_stopping_patience.setVisible(visible)
        self.spin_early_stopping_patience.setVisible(visible)

        # Output folder
        self.browse_out_folder = pyqtutils.append_browse_file(
            self.grid_layout, label="Output folder",
            path=self.parameters.cfg["output_folder"],
            tooltip="Select folder",
            mode=QFileDialog.Directory
        )

        # Wrap layout for Qt
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)
        self.set_layout(layout_ptr)

    def on_early_stopping_changed(self, state):
        # Toggle visibility of patience widgets
        enabled = (state == Qt.Checked)
        self.label_early_stopping_patience.setVisible(enabled)
        self.spin_early_stopping_patience.setVisible(enabled)

    def on_apply(self):
        # Apply button clicked slot
        self.parameters.cfg["model_name"] = self.combo_model.currentText()
        self.parameters.cfg["dataset_folder"] = self.browse_dataset_folder.path
        self.parameters.cfg["epochs"] = self.spin_epochs.value()
        self.parameters.cfg["input_size"] = self.spin_input_size.value()
        self.parameters.cfg["batch_size"] = self.spin_batch.value()
        self.parameters.cfg["dataset_split_ratio"] = self.spin_train_test_split.value(
        )
        self.parameters.cfg["early_stopping"] = self.check_early_stopping.isChecked(
        )
        self.parameters.cfg["early_stopping_patience"] = self.spin_early_stopping_patience.value(
        )
        self.parameters.cfg["output_folder"] = self.browse_out_folder.path
        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build algorithm widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class TrainRfDetrWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the algorithm name attribute -> it must be the same as the one declared in the algorithm factory class
        self.name = "train_rf_detr"

    def create(self, param):
        # Create widget object
        return TrainRfDetrWidget(param, None)
