from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        # Instantiate algorithm object
        from train_rf_detr.train_rf_detr_process import TrainRfDetrFactory
        return TrainRfDetrFactory()

    def get_widget_factory(self):
        # Instantiate associated widget object
        from train_rf_detr.train_rf_detr_widget import TrainRfDetrWidgetFactory
        return TrainRfDetrWidgetFactory()
