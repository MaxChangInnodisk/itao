from PyQt5.QtCore import QThread, pyqtSignal
import subprocess
from itao.utils.qt_logger import CustomLogger

#########################################################################################################

# !tao classification export \
#             -m $USER_EXPERIMENT_DIR/output_retrain/weights/resnet_$EPOCH.tlt \
#             -o $USER_EXPERIMENT_DIR/export/final_model.etlt \
#             -k $KEY

class ExportCMD(QThread):

    trigger = pyqtSignal(str)

    def __init__(self, task, key, retrain_model, output_model):
        super(ExportCMD, self).__init__()

        self.flag = True        
        self.data = {}
        self.trg = False
        self.cur_name = ""

        self.cmd = [    
            "tao", f"{ task }", "export",
            "-m", f"{ retrain_model}",
            "-k", f"{ key }",
            "-o", f"{ output_model }"
        ]

        self.logger = CustomLogger().get_logger('dev')
        self.logger.info('----------------')
        self.logger.info(self.cmd)
        self.logger.info('----------------')

    def run(self):
        proc = subprocess.Popen(self.cmd , stdout=subprocess.PIPE)
        while(self.flag):
            if proc.poll() is not None:
                self.flag = False
                break
            for line in proc.stdout:
                line = line.decode("utf-8", "ignore").rstrip('\n').strip()
                
                if line.rstrip(): self.logger.debug(line)

                if "[INFO]" in line:
                    self.trigger.emit(line)

        self.trigger.emit("end")