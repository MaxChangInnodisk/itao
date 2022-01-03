from PyQt5.QtCore import QThread, pyqtSignal
import subprocess
from itao.environ import SetupEnv
from itao.utils.qt_logger import CustomLogger

#########################################################################################################

# !tao classification inference -e $SPECS_DIR/classification_retrain_spec.cfg \
#                           -m $USER_EXPERIMENT_DIR/output_retrain/weights/resnet_$EPOCH.tlt \
#                           -k $KEY -b 32 -d $DATA_DOWNLOAD_DIR/split/test/person \
#                           -cm $USER_EXPERIMENT_DIR/output_retrain/classmap.json

class InferCMD(QThread):

    trigger = pyqtSignal(dict)
    info = pyqtSignal(str)

    def __init__(self, task, key, spec, retrain_model, batch_size, data, classmap):
        super(InferCMD, self).__init__()
        self.env = SetupEnv()
        self.flag = True        
        self.data = {}
        self.trg = False
        self.cur_name = ""

        self.cmd = [    
            "tao", f"{ task }", "inference",
            "-e", f"{ spec }",
            "-m", f"{ retrain_model}",
            "-k", f"{ key }",
            "-b", f"{ batch_size }",
            "-d", f"{ data }",
            "-cm", f"{ classmap }"
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
                    self.info.emit(line)

                if ":{" in line:
                    self.cur_name = line.replace('"','').replace(':','').replace('{','')
                    self.data[self.cur_name] = []
                    self.trg = True
                elif "}" in line:
                    self.trg = False
                    self.trigger.emit(self.data)
                else:
                    if self.trg:
                        self.data[self.cur_name].append(line.rstrip(" ").rstrip(","))
        self.trigger.emit({})