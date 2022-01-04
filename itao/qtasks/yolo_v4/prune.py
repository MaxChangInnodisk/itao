
from glob import glob
from PyQt5.QtCore import QThread, flush, pyqtSignal
import subprocess
import time, os, glob, sys
from itao.environ import SetupEnv
from itao.utils.qt_logger import CustomLogger
from itao.qtasks.tools import parse_arguments

########################################################################

# %env EPOCH=080
# !mkdir -p $LOCAL_EXPERIMENT_DIR/output/resnet_pruned
# !tao yolo_v4 prune -m $USER_EXPERIMENT_DIR/experiment_dir_unpruned/weights/yolov4_resnet18_epoch_$EPOCH.tlt \
#                    -e $SPECS_DIR/yolo_v4_train_resnet18_kitti_seq.txt \
#                    -o $USER_EXPERIMENT_DIR/experiment_dir_pruned/yolov4_resnet18_e${EPOCH}_pruned.tlt \
#                    -eq intersection \
#                    -pth 0.1 \
#                    -k $KEY

class PruneCMD(QThread):

    trigger = pyqtSignal(str)

    def __init__(self, args:dict ):
        super(PruneCMD, self).__init__()
        self.env = SetupEnv()    
        self.logger = CustomLogger().get_logger('dev')

        # parse arguments
        key_args = [ 'task', 'input_model', 'spec', 'output_model', 'pth', 'eq', 'key']
        ret, new_args, error_args = parse_arguments(key_args=key_args, in_args=args)
        if not ret:
            self.logger.error('Prune: Input arguments is wrong: {}'.format(error_args))
            sys.exit(1)
        
        # define commmand line
        self.cmd = [    
            "tao", f"{ new_args['task'] }", "prune",
            "-m", f"{ new_args['input_model'] }",
            "-e", f"{ new_args['spec'] }",
            "-o", f"{ new_args['output_model'] }",
            "-eq", f"{ new_args['eq'] }",
            "-pth", f"{ new_args['pth'] }",
            "-k", f"{ new_args['key'] }",
        ]

        
        self.logger.info('----------------')
        self.logger.info(self.cmd)
        self.logger.info('----------------')

    def run(self):
        proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE)
        while(True):
            if proc.poll() is not None:
                break
            else:
                for line in proc.stdout:                
                    line = line.decode("utf-8").rstrip('\n').replace('\x08', '')
                    if bool(line.rstrip()): self.logger.debug(line)
                    
                    if 'WARNING' not in line:
                        if 'INFO' in line:
                            self.trigger.emit(line.split('[INFO]')[1])
                        else:
                            self.trigger.emit(line)

        self.trigger.emit("end")
        