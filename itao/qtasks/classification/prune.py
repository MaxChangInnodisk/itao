
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
# !tao classification prune -m $USER_EXPERIMENT_DIR/output/weights/resnet_$EPOCH.tlt \
#            -o $USER_EXPERIMENT_DIR/output/resnet_pruned/resnet18_nopool_bn_pruned.tlt \
#            -eq union \
#            -pth 0.6 \
#            -k $KEY

class PruneCMD(QThread):

    trigger = pyqtSignal(str)

    def __init__(self, args:dict ):
        super(PruneCMD, self).__init__()
        self.env = SetupEnv()    

        # parse arguments
        key_args = [ 'task', 'input_model', 'output_model', 'pth', 'eq', 'key']
        ret, new_args, error_args = parse_arguments(key_args=key_args, in_args=args)
        if not ret:
            self.logger.error('Prune: Input arguments is wrong: {}'.format(error_args))
            sys.exit(1)
        
        # define commmand line
        self.cmd = [    
            "tao", f"{ new_args['task'] }", "prune",
            "-m", f"{ new_args['input_model']  }",
            "-o", f"{ new_args['output_model']  }",
            "-eq", f"{ new_args['eq'] }",
            "-pth", f"{ new_args['pth'] }",
            "-k", f"{ new_args['key'] }",
        ]

        # check is in docker
        if args['is_docker'] and args['is_docker']==True:
            self.cmd.pop(0)
            

        self.logger = CustomLogger().get_logger('dev')
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
                    
                    line = line.decode("utf-8", 'ignore').rstrip('\n').replace('\x08', '')
                    
                    if line.isspace() or 'WARNING' in line:
                        continue
                    else:
                        self.logger.debug(line)
                    
                    if 'INFO' in line:
                        self.trigger.emit(line.split('[INFO]')[1])
                    else:
                        self.trigger.emit(line)

        self.trigger.emit("end")
        