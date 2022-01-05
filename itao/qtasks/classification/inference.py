from PyQt5.QtCore import QThread, pyqtSignal
import subprocess
import sys, os
from itao.environ import SetupEnv
from itao.utils.qt_logger import CustomLogger
from itao.qtasks.tools import parse_arguments

#########################################################################################################

# !tao classification inference -e $SPECS_DIR/classification_retrain_spec.cfg \
#                           -m $USER_EXPERIMENT_DIR/output_retrain/weights/resnet_$EPOCH.tlt \
#                           -k $KEY -b 32 -d $DATA_DOWNLOAD_DIR/split/test/person \
#                           -cm $USER_EXPERIMENT_DIR/output_retrain/classmap.json

class InferCMD(QThread):

    trigger = pyqtSignal(dict)
    info = pyqtSignal(str)

    def __init__(self, args:dict ):
        super(InferCMD, self).__init__()

        self.env = SetupEnv()
        self.logger = CustomLogger().get_logger('dev')

        self.flag = True        
        self.data = {}
        self.trg = False
        self.cur_name = ""

        # parsing argument
        key_args = [ 'task', 'spec', 'key', 'model', 'input_dir', 'batch_size', 'class_map' ]
        ret, new_args, error_args = parse_arguments(key_args=key_args, in_args=args)

        if not ret:
            self.logger.error('Eval: Input arguments is wrong: {}'.format(error_args))
            sys.exit(1)
        
        # combine command line
        self.cmd = [    
            "tao", f"{ new_args['task'] }", "inference",
            "-e", f"{ new_args['spec'] }",
            "-k", f"{ new_args['key'] }",
            "-m", f"{ new_args['model'] }",
            "-d", f"{ new_args['input_dir'] }",
            "-b", f"{ new_args['batch_size'] }",
            "-cm", f"{ new_args['class_map'] }"
        ]
        
        # add gpus if needed
        if 'num_gpus' in args.keys():
            self.cmd.append("--gpus")
            self.cmd.append(f"{ args['num_gpus'] }")

        # add gpu_index if needed
        if 'gpu_index' in args.keys():
            self.cmd.append("--gpu_index")
            self.cmd.append(f"{args['gpu_index']}")

        self.logger.info('----------------')
        self.logger.info(self.cmd)
        self.logger.info('----------------')

    def run(self):
        proc = subprocess.Popen(self.cmd , stdout=subprocess.PIPE)
        
        while(True):
        
            if proc.poll() is not None: break
        
            for line in proc.stdout:

                line = line.decode('utf-8', 'ignore').rstrip('\n').replace('\x08', '')
                
                if not line.isspace(): self.logger.debug(line)

                if "[INFO]" in line:
                    self.info.emit(line.split('[INFO]')[1])

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