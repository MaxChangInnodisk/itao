
from PyQt5.QtCore import QThread, flush, pyqtSignal
import subprocess
import time, os, sys
from itao.environ import SetupEnv
from itao.utils.qt_logger import CustomLogger
from itao.qtasks.tools import parse_arguments

########################################################################
# !tao classification train -e $SPECS_DIR/classification_spec.cfg \
#                       -r $USER_EXPERIMENT_DIR/output \
#                       -k $KEY --gpus 2
########################################################################
# !tao classification train -e $SPECS_DIR/classification_spec.cfg \
#                        -r $USER_EXPERIMENT_DIR/output \
#                        -k $KEY --gpus 2 \
#                        --init_epoch N

class TrainCMD(QThread):

    trigger = pyqtSignal(object)
    info = pyqtSignal(str)

    def __init__(self, args:dict ):
        super(TrainCMD, self).__init__()
        
        # init
        self.logger = CustomLogger().get_logger('dev')
        self.env = SetupEnv()
        
        # variable  
        self.data = {'epoch':None, 'avg_loss':None, 'val_loss':None}

        # symbol
        self.symbol = '[INFO]'

        # parse arguments
        key_args = [ 'task', 'spec', 'output_dir', 'key', 'num_gpus' ]
        ret, new_args, error_args = parse_arguments(key_args=key_args, in_args=args)
        if not ret:
            self.logger.error('Train: Input arguments is wrong: {}'.format(error_args))
            sys.exit(1)

        self.cmd = [    
            "tao", f"{ new_args['task'] }", "train",
            "-e", f"{ new_args['spec'] }", 
            "-r", f"{ new_args['output_dir'] }",
            "-k", f"{ new_args['key'] }",
            "--gpus", f"{ new_args['num_gpus'] }"
        ]

        # check is in docker
        if args['is_docker'] and args['is_docker']==True:
            self.cmd.pop(0)
        
        self.logger.info('----------------')
        self.logger.info(self.cmd)
        self.logger.info('----------------')

    """ 檢查 epoch  """
    def check_epoch_in_line(self, line):
        if 'Epoch ' in line:
                line_cnt = line.split(' ')
                if len(line_cnt)==2:  
                    self.data['epoch'] = int(line_cnt[1].split('/')[0])
                    return 1
                else: 
                    return 0
        return 0
    
    def split_format(self, line, fmt):
        return (line.split(fmt)[1]).split('-')[0]

    """ 檢查是否有 loss """
    def check_loss_in_line_new(self, line):
        if 'val_loss: ' in line:
            self.data['avg_loss'] = round(float( self.split_format(line, 'loss: ') ), 3)
            self.data['val_loss'] = round(float( self.split_format(line, 'val_loss: ') ), 3)
            return 1
        else:
            return 0 

    def run(self):
        proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE)    
        
        while(True):

            if proc.poll() is not None: break

            for line in proc.stdout:
                
                line = line.decode('utf-8', 'ignore').rstrip('\n')
                
                if 'loss' in line and 'val_loss' not in line:
                    continue
                
                if self.check_epoch_in_line(line):
                    continue

                if not line.isspace(): self.logger.debug(line)
                
                if self.symbol in line:
                    self.trigger.emit({'INFO':f"{self.symbol} {line.split(self.symbol)[1]}"})

                if self.check_loss_in_line_new(line):
                    self.trigger.emit(self.data)
                    time.sleep(0.001)

        self.trigger.emit({})  
