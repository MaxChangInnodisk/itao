
from glob import glob
from PyQt5.QtCore import QThread, flush, pyqtSignal
import subprocess
import time, os, glob, sys
from itao.environ import SetupEnv
from itao.utils.qt_logger import CustomLogger
from itao.qtasks.tools import parse_arguments

########################################################################

# !tao classification train -e $SPECS_DIR/classification_retrain_spec.cfg \
#                       -r $USER_EXPERIMENT_DIR/output_retrain \
#                       -k $KEY

class ReTrainCMD(QThread):

    trigger = pyqtSignal(object)

    def __init__(self, args:dict):
        super(ReTrainCMD, self).__init__()
        
        self.env = SetupEnv()
        self.flag = True        
        self.data = {'epoch':None, 'avg_loss':None, 'val_loss':None}
        self.logger = CustomLogger().get_logger('dev')

        key_args = ['task', 'spec', 'output_dir', 'key', 'num_gpus']
        ret, new_args, error_args = parse_arguments(key_args=key_args, in_args=args)
        if not ret:
            self.logger.error('Prune: Input arguments is wrong: {}'.format(error_args))
            sys.exit(1)

        self.cmd = [    
            # "tao", f"{ self.env.get_env('TASK') }", "train",
            "tao", f"{ new_args['task'] }", "train",
            "-e", f"{ new_args['spec'] }", 
            "-r", f"{ new_args['output_dir'] }" ,
            "-k", f"{ new_args['key'] }",
            "--gpus", f"{ new_args['num_gpus'] }"
        ]

        self.logger.info('----------------')
        self.logger.info(self.cmd)
        self.logger.info('----------------')

        self.first_epoch = True
        self.val = None
        self.old_val = None
        self.temp = {}
        self.symbol = '[INFO]'

    def check_epoch_in_line(self, line):
        if 'Epoch ' in line:
                line_cnt = line.split(' ')
                if len(line_cnt)==2:  
                    self.data['epoch'] = int(line_cnt[1].split('/')[0])
                    return 1
                else: 
                    return 0
        return 0
    
    def check_loss_in_line(self, line):
        if 'loss: ' in line:
            self.val = round(float( (line.split('loss: ')[1]).split(' -')[0] ), 3)
            if self.val != self.old_val:
                if 'Validation' in line:            
                    self.data['val_loss']=  self.val 
                    self.old_val = self.val                 
                    return 1
                self.data['avg_loss']= self.val
                self.old_val = self.val
            else:
                self.old_val = self.val
                return 0
            return 1
        else:
            return 0

    def split_format(self, line, fmt):
        return (line.split(fmt)[1]).split('-')[0]

    def check_loss_in_line_new(self, line):
        if 'val_loss: ' in line:
            # self.val = round(float( self.split_format(line, 'loss: ') ), 3)
            self.data['avg_loss'] = round(float( self.split_format(line, 'loss: ') ), 3)
            self.data['val_loss'] = round(float( self.split_format(line, 'val_loss: ') ), 3)
            return 1
        else:
            return 0 

    def run(self):
        proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE)    
        
        while(self.flag):

            if proc.poll() is not None:
                self.flag = False
                break

            for line in proc.stdout:
                
                line = line.decode('utf-8', 'ignore').rstrip('\n')
                if line.rstrip(): self.logger.debug(line)

                if self.check_epoch_in_line(line):
                    continue
                
                if self.symbol in line:
                    self.trigger.emit({'INFO':f"{self.symbol} {line.split(self.symbol)[1]}"})

                if self.check_loss_in_line_new(line):
                    self.trigger.emit(self.data)
                    time.sleep(0.001)

        self.trigger.emit({})  
