
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

# print("To run with multigpu, please change --gpus based on the number of available GPUs in your machine.")
# !tao yolo_v4 train -e $SPECS_DIR/yolo_v4_train_resnet18_kitti.txt \
#                    -r $USER_EXPERIMENT_DIR/experiment_dir_unpruned \
#                    -k $KEY \
#                    --gpus 1

class TrainCMD(QThread):

    trigger = pyqtSignal(object)
    info = pyqtSignal(str)

    def __init__(self, args:dict ):
        super(TrainCMD, self).__init__()
        
        self.logger = CustomLogger().get_logger('dev')
        self.env = SetupEnv()
        self.flag = True        
        self.data = {'epoch':None, 'avg_loss':None, 'val_loss':None}

        # parse arguments
        key_args = [ 'task', 'spec', 'output_dir', 'key', 'num_gpus' ]
        ret, new_args, error_args = parse_arguments(key_args=key_args, in_args=args)
        if not ret:
            self.logger.error('Train: Input arguments is wrong: {}'.format(error_args))
            sys.exit(1)
        
        # define commmand line
        self.cmd = [    
            "tao", f"{ new_args['task'] }", "train",
            "-e", f"{ new_args['spec'] }", 
            "-r", f"{ new_args['output_dir'] }",
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
        self.symbol = 'INFO'

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
        return (line.split(fmt)[1])

    def check_loss_in_line_new(self, line):
        if 'ms/step - loss: ' in line:# and ( 'ETA' in line or 'step' in line):
            res = line.split('ms/step - loss: ')[1]
            self.data['avg_loss'] = round(float(res), 3)
        return 0

    def run(self):
        proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE)    
        
        while(self.flag):

            if proc.poll() is not None:
                self.flag = False
                break

            for line in proc.stdout:
                
                # line = line.decode('utf-8', 'ignore').rstrip('\n').rstrip('\r').replace('\x08', '')
                line = line.decode('utf-8', 'ignore').rstrip('\n').replace('\x08', '')
                
                self.logger.debug(line)

                # check epoch and put current epoch into self.data['epoch']
                self.check_epoch_in_line(line)
                    
                # check loss -> only get the last time with symbol ('ms/step - loss: ')
                symbol = 'ms/step - loss: '
                if ('INFO' in line or symbol in line) and line.rstrip():
                    

                    # get loss and send data
                    if symbol in line:
                        cap_loss = line.split(symbol)[1]
                        
                        if (cap_loss.replace('.', '').rstrip()).isdigit():
                            # print('\n\nIts digit!!!!!! Send data ... ', flush=True)
                            self.data['avg_loss'] = round( float(cap_loss.rstrip()), 3)
                            self.trigger.emit(self.data)

        self.trigger.emit({})  
