from PyQt5.QtCore import QThread, pyqtSignal
import subprocess
import sys
from itao.environ import SetupEnv
from itao.utils.qt_logger import CustomLogger
from itao.qtasks.tools import parse_arguments

########################################################################
# !tao classification train -e $SPECS_DIR/classification_spec.cfg \
#                       -r $USER_EXPERIMENT_DIR/output \
#                       -k $KEY --gpus 2
########################################################################
# !tao yolo_v4 train -e $SPECS_DIR/yolo_v4_train_resnet18_kitti.txt \
#                    -r $USER_EXPERIMENT_DIR/experiment_dir_unpruned \
#                    -k $KEY \
#                    --gpus 1
########################################################################

class TrainCMD(QThread):

    trigger = pyqtSignal(object)
    info = pyqtSignal(str)

    def __init__(self, args:dict ):
        super(TrainCMD, self).__init__()
        
        # basic
        self.logger = CustomLogger().get_logger('dev')  # get logger
        self.env = SetupEnv()   # get environ

        # some common variable
        self.flag = True        
        self.data = {'epoch':None, 'avg_loss':None, 'val_loss':None}    # setup return data format
        self.is_valid = False

        # some symbol for check and return
        self.symbol = 'INFO'    
        self.epoch_symbol = 'Epoch '
        self.loss_symbol = 'ms/step - loss: '   # for average loss
        self.val_start = 'Start to calculate AP for each class' # for validate
        self.val_end = 'Validation loss'
        
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
        # check is in docker
        if args['is_docker'] and args['is_docker']==True:
            self.cmd.pop(0)

        # add gpu index if need 
        if 'gpu_index' in args.keys():
            self.logger.warning('Using GPU (index:{})'.format(args['gpu_index']))
            self.cmd.append("--gpu_index")
            self.cmd.append(f"{args['gpu_index']}")

        # show command line
        self.logger.info('----------------')
        self.logger.info(self.cmd)
        self.logger.info('----------------')

    """ 檢查是否有 epoch 在其中並回傳 True or False """
    def check_epoch_in_line(self, line:str) -> bool:
        if self.epoch_symbol in line:
            line_cnt = line.split(' ')
            if len(line_cnt)==2:  
                self.data['epoch'] = int(line_cnt[1].split('/')[0])
                
    """ 檢查是否正在進行驗證 """
    def check_validate_in_line(self ,line:str) -> None:

        if self.val_start in line: 
            self.is_valid = True
        
        if self.is_valid: 
            self.trigger.emit({'INFO':f"{line}"})

        if self.val_end in line:
            self.is_valid = False

    """ 檢查是否有 Loss 在其中 """
    def check_loss_in_line(self, line:str):
        # 如果有 loss 的相關特徵在內容的話
        if self.loss_symbol in line:
            cap_loss = line.split(self.loss_symbol)[1]
            # 如果是數字的話
            if (cap_loss.replace('.','').rstrip()).isdigit():   
                self.data['avg_loss']=round(float(cap_loss.rstrip()), 3)
                self.trigger.emit(self.data)

    def run(self):
        # 建立子行程
        proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE)    
        
        while(True):

            if proc.poll() is not None:
                break

            for line in proc.stdout:
                
                line = line.decode('utf-8', 'ignore').rstrip('\n').replace('\x08', '')
                
                if line.isspace(): 
                    continue
                else:
                    self.logger.debug(line)

                self.check_epoch_in_line(line)
                self.check_validate_in_line(line)
                self.check_loss_in_line(line)

        self.trigger.emit({})  
