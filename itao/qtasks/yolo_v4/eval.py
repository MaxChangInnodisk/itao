from PyQt5.QtCore import QThread, pyqtSignal
import subprocess
import sys
from itao.environ import SetupEnv
from itao.utils.qt_logger import CustomLogger
from itao.qtasks.tools import parse_arguments

########################################################################
#tao classification evaluate -e $SPECS_DIR/classification_spec.cfg -k $KEY
########################################################################

class EvalCMD(QThread):

    trigger = pyqtSignal(str)

    def __init__(self,  args):
        super(EvalCMD, self).__init__()
        
        # basic
        self.env = SetupEnv() 
        self.logger = CustomLogger().get_logger('dev')

        # variable
        self.symbol='[INFO]'
        self.record = False

        # symbol
        self.data = {'epoch':None, 'avg_loss':None, 'val_loss':None}
        self.record_symbol = 'Start to calculate AP for each class'

        # parsing argument
        key_args = [ 'task', 'spec', 'key', 'model' ]
        ret, new_args, error_args = parse_arguments(key_args=key_args, in_args=args)

        if not ret:
            self.logger.error('Eval: Input arguments is wrong: {}'.format(error_args))
            sys.exit(1)
        
        # combine command line
        self.cmd = [    
            "tao", f"{ new_args['task'] }", "evaluate",
            "-e", f"{ new_args['spec'] }",
            "-k", f"{ new_args['key'] }",
            "-m", f"{ new_args['model'] }"
        ]
        # check is in docker
        if args['is_docker'] and args['is_docker']==True:
            self.cmd.pop(0)
            
        # add gpus if needed
        if 'num_gpus' in args.keys():
            self.cmd.append("--gpus")
            self.cmd.append(f"{ args['num_gpus'] }")

        # add gpu_index if needed
        if 'gpu_index' in args.keys():
            self.cmd.append("--gpu_index")
            self.cmd.append(f"{args['gpu_index']}")

        # show result
        self.logger.info('----------------')
        self.logger.info(self.cmd)
        self.logger.info('----------------')

    """ 檢查是否有需要回傳的訊息 """
    def check_info_in_line(self, line:str) -> None:
        if self.symbol in line:
            self.trigger.emit( f"{self.symbol} {line.split(self.symbol)[1]}")
    
    """ 判斷是否要保持回傳並回傳資料 """
    def check_to_record_line(self, line:str) -> None:
        
        if self.record_symbol in line:
            self.record = True
        
        if self.record:
            self.trigger.emit(line)

    """ 主要執行的地方 """
    def run(self):
        proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE)
        while(True):

            if proc.poll() is not None: break

            for line in proc.stdout:
                
                line = line.decode('utf-8', 'ignore').rstrip('\n').replace('\x08', '')
                
                if line.isspace(): 
                    continue
                else:
                    self.logger.debug(line)

                self.check_info_in_line(line)
                self.check_to_record_line(line)

        self.trigger.emit("end")