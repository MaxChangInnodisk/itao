from glob import glob
from PyQt5.QtCore import QThread, flush, pyqtSignal
import subprocess
import time, os, glob, sys
from itao.environ import SetupEnv
from itao.utils.qt_logger import CustomLogger
from itao.qtasks.tools import parse_arguments

########################################################################

#tao classification evaluate -e $SPECS_DIR/classification_spec.cfg -k $KEY

class EvalCMD(QThread):

    trigger = pyqtSignal(str)

    def __init__(self, args:dict ):
        super(EvalCMD, self).__init__()
        
        # init
        self.env = SetupEnv() 
        self.logger = CustomLogger().get_logger('dev')

        # variable
        self.data = {'epoch':None, 'avg_loss':None, 'val_loss':None}

        # symbol
        self.symbol = '[INFO]'
        self.record_symbol = 'Evaluation Loss'
        self.record = False

        # parsing arguments
        key_args = [ 'task', 'spec', 'key' ]
        ret, new_args, error_args = parse_arguments(key_args=key_args, in_args=args)

        if not ret:
            self.logger.error('Eval: Input arguments is wrong: {}'.format(error_args))
            sys.exit(1)
        
        self.cmd = [    
            "tao", f"{ new_args['task'] }", "evaluate",
            "-e", f"{ new_args['spec'] }",
            "-k", f"{ new_args['key'] }"
        ]
        # check is in docker
        if args['is_docker'] and args['is_docker']==True:
            self.cmd.pop(0)

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

    def run(self):
        proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE)
        while(True):
            
            if proc.poll() is not None: break

            for line in proc.stdout:
                
                line = line.decode("utf-8", 'ignore').rstrip('\n').replace('\x08', '')
                
                if line.isspace():
                    continue
                else:
                    self.logger.debug(line)

                self.check_info_in_line(line)
                self.check_to_record_line(line)

        self.trigger.emit("end")