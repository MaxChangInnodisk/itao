
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

    def __init__(self,  args):
        super(EvalCMD, self).__init__()
        
        self.env = SetupEnv() 
        self.data = {'epoch':None, 'avg_loss':None, 'val_loss':None}

        key_args = [ 'task', 'spec', 'key', 'model' ]
        ret, new_args, error_args = parse_arguments(key_args=key_args, in_args=args)

        if not ret:
            self.logger.error('Eval: Input arguments is wrong: {}'.format(error_args))
            sys.exit(1)
        
        self.cmd = [    
            "tao", f"{ new_args['task'] }", "evaluate",
            "-e", f"{ new_args['spec'] }",
            "-k", f"{ new_args['key'] }",
            "-m", f"{ new_args['model'] }"
        ]

        self.logger = CustomLogger().get_logger('dev')
        self.logger.info('----------------')
        self.logger.info(self.cmd)
        self.logger.info('----------------')

        self.symbols=['[INFO]', 'Start to calculate AP for each class']
        self.record = False

    def run(self):
        proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE)
        while(True):
            if proc.poll() is not None:
                break
            for line in proc.stdout:
                
                line = line.decode('utf-8', 'ignore').rstrip('\n')
                
                if line.rstrip(): self.logger.debug(line)

                if 'WARNING' in line or line.isspace(): 
                    continue

                for symbol in self.symbols:
                    if symbol in line:
                        if symbol=='[INFO]':
                            self.trigger.emit( f"{symbol} {line.split(symbol)[1]}" )
                        else:
                            self.record = True
                if self.record:
                    self.trigger.emit(line)

        self.trigger.emit("end")