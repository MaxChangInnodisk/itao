
from PyQt5.QtCore import QThread, pyqtSignal
import subprocess
import sys
from itao.environ import SetupEnv
from itao.utils.qt_logger import CustomLogger
from itao.qtasks.tools import parse_arguments

#########################################################################################################
# !tao classification export \
#             -m $USER_EXPERIMENT_DIR/output_retrain/weights/resnet_$EPOCH.tlt \
#             -o $USER_EXPERIMENT_DIR/export/final_model.etlt \
#             -k $KEY
#########################################################################################################
# !tao yolo_v4 export -m $USER_EXPERIMENT_DIR/experiment_dir_retrain/weights/yolov4_resnet18_epoch_$EPOCH.tlt  \
#                     -o $USER_EXPERIMENT_DIR/export/yolov4_resnet18_epoch_$EPOCH.etlt \
#                     -e $SPECS_DIR/yolo_v4_retrain_resnet18_kitti.txt \
#                     -k $KEY \
#                     --data_type int8 \
#                     --cal_cache_file $USER_EXPERIMENT_DIR/export/cal.bin

class ExportCMD(QThread):

    trigger = pyqtSignal(str)

    def __init__(self, args:dict):
        super(ExportCMD, self).__init__()

        self.logger = CustomLogger().get_logger('dev')
        self.flag = True        
        self.data = {}
        self.trg = False
        self.cur_name = ""

        # parse arguments
        key_args = [ 'task', 'key', 'spec', 'intput_model', 'output_model', 'dtype' ]
        ret, new_args, error_args = parse_arguments(key_args=key_args, in_args=args)
        if not ret:
            self.logger.error('Train: Input arguments is wrong: {}'.format(error_args))
            sys.exit(1)

        self.cmd = [    
            "tao", f"{ new_args['task'] }", "export",
            "-k", f"{ new_args['key'] }",
            "-m", f"{ new_args['intput_model'] }",
            "-o", f"{ new_args['output_model'] }",
            "-e", f"{ new_args['spec'] }",
            "--data_type", f"{ new_args['dtype'].lower() }"
        ]

        # check is in docker
        if args['is_docker'] and args['is_docker']==True:
            self.cmd.pop(0)
            
        self.logger.info('----------------')
        self.logger.info(self.cmd)
        self.logger.info('----------------')

    def run(self):
        proc = subprocess.Popen(self.cmd , stdout=subprocess.PIPE)
        while(self.flag):
            if proc.poll() is not None:
                self.flag = False
                break
            for line in proc.stdout:
                line = line.decode('utf-8', 'ignore').rstrip('\n').replace('\x08', '')
                
                if not line.isspace(): self.logger.debug(line)

                if "[INFO]" in line:
                    ret = line.split('[INFO]')[1]
                    self.trigger.emit(ret)

        self.trigger.emit("end")