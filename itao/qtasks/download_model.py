from PyQt5.QtCore import QThread, flush, pyqtSignal
import subprocess
import os, shutil, time, glob
from itao import environ


class DownloadModel(QThread):

    trigger = pyqtSignal(str)

    def __init__(self, model='resnet', nlayer='18'):
        super(DownloadModel, self).__init__()
        self.flag = True  
        self.env = environ.SetupEnv()
        self.NGCCLI_URL="https://ngc.nvidia.com/downloads"
        self.CLI = "ngccli_cat_linux.zip"
        self.NGCCLI = os.path.join(self.NGCCLI_URL, self.CLI)
        self.NGCCLI_DIR = os.path.join(self.env.get_env("LOCAL_PROJECT_DIR"), 'ngccli')
        self.NGCCLI_FILE = os.path.join(self.NGCCLI_DIR, self.CLI)

        self.ARCH = model.lower()
        if nlayer.lower() != 'default':
            self.ARCH = self.ARCH + ( nlayer if nlayer.isdigit() else f"_{nlayer}" )
        
        self.ARCH_DIR = os.path.join(self.env.get_env("LOCAL_EXPERIMENT_DIR"), f'pretrained_{self.ARCH}')
    
    def get_model_path(self):
        hdf5_list = glob.glob( f"{os.path.join(self.ARCH_DIR, '**/*.hdf5')}", recursive=True)
        return hdf5_list[0] if len(hdf5_list)>0 else None
        
    def run(self):
        
        if self.get_model_path() is None:
            if os.path.exists(self.ARCH_DIR):
                shutil.rmtree(self.ARCH_DIR)
            # 建立環境變數並建立模型資料夾
            os.environ["PATH"]="{}/ngccli:{}".format(self.env.get_env("LOCAL_PROJECT_DIR"), os.getenv("PATH", ""))
            os.makedirs( self.ARCH_DIR )

            # 下載模型
            cmd = "{}/ngc registry model download-version nvidia/tao/pretrained_{}:{} --dest {}".format(
                self.NGCCLI_DIR,
                self.env.get_env('NGC_TASK'),
                self.ARCH,
                self.ARCH_DIR
            )

            proc = subprocess.Popen(cmd, shell=True, encoding="utf-8", stdout=subprocess.PIPE)
            while(self.flag):
                if proc.poll() is not None:
                    self.flag = False
                    break
                for line in proc.stdout:
                    self.trigger.emit(line)
            
            if len(os.listdir(self.ARCH_DIR))>0:
                print('\n\nGet Pretrained Model', len(os.listdir(self.ARCH_DIR)))
                self.trigger.emit(f"[END]:{self.get_model_path()}")
            else:
                self.trigger.emit(f"[ERROR]:Please download again.")
        else:

            self.trigger.emit(f"[EXIST]:{self.get_model_path()}")