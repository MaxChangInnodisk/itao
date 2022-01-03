from PyQt5.QtCore import QThread, flush, pyqtSignal
import subprocess
import os, shutil, time
from itao import environ

class InstallNGC(QThread):

    trigger = pyqtSignal(str)

    def __init__(self):
        super(InstallNGC, self).__init__()
        self.flag = True  
        self.env = environ.SetupEnv()
        self.NGCCLI_URL="https://ngc.nvidia.com/downloads"
        self.CLI = "ngccli_cat_linux.zip"
        self.NGCCLI = os.path.join(self.NGCCLI_URL, self.CLI)
        self.NGCCLI_DIR = os.path.join(self.env.get_env("LOCAL_PROJECT_DIR"), 'ngccli')
        self.NGCCLI_FILE = os.path.join(self.NGCCLI_DIR, self.CLI)
    
    def check_stats(self):
        if os.path.exists(self.NGCCLI_DIR):
            if 'ngc' in os.listdir(self.NGCCLI_DIR):
                return True
            else:   
                shutil.rmtree(self.NGCCLI_DIR)
        os.makedirs( self.NGCCLI_DIR )
        return False

    def run(self):
        if self.check_stats():
            self.trigger.emit("exist")
            self.flag = False
        else:
            cmd = "wget {} -P {} && unzip -u {} -d {} && rm -rf {}/*.zip".format(
                self.NGCCLI, self.NGCCLI_DIR,
                self.NGCCLI_FILE, self.NGCCLI_DIR, self.NGCCLI_DIR
            )           
            proc = subprocess.Popen(cmd, shell=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            while(self.flag):
                if proc.poll() is not None:
                    self.flag = False
                    break
                # out, err = proc.communicate()
                # for e in err.split(' '):
                #     if "%" in e:
                #         self.trigger.emit(e)
                # self.trigger.emit(err)
                # for line in proc.stdout:
                #     # line = line.rstrip('\n')
                #     self.trigger.emit(line)

            self.trigger.emit("end")

    def stop(self):
        self.flag=False