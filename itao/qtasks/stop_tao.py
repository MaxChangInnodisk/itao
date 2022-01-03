from glob import glob
from PyQt5.QtCore import QThread, flush, pyqtSignal
import subprocess

class StopTAO(QThread):

    trigger = pyqtSignal(str)

    def __init__(self ):
        super(StopTAO, self).__init__()
        pass
        
    def run(self):
        proc = subprocess.Popen(["tao", "stop", "--all"], stdout=subprocess.PIPE, encoding='utf-8')
        while(True):
            if proc.poll() is not None:
                break
            for line in proc.stdout:
                self.trigger.emit(line)
        self.trigger.emit("end")