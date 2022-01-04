from PyQt5 import QtWidgets
from PyQt5 import QtGui, uic
from PyQt5.QtWidgets import QFileDialog, QVBoxLayout
import datetime
import sys, os
from typing import *
import pyqtgraph as pg
import subprocess

""" 自己的函式庫自己撈"""
sys.path.append("../itao")
from itao.environ import SetupEnv
from itao.utils.spec_tools_v2 import DefineSpec
from itao.qtasks.install_ngc import InstallNGC
from itao.qtasks.stop_tao import StopTAO
from itao.utils.qt_logger import CustomLogger

from demo.configs import OPT, ARCH_LAYER, TRAIN_CONF, RETRAIN_CONF, PRUNE_CONF, INFER_CONF, EXPORT_CONF

class Init(QtWidgets.QMainWindow):

    def __init__(self) -> None:
        super().__init__() # Call the inherited classes __init__ method

        # 取得共用的 logger
        self.logger = CustomLogger().get_logger('dev')

        # iTAO 初始化
        self.logger.info('Initial iTAO ... ')
        self.ui = uic.loadUi(os.path.join("ui", "itao_v0.2.ui"), self) # Load the  file   # 使用 ui 檔案的方式
        # from demo.pyqt_gui import Ui_iTAO
        # self.ui = Ui_iTAO() # 使用 pyuic 轉換成 python 後的使用方法
        # self.ui.setupUi(self)
        self.setWindowTitle('iTAO')

        # 基本常數設定
        self.logger.info('Setting Basic Variable ... ')
        self.first_page_id = 0  # 1-1 = 0
        self.end_page_id = 3    # 4-1 = 3
        self.div_symbol = "----------------------------------------------------\n"

        self.option, self.option_nlayer = OPT, ARCH_LAYER
        self.train_conf, self.retrain_conf, self.prune_conf = TRAIN_CONF, RETRAIN_CONF, PRUNE_CONF
        self.infer_conf, self.export_conf = INFER_CONF, EXPORT_CONF

        if len(sys.argv)>=2 and sys.argv[1].lower()=='debug':
            self.debug = True  
            self.debug_page = int(sys.argv[2]) if sys.argv[2] is not None else 0
        else:
            self.debug = False 
            
        if self.debug: 
            self.logger.warning('Debug Mode')

        self.t1_first_time, self.t2_firt_time, self.t3_first_time, self.t4_first_time = True, True, True, True   # 初次進入頁面
        self.first_line = True  # 第一行
        

        """ 環境相關 """
        self.train_cmd, self.eval_cmd, self.retrain_cmd, self.prune_cmd, self.infer_cmd, self.export_cmd = None, None, None, None, None, None
        self.kmeans_cmd = None
        self.train_spec, self.retrain_spec = None, None
        self.itao_env = SetupEnv()  # 建立 configs/itao_env.json 檔案，目的在於建立共用的變數以及 Docker 與 Local 之間的路徑
        self.itao_env.create_env_file() 
        self.ngc = InstallNGC() # NGC 的安裝Thread
        self.stop_tao = StopTAO() # 統一關閉 TAO 的方法

        """ Console """
        self.t1_info = ""
        self.console_cnt = ""
        self.div_is_inserted = False
        self.space = len('learning_rate      ') # get longest width in console


        """ 將元件統一 """
        # font = QtGui.QFont("Arial", 12)   # 設定字體
        # self.setFont(font)
        self.page_buttons_status={0:[0,0], 1:[1,0], 2:[1,0], 3:[1,1]}
        self.tabs = [ self.ui.tab_1, self.ui.tab_2, self.ui.tab_3, self.ui.tab_4 ]
        self.progress = [ self.ui.t1_progress, self.ui.t2_progress, self.ui.t3_progress, self.ui.t4_progress]
        self.frames = [None, self.ui.t2_frame, self.ui.t3_frame, None]
        self.consoles = [ self.ui.t1_console, self.ui.t2_console, self.ui.t3_console, self.ui.t4_console]

        [ self.ui.main_tab.setTabEnabled(i, False if not self.debug else True ) for i in range(len(self.tabs))]     # 將所有分頁都關閉
        

        """ 建立 & 初始化 Tab2 跟 Tab3 的圖表 """

        pg.setConfigOptions(antialias=True)
        pg.setConfigOption('background', 'w')
        self.pws = [None, pg.PlotWidget(self), pg.PlotWidget(self), None]
        self.pw_lyrs = [None, QVBoxLayout(), QVBoxLayout(), None]
        [ a.hide() for a in self.pws if a!=None ]    # 先關閉等待 init_console 的時候才開

        """ 設定 Previous、Next 的按鈕 """
        self.current_page_id = self.first_page_id   # 將當前頁面編號 (current_page_id) 設定為 第一個 ( first_page_id )
        
        self.ui.main_tab.setCurrentIndex(self.first_page_id)
        self.ui.bt_next.clicked.connect(self.ctrl_page_event)
        self.ui.bt_previous.clicked.connect(self.ctrl_page_event)
        
        self.update_page()  # 更新頁面資訊
        self.ui.main_tab.currentChanged.connect(self.update_page)

    """ 檢查 tao 的狀況 """
    def check_tao(self):
        proc = subprocess.run( ['tao','-h'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8", timeout=1)      
        return 1 if proc.returncode == 0 else 0

    """ 取得資料夾路徑 """
    def get_folder(self):
        folder_path = None
        if self.current_page_id==0:
            folder_path = QFileDialog.getExistingDirectory(self, "Open folder", "./tasks/data", options=QFileDialog.DontUseNativeDialog)
            trg_folder_path = self.itao_env.replace_docker_root(folder_path)
            
            if 'classification' == self.itao_env.get_env('NGC_TASK'):
                self.train_spec.mapping('train_dataset_path', '"{}"'.format(os.path.join(trg_folder_path, 'train')))
                self.train_spec.mapping('val_dataset_path', '"{}"'.format(os.path.join(trg_folder_path, 'val')))
                self.train_spec.mapping('eval_dataset_path', '"{}"'.format(os.path.join(trg_folder_path, 'test')))
            elif 'detection' in self.itao_env.get_env('NGC_TASK'):
                self.train_spec.mapping('image_directory_path', '"{}"'.format(os.path.join(trg_folder_path, 'images')))
                self.train_spec.mapping('label_directory_path', '"{}"'.format(os.path.join(trg_folder_path, 'labels')))
                
            # self.train_conf['dataset_path'] = folder_path
            self.itao_env.update('LOCAL_DATASET', folder_path)
            self.itao_env.update('DATASET', trg_folder_path)
            
            self.sel_idx[5]=1 
            self.update_progress(self.current_page_id, self.sel_idx.count(1), len(self.t1_objects))
        elif self.current_page_id==3:
            root = "./tasks/data"
            folder_path = QFileDialog.getExistingDirectory(self, "Open folder", root, options=QFileDialog.DontUseNativeDialog)
            self.infer_folder = folder_path
        else:
            folder_path = QFileDialog.getExistingDirectory(self, "Open folder", "./", options=QFileDialog.DontUseNativeDialog)

        self.logger.info('Selected Folder: {}'.format(folder_path))

    """ 取得檔案路徑 """
    def get_file(self):
        
        filename, filetype = QFileDialog.getOpenFileNames(self, "Open file", "./", options =QFileDialog.DontUseNativeDialog)
        if self.current_page_id==0:
            pass
        elif self.current_page_id==3:
            self.infer_files = filename

        self.logger.info('Selected File: {}'.format(filename))

    """ 安裝 NGC CLI """
    def install_ngc_event(self, data):
        if data=="exist" or data=="end":
            self.logger.info('Installed NGC CLI')
            self.insert_text("Done", t_fmt=False)
            self.insert_text("Choose a task ... ", endsym='')
        else:
            self.consoles[self.current_page_id].insertPlainText(data)

    """ 第一次進入 tab 1 的事件 """
    def t1_first_time_event(self):
        if self.t1_first_time:
            self.logger.info('First time loading tab 1 ... ')
            self.first_line=True

            itao_stats = 'Checking environment (iTAO) ... {}'.format('Actived' if self.check_tao() else 'Failed') # 檢查 itao 環境
            self.logger.info(itao_stats)
            self.insert_text(itao_stats)   
            
            self.insert_text('Installing NGC CLI ... ', endsym=' ') 

            self.ngc.start()    # 開始安裝
            self.ngc.trigger.connect(self.install_ngc_event)   # 綁定事件

            self.t1_first_time=False

    """ 第一次進入 tab 2 的事件 """
    def t2_first_time_event(self):
        if self.t2_firt_time:
            self.logger.info('First time loading tab 2 ... ')
            self.first_line=True
            
            BASIC = {
                'Epoch':'The number of epochs is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset.',
                'Batch': 'The batch size is a hyperparameter that defines the number of samples to work through before updating the internal model parameters.',
                'Checkpoint' :'If you want to resume your training, please press button to choose a pretrain model.'
            }

            self.insert_text('Setup specification for AI training ...', div=True, config=BASIC)

            self.insert_text('\n* Suggestion:', t_fmt=False)
            self.insert_text('Epoch -> 50~100 ( Depends on the value of loss)', t_fmt=False)
            self.insert_text('Batch Size -> 4, 8, 16 ( Higher value needs more memory of the GPU)', t_fmt=False)
            self.mv_cursor(pos='end')
            self.t2_firt_time=False

    """ 第一次進入 tab 3 的事件 """
    def t3_first_time_event(self):
        # prune and retrain
        if self.t3_first_time:
            # setup retrain spec
            self.logger.info('First time loading tab 3 ... ')
            self.logger.info('Define retrain specification ... ')

            self.first_line=True
            BASIC = {
                'Threshold':'Pruning removes parameters from the model to reduce the model size',
            }
            self.insert_text('Prune the AI model first ...', div=True, config=BASIC)
            self.insert_text('\n* Suggestion:', t_fmt=False)
            self.insert_text('Epoch -> 50~100 ( Depends on the value of loss)', t_fmt=False)
            self.insert_text('Threshold -> 0.3~0.6 (Higher `pth` gives you smaller model (and thus higher inference speed) but worse accuracy)', t_fmt=False)
            self.mv_cursor(pos='end')

            self.t3_first_time=False
            
    """ 更新頁面與按鈕 """
    def update_page(self):
        
        idx = self.ui.main_tab.currentIndex()
        self.current_page_id = idx
        self.logger.info('Page Change: Tab {} '.format(int(idx)+1))
        
        # self.logger.debug('Update status of button ... ')
        self.ui.bt_next.setText('Next')
        self.ui.main_tab.setTabEnabled(self.current_page_id, True)
        self.ui.bt_previous.setEnabled(self.page_buttons_status[self.current_page_id][0])
        self.ui.bt_next.setEnabled(self.page_buttons_status[self.current_page_id][1])
        
        if self.current_page_id==0:
            self.t1_first_time_event()

        elif self.current_page_id==1:
            self.t2_first_time_event()

        elif self.current_page_id==2:
            self.t3_first_time_event()
        else:
            self.first_line=True
            self.ui.bt_next.setText('Close')

    """ 更新頁面的事件 next, previous 按鈕 """
    def ctrl_page_event(self):
        trg = self.sender().text().lower()
        if trg=="next":
            if self.current_page_id < self.end_page_id :
                self.current_page_id = self.current_page_id + 1
                self.ui.main_tab.setCurrentIndex(self.current_page_id)
        elif trg=="close":
            self.close()
        else:   # previous
            if self.current_page_id > self.first_page_id :
                self.current_page_id = self.current_page_id - 1
                self.ui.main_tab.setCurrentIndex(self.current_page_id)
        
    """ 初始化圖表 """
    def init_plot(self, idx=None, xlabel="Epochs", ylabel="Loss", clean=False):
        idx = self.current_page_id if idx==None else idx    # 取得頁面
            
        if self.pws[idx]==None: # 如果沒有圖表則跳出
            self.logger.error('No frame in this tab ... ')
            return  

        if clean: self.pws[idx].clear() # 如果有圖表就先清空        
        self.pws[idx].addLegend(offset=(0., .5))        # 加入說明 設定為右上
        self.pws[idx].setLabel("left", ylabel)          # 加入y軸標籤
        self.pws[idx].setLabel("bottom", xlabel)        # 加入x軸標籤
        self.frames[idx].setLayout(self.pw_lyrs[idx])   # 設定 layout 
        self.pw_lyrs[idx].addWidget(self.pws[idx])      # 將圖表加入設定好的 layout
        
        if idx==1:
            [val.clear() for _, val in self.t2_var.items() ]
            epoch = int( self.itao_env.get_env('TRAIN', 'EPOCH') )
        elif idx==2:
            [val.clear() for _, val in self.t3_var.items() ]
            epoch = int( self.itao_env.get_env('RETRAIN', 'EPOCH') ) if ylabel!="MB" else 5

        self.pws[idx].setXRange(1, epoch)
        self.pws[idx].showGrid(x=True, y=True)          # 顯示圖表
        self.pws[self.current_page_id].show()

    """ 初始化 Console """
    def init_console(self):
        self.consoles[self.current_page_id].clear()    
        self.first_line=True
        # if self.current_page_id in [1,2]:
        #     if self.current_page_id==1: [val.clear() for _, val in self.t2_var.items() ]
        #     if self.current_page_id==2: [val.clear() for _, val in self.t3_var.items() ]

    """ 更新進度條，如果進度條滿了也會有對應對動作 """
    def update_progress(self, idx, cur, limit):
        val = int(cur*(100/limit))
        self.progress[idx].setValue(val)
        if val>=100:
            self.page_finished_event()
    
    """ 取得 tao 當前的任務 """
    # def get_tao_task(self):
    #     if "classification" in self.itao_env.get_env('NGC_TASK').lower():
    #         return "classification"
    #     else:
    #         return self.itao_env.get_env('MODEL').lower()

    """ 檢查並建立資料夾 """
    def check_dir(self, path):
        if not os.path.exists(path): os.makedirs(path)

    """ 掛載路徑 """
    def mount_env(self):
        self.logger.info('Update environ of mount file ... ')

        # local_project_dir = os.path.join(os.getcwd(), 'tasks')
        # tao_task = self.get_tao_task()
        # tao_task = self.itao_env.get_env('TASK')
                
        # local_task_dir = os.path.join(local_project_dir, tao_task)
        # local_data_dir = os.path.join(local_project_dir, 'data')
        # local_spec_dir = os.path.join(local_task_dir, 'specs')
        # local_out_dir = os.path.join(local_task_dir, 'output')
        
        # self.itao_env.update('LOCAL_PROJECT_DIR', local_project_dir)
        # self.itao_env.update('LOCAL_DATA_DIR', local_data_dir)
        # self.itao_env.update('LOCAL_EXPERIMENT_DIR', local_task_dir)
        # self.itao_env.update('LOCAL_SPECS_DIR', local_spec_dir)

        # self.itao_env.update('LOCAL_OUTPUT_DIR', local_out_dir)
        # self.check_dir(local_out_dir)

        ########################################################################################
        # dest_project_dir = self.itao_env.get_workspace_path()
        # dest_dir = os.path.join(dest_project_dir, tao_task)
        # dest_data_dir = os.path.join(dest_project_dir, 'data')
        # dest_spec_dir = os.path.join(dest_dir, 'specs')
        # dest_out_dir = os.path.join(dest_dir, 'output')
        
        # self.itao_env.update('USER_EXPERIMENT_DIR', dest_dir)
        # self.itao_env.update('DATA_DOWNLOAD_DIR', dest_data_dir)
        # self.itao_env.update('SPECS_DIR', dest_spec_dir)
        # self.itao_env.update('OUTPUT_DIR', dest_out_dir)
        # self.itao_env.update2('TRAIN', 'OUTPUT_DIR', self.itao_env.replace_docker_root(local_out_dir))

        ########################################################################################
        ret = self.itao_env.create_mount_json()
        self.insert_text("Creating Mount File ... {}".format(
            "Sucessed!" if ret else "Failed!"
            ))

    """ 進度條滿了 -> 頁面任務完成 -> 對應對動作 """
    def page_finished_event(self):
        if self.current_page_id==0:
            self.mount_env()
            self.insert_text("Show config", config=self.itao_env.get_env('TRAIN'))
            self.swith_page_button(previous=0, next=1)

            self.train_spec.set_label_for_detection(key='target_class_mapping')

        elif self.current_page_id==1:
            pass
        elif self.current_page_id==2:
            pass
        elif self.current_page_id==3:
            pass
        else:
            pass

    """ 於對應的區域顯示對應的配置檔內容 """
    def insert_text(self, title, t_fmt=True, config=None, endsym='\n', div=False):
        
        if t_fmt == True and self.div_is_inserted and not self.first_line:
            self.consoles[self.current_page_id].insertPlainText(f"{self.div_symbol}")
            self.div_is_inserted=False

        self.mv_cursor(pos='end')
        time_format = ""
        if t_fmt:
            now = datetime.datetime.now()
            time_format = "{:02}-{:02} {:02}:{:02}:{:02} {:2}".format(now.month, now.day, now.hour, now.minute, now.second, " ")

        self.consoles[self.current_page_id].insertPlainText(f"{time_format}{title}{endsym}")
        
        if div == True or config != None:
            self.consoles[self.current_page_id].insertPlainText(f"{self.div_symbol}")
            self.div_is_inserted=True

        if config != None:
            for key, val in config.items():
                if val !="":
                    self.consoles[self.current_page_id].insertPlainText(f"{key:<16}: {val}\n")
                    
        self.first_line = False
        self.mv_cursor(pos='end')
        self.consoles[self.current_page_id].update()
    
    """ backup all content """
    def backup_console(self):
        self.console_cnt = self.consoles[self.current_page_id].toPlainText()
    
    """ restore all content """
    def restore_console(self):
        self.consoles[self.current_page_id].setPlainText(self.console_cnt)

    """ 用於修改各自頁面的狀態 """
    def swith_page_button(self, previous, next=None):
        self.page_buttons_status[self.current_page_id][:] = [previous, next if next !=None else previous]
        self.ui.bt_previous.setEnabled(self.page_buttons_status[self.current_page_id][0])
        self.ui.bt_next.setEnabled(self.page_buttons_status[self.current_page_id][1] if next != None else self.page_buttons_status[self.current_page_id][0])

    """ 移動到最後一行"""
    def mv_cursor(self, pos='start'):
        if pos=='start':
            self.consoles[self.current_page_id].textCursor().movePosition(QtGui.QTextCursor.Start)  # 將位置移到LOG最下方 (1)
            self.consoles[self.current_page_id].ensureCursorVisible()                               # 將位置移到LOG最下方 (2)
        elif pos=='end':
            cursor = self.consoles[self.current_page_id].textCursor()
            cursor.movePosition(QtGui.QTextCursor.End)  # 將位置移到LOG最下方 (1)
            self.consoles[self.current_page_id].setTextCursor(cursor)
            self.consoles[self.current_page_id].ensureCursorVisible()                               # 將位置移到LOG最下方 (2)

    def tao_stop_event(self, data):
        if data == "end":
            self.insert_text('Done', t_fmt=False)
            if self.current_page_id==1:
                self.ui.t2_bt_train.setEnabled(True)
                self.ui.t2_bt_stop.setEnabled(False) 
            # if self.debug:
            self.swith_page_button(True)
