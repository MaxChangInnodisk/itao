from PyQt5 import QtGui, uic, QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QVBoxLayout, QMessageBox
from PyQt5.QtGui import QFont
import datetime
import sys, os
from typing import *
# from PyQt5.uic.uiparser import QtCore
import pyqtgraph as pg
import subprocess

""" 自己的函式庫自己撈"""
sys.path.append("../itao")
from itao.environ import SetupEnv
from itao.utils.spec_tools_v2 import DefineSpec
from itao.qtasks.install_ngc import InstallNGC
from itao.qtasks.stop_tao import StopTAO
from itao.utils.qt_logger import CustomLogger

from demo.configs import OPT, ARCH_LAYER

class Init(QtWidgets.QMainWindow):

    """ 定義共用變數 """
    def __init__(self) -> None:
        super().__init__() # Call the inherited classes __init__ method

        # 取得共用的 logger
        self.logger = CustomLogger().get_logger('dev')

        # iTAO 初始化
        self.logger.info('Initial iTAO ... ')
        self.ui = uic.loadUi(os.path.join("ui", "itao_v0.3.ui"), self) # Load the  file   # 使用 ui 檔案的方式
        self.setWindowTitle('iTAO')

        # 基本常數設定
        self.logger.info('Setting Basic Variable ... ')
        self.debug, self.debug_page, self.debug_opt, self.is_docker = None, None, None, None

        # 選項參數
        self.option, self.option_nlayer = OPT, ARCH_LAYER

        # 頁面設定：初次進入頁面&第一行
        self.first_page_id = 0  # 1-1 = 0
        self.end_page_id = 3    # 4-1 = 3
        self.t1_first_time, self.t2_firt_time, self.t3_first_time, self.t4_first_time = True, True, True, True
        

        # 與 TAO 的命令相關
        self.train_cmd, self.eval_cmd, self.retrain_cmd, self.prune_cmd, self.infer_cmd, self.export_cmd = None, None, None, None, None, None
        self.kmeans_cmd = None

        # 環境相關：spec的物件、env的物件、安裝ngc 的物件、關閉 tao 執行程式的物件
        self.train_spec, self.retrain_spec = None, None
        self.itao_env = None
        self.ngc = None
        self.stop_tao = None  
        
        # 與 Console 相關的參數
        self.div_symbol = "----------------------------------------------------\n"
        self.console_cnt = ""
        self.first_line = True
        self.div_is_inserted = False

        # 將元件統一
        self.page_buttons_status={0:[0,0], 1:[1,0], 2:[1,0], 3:[1,1]}
        self.tabs = [ self.ui.tab_1, self.ui.tab_2, self.ui.tab_3, self.ui.tab_4 ]
        self.progress = [ self.ui.t1_progress, self.ui.t2_progress, self.ui.t3_progress, self.ui.t4_progress]
        self.frames = [None, self.ui.t2_frame, self.ui.t3_frame, None]
        self.consoles = [ self.ui.t1_console, self.ui.t2_console, self.ui.t3_console, self.ui.t4_console]

        # 將所有分頁都關閉
        [ self.ui.main_tab.setTabEnabled(i, False if not self.debug else True ) for i in range(len(self.tabs))]     
        
        """ 建立 & 初始化 Tab2 跟 Tab3 的圖表 """
        pg.setConfigOptions(antialias=True)
        pg.setConfigOption('background', 'w')
        self.pws = [None, pg.PlotWidget(self), pg.PlotWidget(self), None]
        self.pw_lyrs = [None, QVBoxLayout(), QVBoxLayout(), None]
        [ a.hide() for a in self.pws if a!=None ]    # 先關閉等待 init_console 的時候才開

        # 設定 Previous、Next 的按鈕，綁定事件
        self.current_page_id = self.first_page_id   # 將當前頁面編號 (current_page_id) 設定為 第一個 ( first_page_id )
        self.ui.main_tab.setCurrentIndex(self.first_page_id)
        self.ui.bt_next.clicked.connect(self.ctrl_page_event)
        self.ui.bt_previous.clicked.connect(self.ctrl_page_event)
        
        # 設定 GPU 編號 (--gpu_index)
        self.gpu_idx = 0

    """ 開始運行 iTAO 的事件 """
    def start(self, debug, debug_page, debug_opt, is_docker):
        
        self.logger.info('Setting up running mode ...')
        
        # 取得到運行模式
        self.debug, self.debug_page, self.debug_opt, self.is_docker = debug, debug_page, debug_opt, is_docker

        self.itao_env = SetupEnv()  # 建立 configs/itao_env.json 檔案，目的在於建立共用的變數以及 Docker 與 Local 之間的路徑
        self.itao_env.create_env_file(is_docker=self.is_docker) 
        self.ngc = InstallNGC() # NGC 的安裝Thread
        self.stop_tao = StopTAO() # 統一關閉 TAO 的方法

        self.update_page()  # 更新頁面資訊
        self.ui.main_tab.currentChanged.connect(self.update_page)

        # 設定全螢幕
        self.change_font_size_event('Arial', 14)
        self.showFullScreen()   
        # 顯示並顯示警告視窗
        self.show()   
        self.show_warning_msg()

    # --------------------------------------------------------------------------------------
    """ 更新 Tab 要運行的功能 """
    def update_t1_actions(self):
        pass
    
    def update_t2_actions(self):
        pass

    def update_t3_actions(self):
        pass

    def update_t4_actions(self):
        pass
    # --------------------------------------------------------------------------------------
    """ 第一次進入 Tab 的事件 """
    def t1_first_time_event(self):
        pass

    def t2_first_time_event(self):
        pass

    def t3_first_time_event(self):
        pass
    
    def t4_first_time_event(self):
        pass
    # --------------------------------------------------------------------------------------
    """ 顯示警告視窗 """
    def show_warning_msg(self):
        # set default
        title = 'Warning Message ( Key Event )'
        msg = "F12: enter/exit full screen mode. \n\nEscape: quit the app."
        # create msg box
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setWindowTitle(title)
        msgBox.setText(msg)
        msgBox.setStandardButtons(QMessageBox.Ok)
        returnValue = msgBox.exec()
        if returnValue == QMessageBox.Ok:
            self.logger.info('Press OK with the warning message.')

    """ 按鍵按下的事件 """
    def keyPressEvent(self, event):
        
        if event.key() == QtCore.Qt.Key_Escape:
            # self.stop_tao.start()
            sys.exit(0)
        if event.key() == QtCore.Qt.Key_F12:
            if self.isFullScreen():
                self.change_font_size_event('Arial', 11)
                
                self.showNormal()
            else:
                self.change_font_size_event('Arial', 14)
                self.showFullScreen()
    
    """ 修改所有的字體大小 """
    def change_font_size_event(self, style='Arial', size=11):
        # setFont(QFont(style, size))
        self.ui.setFont(QFont(style, size)) # not work in all widgets

        from PyQt5.QtWidgets import QLabel, QDoubleSpinBox, QComboBox, QLineEdit, QTextBrowser, QPlainTextEdit, QPushButton

        [ item.setFont(QFont(style, size)) for item in self.findChildren(QLabel) ]
        [ item.setFont(QFont(style, size)) for item in self.findChildren(QDoubleSpinBox) ]
        [ item.setFont(QFont(style, size)) for item in self.findChildren(QComboBox) ]
        [ item.setFont(QFont(style, size)) for item in self.findChildren(QLineEdit) ]
        [ item.setFont(QFont(style, size)) for item in self.findChildren(QTextBrowser) ]
        [ item.setFont(QFont(style, size)) for item in self.findChildren(QPlainTextEdit) ]
        [ item.setFont(QFont(style, size-1)) for item in self.findChildren(QPushButton) ]
        
        # other
        self.ui.t1_option.setFont(QFont(style, size))
        
    """ 檢查 tao 的狀況 """
    def check_tao(self):
        proc = subprocess.run( ['tao','-h'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8", timeout=1)      
        return 1 if proc.returncode == 0 else 0

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
        elif self.current_page_id==3:
            self.t4_first_time_event()
        else:
            pass

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

    """ 更新進度條，如果進度條滿了也會有對應對動作 """
    def update_progress(self, idx, cur, limit):
        val = int(cur*(100/limit))
        self.progress[idx].setValue(val)
        if val>=100:
            self.page_finished_event()
    
    """ 檢查並建立資料夾 """
    def check_dir(self, path):
        if not os.path.exists(path): os.makedirs(path)

    """ 掛載路徑 """
    def mount_env(self):
        self.logger.info('Update environ of mount file ... ')
        ret = self.itao_env.create_mount_json()
        self.insert_text("Creating Mount File ... {}".format(
            "Sucessed!" if ret else "Failed!"
            ))

    """ 進度條滿了 -> 頁面任務完成 -> 對應對動作 """
    def page_finished_event(self):
        if self.current_page_id==0:
            pass
            # self.mount_env()
            # self.insert_text("Show config", config=self.itao_env.get_env('TRAIN'))
            # self.swith_page_button(previous=0, next=1)
            
            # if 'detection' in self.itao_env.get_env('TASK'):
            #     self.train_spec.set_label_for_detection(key='target_class_mapping')
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
    
    """ 備份所有 Console 內容 """
    def backup_console(self):
        self.console_cnt = self.consoles[self.current_page_id].toPlainText()
    
    """ 回覆剛剛備份的 Console 內容 """
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

    """ 停止所有的 TAO """
    def tao_stop_event(self, data):
        if data == "end":
            self.insert_text('Done', t_fmt=False)
            if self.current_page_id==1:
                self.ui.t2_bt_train.setEnabled(True)
                self.ui.t2_bt_stop.setEnabled(False) 
            # if self.debug:
            self.swith_page_button(True)

    """ 取得最新訓練的模型 """
    def trained_model_list(self, mode='TRAIN') -> list:
        
        output_dir = self.itao_env.get_env(mode, 'LOCAL_OUTPUT_DIR')
        trained_model_list = os.listdir( os.path.join(output_dir, 'weights'))
        trained_model_list.sort()
        new_trained_model_list =[ model for model in trained_model_list if self.train_spec.find_key('arch') in model ]

        min_idx, max_idx = 0, len(new_trained_model_list)

        for cur_idx, model in enumerate(new_trained_model_list):
            cur_epoch = os.path.splitext(model)[0].split('_')[-1]
            if cur_epoch.isdigit(): 
                cur_epoch = int(cur_epoch)
                if cur_epoch == int(self.itao_env.get_env(mode, 'EPOCH')):
                    max_idx = cur_idx
                    if max_idx>5:
                        min_idx = max_idx-5

        cut_trained_model = new_trained_model_list[min_idx:max_idx+1]
        cut_trained_model.reverse()
        return [ os.path.join( os.path.join(output_dir, 'weights'), model ) for model in cut_trained_model ] 
