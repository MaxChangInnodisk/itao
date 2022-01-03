#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys, os
from typing import *
from PyQt5 import QtGui

""" 自己的函式庫自己撈"""
sys.path.append("../itao")
from itao.csv_tools import csv_to_list
from demo.qt_init import Init

INFER_IMG_ROOT = './infer_images'
INFER_LBL_ROOT = './infer_labels'
# DEBUG_MODE=True if len(sys.argv)>=2 and sys.argv[1].lower()=='debug' else False
DIV = "----------------------------------------------------\n"
        
class Tab4(Init):
    def __init__(self):
        super().__init__()

        self.precision_radio = {"INT8":self.ui.t4_int8, "FP16":self.ui.t4_fp16, "FP32":self.ui.t4_fp32}
        
        self.worker_infer, self.export_name, self.precision = None, None, None
        self.infer_files = None
        self.infer_folder = None

        self.ui.t4_bt_upload.clicked.connect(self.get_folder)
        self.ui.t4_bt_infer.clicked.connect(self.infer_event)
        self.ui.t4_bt_export.clicked.connect(self.export_event)

        self.export_log_key = [  "Registry: ['nvcr.io']",
                                "keras_exporter",
                                "keras2onnx",
                                "Stopping container" ]

        self.ui.t4_bt_pre_infer.clicked.connect(self.ctrl_result_event)
        self.ui.t4_bt_next_infer.clicked.connect(self.ctrl_result_event)
        self.ls_infer_name, self.ls_infer_label = [], []
        self.cur_pixmap = 0

        if self.debug:
            self.t4_debug = True
            if int(self.debug_page)==4:
                self.t4_debug = False

    """ 檢查 radio 按了哪個 """
    def check_radio(self):
        for precision, radio in self.precision_radio.items():
            if radio.isChecked(): return precision
        return ''

    
    def export_finish(self):
        info = "Export ... Done ! \n"
        self.logger.info(info)
        self.consoles[self.current_page_id].insertPlainText(info)
        self.update_progress(self.current_page_id, len(self.export_log_key), len(self.export_log_key))
        if self.worker_export is not None: self.worker_export.quit()
        self.swith_page_button(True)

    """ 更新輸出的LOG """
    def update_export_log(self, data):
        if data != "end":
            self.consoles[self.current_page_id].insertPlainText(f"{data}\n")
            self.mv_cursor()
            [ self.update_progress(self.current_page_id, self.export_log_key.index(key)+1, len(self.export_log_key))  for key in self.export_log_key if key in data ]  
        else:
            self.export_finish()

    """ 匯出的事件 """
    def export_event(self):

        info = 'Export Model ... '
        self.init_console()
        self.insert_text(info)
        self.logger.info(info)
        
        _export_name = self.ui.t4_etlt_name.toPlainText()
        _export_path = os.path.join( self.itao_env.get_env('USER_EXPERIMENT_DIR'), 'export')
        self.export_path = os.path.join( _export_path, _export_name)
        self.precision = self.check_radio()

        info = f"Export Path : {self.export_path}\n"
        self.logger.info(info)
        self.consoles[self.current_page_id].insertPlainText(info) 

        info = f"Precision : {self.precision}\n"
        self.logger.info(info)
        self.consoles[self.current_page_id].insertPlainText(info) 

        # self.worker_export = TAO_EXPORT()
        self.worker_export = ExportCMD(
            task = self.itao_env.get_env('TASK'),
            key = self.itao_env.get_env('KEY'),
            retrain_model = self.retrain_spec.find_key('model_path'),
            output_model= self.export_path
        )

        if not self.t4_debug:
            self.worker_export.start()
            self.worker_export.trigger.connect(self.update_export_log)
        else:
            self.export_finish()

    """ 按下 Inference 按鈕之事件 """
    def infer_event(self):
        self.init_console()
        info = "Do Inference ... "
        self.logger.info(info)
        self.insert_text(info, div=True)


        # self.worker_infer = TAO_INFER()
        self.worker_infer = InferCMD(
            task = self.itao_env.get_env('TASK'),
            key = self.itao_env.get_env('KEY'),
            spec = self.itao_env.replace_docker_root(self.retrain_spec.get_spec_path(), mode='docker'),
            retrain_model = self.retrain_spec.find_key('model_path'),
            batch_size=16,
            data = self.itao_env.replace_docker_root(self.infer_folder),
            classmap = os.path.join( self.itao_env.get_env('RETRAIN_OUTPUT_DIR'), 'classmap.json')
        )

        if not self.t4_debug:   
            self.worker_infer.start()
            self.worker_infer.trigger.connect(self.update_infer_log)
            self.worker_infer.info.connect(self.update_infer_log)
        else:
            self.infer_finish_event()

    """ 更新 Inference 的資訊 """
    def update_infer_log(self, data):
        if bool(data):
            self.consoles[self.current_page_id].insertPlainText(f"{data}\n")
        else:
            self.worker_infer.quit()
            self.infer_finish_event()
            
    def infer_finish_event(self):

        info = "Inference ... Done ! \n"
        self.logger.info(info)
        self.consoles[self.current_page_id].insertPlainText(info)

        self.ui.t4_bt_next_infer.setEnabled(True)
        self.ui.t4_bt_pre_infer.setEnabled(True)
        self.swith_page_button(True)

        if self.itao_env.get_env('TASK') == 'classification':
            self.new_load_result()
        else:
            self.load_result()
        
    def new_load_result(self):
        # 更新大小，在這裡更新才會是正確的大小
        self.frame_size = self.ui.t4_frame.width() if self.ui.t4_frame.width()<self.ui.t4_frame.height() else self.ui.t4_frame.height()
        
        # 把所有的檔案給 Load 進 ls_infer_name
        self.cur_pixmap = 0

        # get label info from csv
        csv_path = os.path.join(self.infer_folder, 'result.csv' if not self.t4_debug else 'debug_result.csv')
        results = csv_to_list(csv_path)

        for res in results:
            file_path, det_class, det_prob = res
            file_path = self.itao_env.replace_docker_root(file_path, mode='root')
            self.ls_infer_name.append(file_path)
            self.ls_infer_label.append([det_class, det_prob])

        self.show_result()

    """ 控制顯示結果的事件 """
    def ctrl_result_event(self):
        who = self.sender().text()
        if who=="<":
            if self.cur_pixmap > 0: self.cur_pixmap = self.cur_pixmap - 1
        else: # who==">":
            if self.cur_pixmap < len(self.ls_infer_name)-1: self.cur_pixmap = self.cur_pixmap + 1
        self.show_result()
    
    """ 將 pixmap、title、log 顯示出來， """
    def show_result(self):
        
        # setup pixmap
        # self.logger.info('Showing results ... ')
        pixmap = QtGui.QPixmap(self.ls_infer_name[self.cur_pixmap])
        self.ui.t4_frame.setPixmap(pixmap.scaled(self.frame_size-10, self.frame_size-10))

        # get file name and update information
        img_name = os.path.basename(self.ls_infer_name[self.cur_pixmap])
        self.ui.t4_infer_name.setText(img_name )
        self.insert_text(self.ls_infer_name[self.cur_pixmap].replace(self.itao_env.get_env('LOCAL_PROJECT_DIR'),""), t_fmt=False)
        
        # show result of target file
        if self.itao_env.get_env('TASK') == 'classification':
            self.insert_text(self.ls_infer_label[self.cur_pixmap], t_fmt=False)
        else:
            [ self.consoles[self.current_page_id].insertPlainText(f"{idx}: {cnt}") for idx,cnt in enumerate(self.ls_infer_label[self.cur_pixmap]) ]
        
        # update postion of cursor
        self.mv_cursor()

    """ 將名字與標籤檔儲存下來，方便後續調用 """
    def load_result(self):
        # 更新大小，在這裡更新才會是正確的大小
        self.frame_size = self.ui.t4_frame.width() if self.ui.t4_frame.width()<self.ui.t4_frame.height() else self.ui.t4_frame.height()
        
        # 把所有的檔案給 Load 進 ls_infer_name
        self.cur_pixmap = 0
        for file in self.infer_files:
            base_name = os.path.basename(file)
            # 儲存名稱的相對路徑
            self.ls_infer_name.append(os.path.join( INFER_IMG_ROOT, base_name ))
            # 儲存標籤檔的相對路徑
            label_name = os.path.splitext(os.path.join( INFER_LBL_ROOT, base_name ))[0]+'.txt'
            
            with open(label_name, 'r') as lbl:
                result = []
                content = lbl.readlines()
                [ result.append(cnt) for cnt in content if float(cnt.split(' ')[-1]) > self.ui.t4_thres.value() ]
                self.ls_infer_label.append(result)
        self.cur_pixmap = 0
        self.show_result()
