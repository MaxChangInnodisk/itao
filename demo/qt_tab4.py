#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys, os
from typing import *
from xml.etree.ElementTree import iterparse
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QFileDialog

""" 自己的函式庫自己撈"""
sys.path.append(os.path.abspath(f'{os.getcwd()}'))
from itao.utils.csv_tools import csv_to_list
from demo.qt_init import Init
        
class Tab4(Init):
    
    def __init__(self):
        super().__init__()

        # basic variable
        self.precision_radio = {"INT8":self.ui.t4_int8, "FP16":self.ui.t4_fp16, "FP32":self.ui.t4_fp32}
        self.worker_infer = None
        self.infer_files = None
        self.ls_infer_name, self.ls_infer_label = [], []
        self.cur_pixmap = 0
        self.small_frame_size, self.big_frame_size = 0, 0

        # connect event to button
        self.ui.t4_bt_upload.clicked.connect(self.get_infer_data_folder)
        self.ui.t4_bt_infer.clicked.connect(self.infer_event)
        self.ui.t4_bt_export.clicked.connect(self.export_event)
        self.ui.t4_bt_pre_infer.clicked.connect(self.ctrl_result_event)
        self.ui.t4_bt_next_infer.clicked.connect(self.ctrl_result_event)
        # self.ui.t4_combo_infer_model.currentIndexChanged.connect(self.update_infer_model)

        # define key of scheduler
        self.export_log_key = [ "Registry: ['nvcr.io']",
                                "keras_exporter",
                                "keras2onnx",
                                "Stopping container" ]
        # for debug
        self.run_t4_option = { 'infer':True, 'export':True }
        if self.debug:
            for key in self.run_t4_option.keys():
                self.run_t4_option[key]=True if int(self.debug_page)==4 and key==self.debug_opt else False

        self.infer_key = ['root: Registry', 'Loading experiment spec','Processing', 'Inference complete', 'Stopping container']

    """ 取得 T4 應該執行的功能 """
    def update_t4_actions(self):
        act_enabled = []
        act_disabled = []
        if self.debug:
            for key in self.run_t4_option.keys():
                if int(self.debug_page)==4 and key==self.debug_opt:
                    self.run_t4_option[key]=True
                    act_enabled.append(key)
                else:
                    self.run_t4_option[key]=False
                    act_disabled.append(key)
        
        if self.is_docker:
            self.infer_key.pop(0)
            self.infer_key.pop(4-1)
            print(self.infer_key)
                    
        self.logger.info("T4 Actions {} is enabled".format(act_enabled))
        self.logger.info("T4 Actions {} is disabled".format(act_disabled))

    """ 第一次進入 tab 4 的事件 """
    def t4_first_time_event(self):
        if self.t4_first_time:
            self.update_t4_actions()
            
            self.first_line=True
            self.ui.bt_next.setText('Close')

            # 放在這裡綁定主要是怕更新到，一直報錯
            self.ui.t4_combo_export_model.currentIndexChanged.connect(self.update_export_model)

            self.t4_first_time = False

    """ 取得要運行 Inference 的資料夾 """
    def get_infer_data_folder(self):
        folder_path = None

        root = "./tasks/data"
        folder_path = QFileDialog.getExistingDirectory(self, "Open folder", root, options=QFileDialog.DontUseNativeDialog)
        self.infer_folder = folder_path
        self.itao_env.update2('INFER', 'LOCAL_INPUT_DATA', folder_path)
        self.itao_env.update2('INFER', 'INPUT_DATA', self.itao_env.replace_docker_root(folder_path))

        self.logger.info('Selected Folder: {}'.format(folder_path))

    # Export -------------------------------------------------------------------------------

    """ 檢查 radio 按了哪個 """
    def check_radio(self):
        for precision, radio in self.precision_radio.items():
            if radio.isChecked(): return precision
        return ''

    """ 更新最新選擇要匯出的模型 """
    def update_export_model(self):
        if not self.t4_first_time:
            sel_model = self.ui.t4_combo_export_model.currentText()
            root = os.path.dirname(self.itao_env.get_env('RETRAIN', 'OUTPUT_MODEL'))
            intput_model = os.path.join( root, sel_model) 
            self.itao_env.update2('EXPORT', 'INPUT_MODEL', intput_model)

    """ 當 export 完成的時候 """
    def export_finish(self):
        info = "Export ... Done ! \n"
        self.logger.info(info)
        self.insert_text(info)
        
        self.update_progress(self.current_page_id, len(self.export_log_key), len(self.export_log_key))
        if self.worker_export is not None: self.worker_export.quit()
        self.swith_page_button(True)

    """ 更新輸出的LOG """
    def update_export_log(self, data):
        if data != "end":
            self.insert_text(data, t_fmt=False)
            self.mv_cursor()
            [ self.update_progress(self.current_page_id, self.export_log_key.index(key)+1, len(self.export_log_key))  for key in self.export_log_key if key in data ]  
        else:
            self.export_finish()

    """ 警告視窗：刪除已存在的模型 """
    def show_delete_msg(self):
        # set default
        title = 'Warning Message ( Delete Event )'
        msg = 'Do you want to delete the existed model ? \n( OK to continue, Cancel to backup and export again)'
        # create msg box
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setWindowTitle(title)
        msgBox.setText(msg)
        
        msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        returnValue = msgBox.exec()
        if returnValue == QMessageBox.Ok:
            self.logger.warning('User agree with delete export model')
            return 1
        elif returnValue == QMessageBox.Cancel:
            self.logger.warning('User cancel the action of delete')
            return 0 
        
    """ 更新 輸出用的參數 """
    def update_export_conf(self):
        # get precision
        precision = self.check_radio()
        self.itao_env.update2('EXPORT', 'PRECISION', precision)

        # setup export path
        export_name = self.ui.t4_etlt_name.text()
        export_dir = os.path.join( self.itao_env.get_env('USER_EXPERIMENT_DIR'), 'export')
        
        local_export_dir = self.itao_env.replace_docker_root(export_dir, mode='root')
        
        if os.path.exists(local_export_dir):
            import shutil
            self.logger.warning('Clean export directory.')
            ret = self.run_t4_option['export'] = self.show_delete_msg()
            if ret:
                shutil.rmtree( local_export_dir )
            else:
                self.logger.warning('break from `update export conf`')
                return

        os.makedirs(local_export_dir)

        export_model_path = os.path.join( export_dir, export_name)
        self.itao_env.update2('EXPORT', 'OUTPUT_MODEL', export_model_path)

        # setup input model
        self.update_export_model()
        # self.itao_env.update2('EXPORT', 'INPUT_MODEL', self.itao_env.get_env('RETRAIN', 'OUTPUT_MODEL'))

        # show information
        info = f"Export Path : {export_dir}\n"

    """ 匯出的事件 """
    def export_event(self):

        self.init_console()
        self.update_export_conf()

        info = 'Export Model ... '
        self.logger.info(info)
        self.insert_text(info)

        
        cmd_args = {
            'task':self.itao_env.get_env('TASK'), 
            'key':self.itao_env.get_env('KEY'), 
            'spec': self.itao_env.get_env('RETRAIN','SPECS'), 
            'intput_model': self.itao_env.get_env('RETRAIN', 'INPUT_MODEL'), 
            'output_model': self.itao_env.get_env('EXPORT', 'OUTPUT_MODEL'), 
            'dtype': self.itao_env.get_env('EXPORT', 'PRECISION'),
            'is_docker':self.is_docker 
        }

        self.worker_export = self.export_cmd( args=cmd_args )

        if self.run_t4_option['export']:
            self.worker_export.start()
            self.worker_export.trigger.connect(self.update_export_log)
        else:
            self.export_finish()

    # Infer -------------------------------------------------------------------------------

    """ 更新最新選擇要推論的模型 """
    def update_infer_model(self):
        sel_model = self.ui.t4_combo_infer_model.currentText()
        root = os.path.dirname(self.itao_env.get_env('RETRAIN', 'OUTPUT_MODEL'))
        intput_model = os.path.join( root, sel_model) 
        self.itao_env.update2('INFER', 'INPUT_MODEL', intput_model)
          
    """ 更新 infer 用的參數 """
    def update_infer_conf(self):
        self.itao_env.update2('INFER', 'SPECS', self.itao_env.get_env('RETRAIN', 'SPECS'))
        self.itao_env.update2('INFER', 'INPUT_MODEL', self.itao_env.get_env('RETRAIN', 'OUTPUT_MODEL'))
        self.itao_env.update2('INFER', 'BATCH_SIZE', self.itao_env.get_env('RETRAIN', 'BATCH_SIZE'))

        # updating inference model -> INFER, INPUT_MODEL
        self.update_infer_model()

        # create inference folder
        local_results_dir = os.path.join(self.itao_env.get_env('LOCAL_PROJECT_DIR'), 'results')
        if not os.path.exists(local_results_dir):
            self.logger.info('Create folder to saving results of inference.')
            os.makedirs(local_results_dir)
        self.itao_env.update2('INFER', 'LOCAL_RESULTS_DIR', local_results_dir)

        # setup path of images and labels which is generated after inference
        infer_image_folder = os.path.join(local_results_dir, f'{self.itao_env.get_env("TASK")}/images')
        infer_label_folder = os.path.join(local_results_dir, f'{self.itao_env.get_env("TASK")}/labels')
        self.itao_env.update2('INFER', 'LOCAL_RES_IMG_DIR', infer_image_folder)
        self.itao_env.update2('INFER', 'LOCAL_RES_LBL_DIR', infer_label_folder)
        self.itao_env.update2('INFER', 'RES_IMG_DIR', self.itao_env.replace_docker_root(infer_image_folder))
        self.itao_env.update2('INFER', 'RES_LBL_DIR', self.itao_env.replace_docker_root(infer_label_folder))
        if not os.path.exists(infer_image_folder): os.makedirs(infer_image_folder)
        if not os.path.exists(infer_label_folder): os.makedirs(infer_label_folder)

        if 'classification' in self.itao_env.get_env('TASK'):
            
            self.itao_env.update2('INFER', 'CLASS_MAP', os.path.join( self.itao_env.get_env('RETRAIN','OUTPUT_DIR'), 'classmap.json'))

        # self.itao_env.update2('INFER', 'INPUT_MODEL', self.itao_env.get_env('RETRAIN', 'OUTPUT_MODEL'))

        pass

    """ 按下 Inference 按鈕之事件 """
    def infer_event(self):
        self.init_console()
        info = "Do Inference ... "
        self.logger.info(info)
        self.insert_text(info, div=True)
        self.update_infer_conf()

        # define arguments of command line
        #-----------------------------------------------------------------------------------
        cmd_args = {
            'task' : self.itao_env.get_env('TASK'),
            'key' : self.itao_env.get_env('KEY'),
            'spec' : self.itao_env.get_env('INFER', 'SPECS'),
            'model': self.itao_env.get_env('RETRAIN', 'OUTPUT_MODEL'),
            'input_dir': self.itao_env.get_env('INFER', 'INPUT_DATA'),
            'is_docker':self.is_docker
        }

        # add option of gpu
        cmd_args['num_gpus'] = self.itao_env.get_env('NUM_GPUS')
        cmd_args['gpu_index'] = self.gpu_idx

        if 'classification' in self.itao_env.get_env('TASK'):
            # [ 'task', 'spec', 'key', 'model', 'input_dir', 'batch_size', 'class_map' ]
            cmd_args['batch_size'] = self.itao_env.get_env('INFER', 'BATCH_SIZE')
            cmd_args['class_map'] = self.itao_env.get_env('INFER', 'CLASS_MAP')

        elif 'yolo' in self.itao_env.get_env('TASK'):
            # [ 'task', 'spec', 'key', 'model', 'input_dir', 'output_dir', 'output_label' ]
            cmd_args['output_dir'] = self.itao_env.get_env('INFER', 'RES_IMG_DIR')
            cmd_args['output_label'] = self.itao_env.get_env('INFER', 'RES_LBL_DIR')
        #-----------------------------------------------------------------------------------

        # define worker and run
        self.worker_infer = self.infer_cmd( args=cmd_args )

        if self.run_t4_option['infer']:   
            self.worker_infer.trigger.connect(self.update_infer_log)
            self.worker_infer.info.connect(self.update_infer_log)
            self.worker_infer.start()
        else:
            self.infer_finish_event()

    """ 更新 Inference 的資訊 """
    def update_infer_log(self, data):
        if type(data)==str:

            self.insert_text(data, t_fmt=False)
            self.mv_cursor()
            [ self.update_progress(self.current_page_id, self.infer_key.index(key)+1, len(self.infer_key))  for key in self.infer_key if key in data ]
        
        elif type(data)==dict:
            
            if bool(data):
                self.insert_text(data, t_fmt=False)
                self.mv_cursor()
            else:
                self.worker_infer.quit()
                self.infer_finish_event()
        
    """ 完成 Inference 之後的事件 """
    def infer_finish_event(self):

        info = "Inference ... Done ! \n"
        self.logger.info(info)
        self.insert_text(info)

        self.ui.t4_bt_next_infer.setEnabled(True)
        self.ui.t4_bt_pre_infer.setEnabled(True)
        self.swith_page_button(True)

        if self.itao_env.get_env('TASK') == 'classification':
            self.load_result_classification()
        else:
            self.load_result_detection()
    
    # Show Results -------------------------------------------------------------------------------

    """ classification 用的 load results """
    def load_result_classification(self):
        # 更新大小，在這裡更新才會是正確的大小
        self.frame_size = self.ui.t4_frame.width() if self.ui.t4_frame.width()<self.ui.t4_frame.height() else self.ui.t4_frame.height()
        
        # 把所有的檔案給 Load 進 ls_infer_name
        self.cur_pixmap = 0

        # get label info from csv
        csv_path = os.path.join(self.infer_folder, 'result.csv')
        results = csv_to_list(csv_path)

        for res in results:
            file_path, det_class, det_prob = res
            file_path = self.itao_env.replace_docker_root(file_path, mode='root')
            self.ls_infer_name.append(file_path)
            self.ls_infer_label.append([det_class, det_prob])

        self.show_result()

    """ object detection 用的 load results """
    def load_result_detection(self):
        # 更新大小，在這裡更新才會是正確的大小
        self.frame_size = self.ui.t4_frame.width() if self.ui.t4_frame.width()<self.ui.t4_frame.height() else self.ui.t4_frame.height()
        
        infer_img_dir = self.itao_env.get_env('INFER', 'LOCAL_RES_IMG_DIR')
        infer_lbl_dir = self.itao_env.get_env('INFER', 'LOCAL_RES_LBL_DIR')

        # 把所有的檔案給 Load 進 ls_infer_name
        self.cur_pixmap = 0
        for file in os.listdir(infer_img_dir):
            base_name = os.path.basename(file)
            # 儲存名稱的相對路徑
            self.ls_infer_name.append(os.path.join( infer_img_dir, base_name ))
            # 儲存標籤檔的相對路徑
            label_name = os.path.splitext(os.path.join( infer_lbl_dir, base_name ))[0]+'.txt'
            
            with open(label_name, 'r') as lbl:
                result = []
                content = lbl.readlines()
                for cnt in content:
                    cnts = cnt.split(' ')
                    label, bbox, prob = cnts[0], tuple([ int(float(c)) for c in cnts[4:8] ]), float(cnts[-1])
                    if prob > self.ui.t4_thres.value():
                        result.append('{}, {:03}, {}'.format(label, prob, bbox))

                self.ls_infer_label.append(result)

        self.show_result()

    """ 控制顯示結果的事件 """
    def ctrl_result_event(self):
        who = self.sender().text()
        if who=="<":
            if self.cur_pixmap > 0: self.cur_pixmap = self.cur_pixmap - 1
            self.show_result()
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
        self.insert_text('\n', t_fmt=False)
        self.insert_text(self.ls_infer_name[self.cur_pixmap].replace(self.itao_env.get_env('LOCAL_PROJECT_DIR'),""), t_fmt=False, div=True)
        # update postion of cursor
        self.mv_cursor()

        # show result of target file
        if self.itao_env.get_env('TASK') == 'classification':
            self.insert_text(self.ls_infer_label[self.cur_pixmap], t_fmt=False)
        else:
            [ self.insert_text(f"{idx}: {cnt}", t_fmt=False) for idx,cnt in enumerate(self.ls_infer_label[self.cur_pixmap]) ]
