#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys, os
from typing import *
import pyqtgraph as pg
import importlib

""" 自己的函式庫自己撈"""
sys.path.append("../itao")
import itao
from itao.utils.spec_tools_v2 import DefineSpec
from itao.qtasks.download_model import DownloadModel
from itao.dataset_format import get_dset_format, DSET_FMT_LS
from demo.qt_init import Init


class Tab1(Init):
    def __init__(self):
        super().__init__()
        
        self.t1_objects = [ self.ui.t1_combo_task, self.ui.t1_combo_model , self.ui.t1_combo_bone , self.ui.t1_combo_layer , self.ui.t1_bt_download ,self.ui.t1_bt_dset ]
        self.sel_idx = [-1,-1,-1,-1,-1,-1]

        # add option of tasks
        self.ui.t1_combo_task.clear()
        self.ui.t1_combo_task.addItems(list(self.option.keys()))
        self.ui.t1_combo_task.setCurrentIndex(-1)

        self.ui.t1_combo_task.currentIndexChanged.connect(self.get_task)
        self.ui.t1_combo_model.currentIndexChanged.connect(self.get_model)
        self.ui.t1_combo_bone.currentIndexChanged.connect(self.get_backbone)
        self.ui.t1_combo_layer.currentIndexChanged.connect(self.get_nlayer)
        self.ui.t1_bt_download.clicked.connect(self.pretrained_download)
        self.ui.t1_bt_dset.clicked.connect(self.get_folder)

        self.debound = [0,0,0,0,0,0]
        self.setting_combo_box = False

    ####################################################################################################
    #                                                T1                                                #
    ####################################################################################################

    """ T1 -> 取得任務 並更新 模型清單 """
    def get_task(self):
        
        if self.debound[0]==0: # 搭配上一個動作
            self.insert_text("Done", t_fmt=False)
            self.debound[0]=1
        
        # 取得選擇的任務
        self.logger.info('Update combo box: {}'.format('task'))
        # self.train_conf['task'] = self.ui.t1_combo_task.currentText()

        # 更新到文件當中方便使用
        ngc_task = self.ui.t1_combo_task.currentText().lower()
        ngc_task = 'classification' if 'classification' in ngc_task else ngc_task.replace(" ", "_")
        self.itao_env.update('NGC_TASK', ngc_task)
        
        # 更新進度條
        self.sel_idx[0]=1
        self.update_progress(self.current_page_id, self.sel_idx.count(1), len(self.t1_objects))

        # 設定下一個 combo box ( model )
        self.setting_combo_box = True   # 由於設定的時候會造成遞迴，所以要透過 setting_combo_box 來防止遞迴
        self.ui.t1_combo_model.clear()
        self.ui.t1_combo_model.addItems(list(self.option[self.ui.t1_combo_task.currentText()].keys()))   # 加入元素的時候會導致 編號改變而跳到下一個 method
        self.ui.t1_combo_model.setCurrentIndex(-1)
        self.ui.t1_combo_model.setEnabled(True)
        self.setting_combo_box = False

    def update_env(self):

        # 取得當前的工作路徑，與工作資料夾
        local_project_dir = os.path.join(os.getcwd(), 'tasks')
        self.itao_env.update('LOCAL_PROJECT_DIR', local_project_dir)

        # 取得當前任務的路徑
        task = self.itao_env.get_env('TASK')
        local_task_path = os.path.join( local_project_dir, task)
        self.itao_env.update('LOCAL_EXPERIMENT_DIR', local_task_path )
        self.itao_env.update('USER_EXPERIMENT_DIR', self.itao_env.replace_docker_root(local_task_path))

        # 取得數據集的資料夾
        local_data_dir = os.path.join(os.getcwd(), 'data')
        self.itao_env.update('LOCAL_DATA_DIR', local_data_dir)
        self.itao_env.update('DATA_DOWNLOAD_DIR', self.itao_env.replace_docker_root(local_task_path))

        # 取得輸出的資料夾
        local_output_dir = os.path.join(local_task_path, 'output')
        self.itao_env.update2('TRAIN', 'LOCAL_OUTPUT_DIR', local_output_dir)
        self.itao_env.update2('TRAIN', 'OUTPUT_DIR', self.itao_env.replace_docker_root(local_output_dir))
        self.check_dir(local_output_dir)

       # 更新 specs 的目錄
        local_spec_path = os.path.join(local_task_path, 'specs')
        self.itao_env.update('LOCAL_SPECS_DIR', local_spec_path)
        self.itao_env.update('SPECS_DIR', self.itao_env.replace_docker_root(local_spec_path))

        # 定義訓練的 spec
        self.train_spec = DefineSpec(mode='train')
        self.retrain_spec = DefineSpec(mode='retrain')

        # 取得特定的API
        self.logger.info('Loading Target Module ... ')
        # sys.path.append("../itao")
        self.module = importlib.import_module(f"itao.qtasks.{self.itao_env.get_env('TASK')}", package='../itao')
        self.train_cmd = self.module.TrainCMD
        self.eval_cmd = self.module.EvalCMD
        self.retrain_cmd = self.module.ReTrainCMD
        self.prune_cmd = self.module.PruneCMD
        self.infer_cmd = self.module.InferCMD
        self.export_cmd = self.module.ExportCMD
        try:
            self.kmeans_cmd = self.module.KmeansCMD
        except:
            self.kmeans = ""

 
            
    """ T1 -> 取得模型 並更新 主幹清單 """
    def get_model(self):
        if self.ui.t1_combo_model.currentIndex()== -1:
            # 延續上一個動作
            if self.debound[1]==0: 
                self.insert_text("Choose a model ... ", endsym="")
                self.debound[1]=1

        elif not self.setting_combo_box:    # 如果沒有正在設定 combo box 則繼續

            # 延續上一個動作
            if self.debound[1]==1: 
                self.insert_text("Done", t_fmt=False)
                self.debound[1]=2
            
            # 取得 model
            self.logger.info('Update combo box: {}'.format('model'))
            task = self.ui.t1_combo_task.currentText()
            model = self.ui.t1_combo_model.currentText()
            self.itao_env.update('TASK', 'classification' if 'classification' == self.itao_env.get_env('NGC_TASK') else model.lower())

            # 更新進度條
            self.sel_idx[1]=1
            self.update_progress(self.current_page_id, self.sel_idx.count(1), len(self.t1_objects))

            # 更新 元素
            self.setting_combo_box = True
            self.ui.t1_combo_bone.clear()
            self.ui.t1_combo_bone.addItems(list(self.option[task][model]))
            self.ui.t1_combo_bone.setCurrentIndex(-1)
            self.ui.t1_combo_bone.setEnabled(True)
            self.setting_combo_box = False

    """ T1 -> 取得主幹 並更新 層數清單 """
    def get_backbone(self):
        if self.ui.t1_combo_bone.currentIndex()== -1:
            # 延續上一個動作
            if self.debound[2]==0: 
                self.insert_text("Choose a backbone ... ", endsym="")
                self.debound[2]=1

        elif not self.setting_combo_box:
            # 延續上一個動作
            if self.debound[2]==1: 
                self.insert_text("Done", t_fmt=False)
                self.debound[2] = 2
            
            # 取得 backbone 的資訊
            self.logger.info('Update combo box: {}'.format('backbone'))
            self.itao_env.update('BACKBONE', self.ui.t1_combo_bone.currentText().lower())

            # 更新 進度條
            self.sel_idx[2]=1
            self.update_progress(self.current_page_id, self.sel_idx.count(1), len(self.t1_objects))

            # 這邊需要更新itao_env.json 環境變數，後面才能夠取得 spec 的檔案
            self.update_env()

            # 更新 spec 裡面的 arch
            self.train_spec.mapping('arch', '"{}"'.format(self.itao_env.get_env('BACKBONE').lower()))

            # 加入新的元素
            self.setting_combo_box = True
            if self.ui.t1_combo_bone.currentText() in self.option_nlayer.keys():
                self.ui.t1_combo_layer.clear()
                self.ui.t1_combo_layer.setEnabled(True)
                new_layers = [ layer.replace("_","") for layer in self.option_nlayer[self.ui.t1_combo_bone.currentText()]]
                self.ui.t1_combo_layer.addItems( new_layers )
                self.ui.t1_combo_layer.setCurrentIndex(-1)
                self.ui.t1_combo_layer.setEnabled(True)
                self.setting_combo_box = False

    """ T1 -> 取得層數 """
    def get_nlayer(self):
        if self.ui.t1_combo_layer.currentIndex()== -1:
            # 延續上一個動作
            if self.debound[3]==0:
                self.insert_text("Select a number of layer ... ", endsym="")
                self.debound[3]=1

        elif not self.setting_combo_box:
            # 延續上一個動作
            if self.debound[3]==1: 
                self.insert_text("Done", t_fmt=False)
                self.debound[3]=2

            # 更新 nlayer
            self.logger.info('Update combo box: {}'.format('n_layers'))
            self.itao_env.update('NLAYER', self.ui.t1_combo_layer.currentText())

            # 更新進度條
            self.sel_idx[3]=1
            self.update_progress(self.current_page_id, self.sel_idx.count(1), len(self.t1_objects)) 

            # 更新 spec 的 n_layers
            if 'classification' in self.itao_env.get_env('NGC_TASK'):
                self.train_spec.mapping('n_layers', self.itao_env.get_env('NLAYER'))
            elif 'detection' in self.itao_env.get_env('NGC_TASK'):
                self.train_spec.mapping('nlayers', self.itao_env.get_env('NLAYER'))
            
            # 延續上一個動作
            self.setting_combo_box = False
            if self.debound[3]==2: 
                self.ui.t1_bt_download.setEnabled(True)
                self.insert_text("Press button to download model from NVIDIA NGC ... ")

    """ T1 -> 按下 download 開始下載 """
    def pretrained_download(self):
        
        self.down_model = DownloadModel(
            model=self.itao_env.get_env('BACKBONE'),
            nlayer=self.itao_env.get_env('NLAYER')
        )
        # 下載模型的事件
        self.logger.info('Start to download model ... ')
        self.down_model.start()
        self.down_model.trigger.connect(self.download_model_event)
        self.insert_text('Downloading pre-trained model ... ')

        # 將 t1 console 先備份
        self.t1_info = self.consoles[self.current_page_id].toPlainText()
        self.final_info=""

    """ T1 -> 顯示數據級結構 """
    def get_dset_format(self):
        for key, val in DSET_FMT_LS.items():
            if key.lower() in self.itao_env.get_env('NGC_TASK').lower():
                return val

    """ T1 -> 下載模型的事件 """
    def download_model_event(self, data):

        if '[END]' in data or "[EXIST]" in data:
            
            # 從資料取得模型的路徑，詳情需要去看 download_tools.py
            model_path = data.split(":")[1]
            finish_info  = 'Pre-trained Model is downloaded. ({})'.format(model_path)
            self.itao_env.update2('TRAIN', 'LOCAL_PRETRAINED_MODEL', model_path)
            self.logger.info(finish_info)
            self.insert_text(finish_info)

            # 更新 specs 的 pretrained_model_path 的部份
            if 'classification' in self.itao_env.get_env('NGC_TASK'):
                self.train_spec.mapping('pretrained_model_path', f'"{self.itao_env.replace_docker_root(model_path)}"')
            elif 'detection' in self.itao_env.get_env('NGC_TASK'):
                self.train_spec.mapping('pretrain_model_path', f'"{self.itao_env.replace_docker_root(model_path)}"')
            
            # 其他
            self.ui.t1_bt_dset.setEnabled(True)
            self.insert_text("Please select a dataset with correct format ... ", div=True)
            self.insert_text(self.get_dset_format(), t_fmt=False)

            # 如果結束，會更新進度條
            self.sel_idx[4]=1
            self.update_progress(self.current_page_id, self.sel_idx.count(1), len(self.t1_objects))
        elif '[ERROR]' in data:
            info = data.split(":")[1]
            self.insert_text(info)
        # 如果還沒結束就會動態更新 console 內容
        else:
            if 'Download speed' in data:
                self.final_info = self.t1_info + data
                self.consoles[self.current_page_id].setPlainText(self.final_info)
            else:
                self.consoles[self.current_page_id].insertPlainText(data)
                self.mv_cursor(pos='end')