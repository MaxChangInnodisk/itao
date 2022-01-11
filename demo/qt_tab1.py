#!/usr/bin/python3
# -*- coding: utf-8 -*-
from logging import log
import sys, os
from typing import *
import pyqtgraph as pg
import importlib
from PyQt5.QtWidgets import QFileDialog, QComboBox

""" 自己的函式庫自己撈"""
import sys, os
sys.path.append(os.path.abspath(f'{os.getcwd()}'))
from itao.utils.spec_tools_v2 import DefineSpec
from itao.qtasks.download_model import DownloadModel
from itao.dataset_format import DSET_FMT_LS
from demo.qt_init import Init
from itao.utils import gpu_tools

class Tab1(Init):
    
    def __init__(self):
        super().__init__()
        
        self.t1_objects = [ self.ui.t1_combo_task, self.ui.t1_combo_model , self.ui.t1_combo_bone , self.ui.t1_combo_layer , self.ui.t1_bt_download ,self.ui.t1_bt_dset ,self.ui.t1_combo_gpus]
        self.sel_idx = [-1,-1,-1,-1,-1,-1,-1]

        self.ui.t1_combo_gpus.currentIndexChanged.connect(self.sel_gpu_event)

        # add option of tasks
        self.ui.t1_combo_task.clear()
        self.ui.t1_combo_task.addItems(list(self.option.keys()))
        self.ui.t1_combo_task.setCurrentIndex(-1)

        self.ui.t1_combo_task.currentIndexChanged.connect(self.get_task)
        self.ui.t1_combo_model.currentIndexChanged.connect(self.get_model)
        self.ui.t1_combo_bone.currentIndexChanged.connect(self.get_backbone)
        self.ui.t1_combo_layer.currentIndexChanged.connect(self.get_nlayer)
        self.ui.t1_bt_download.clicked.connect(self.pretrained_download)
        self.ui.t1_bt_dset.clicked.connect(self.get_dset)

        self.ui.t1_progress.valueChanged.connect(self.finish_options)

        self.debound = [0,0,0,0,0,0]
        self.setting_combo_box = False
        self.t1_info = ""

    """ 安裝 NGC CLI """
    def install_ngc_event(self, data):
        if data=="exist" or data=="end":
            self.logger.info('Installed NGC CLI')
            self.insert_text("Done", t_fmt=False)
            self.insert_text("Choose a task ... ", endsym='')
        else:
            self.consoles[self.current_page_id].insertPlainText(data)

    """ 取得 T1 應該執行的功能 """
    def update_t1_actions(self):
        pass

    """ 第一次進入 tab 1 的事件 """
    def t1_first_time_event(self):
        if self.t1_first_time:
            self.logger.info('First time loading tab 1 ... ')
            self.update_t1_actions()
            self.first_line=True

            itao_stats = 'Checking environment (iTAO) ... {}'.format('Actived' if self.check_tao() else 'Failed') # 檢查 itao 環境
            self.logger.info(itao_stats)
            self.insert_text(itao_stats)   
            
            self.insert_text('Installing NGC CLI ... ', endsym=' ') 

            self.ngc.start()    # 開始安裝
            self.ngc.trigger.connect(self.install_ngc_event)   # 綁定事件

            self.t1_first_time=False

    """ 取得能使用的 GPU (itao.utils.gpu_tools)  """
    def get_available_gpu(self) -> list:
            self.gpus = gpu_tools.get_available_device()
            gpus_name = [ gpu.name for gpu in self.gpus ]
            
            self.ui.t1_combo_gpus.clear()
            self.ui.t1_combo_gpus.addItems(gpus_name)
            self.ui.t1_combo_gpus.setCurrentIndex(-1)

    """ 更新環境變數 """
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

    """ 更新選項以及進度條 -> 當使用者往回調整的時候做出對應變化 """
    def update_options_and_bar(self, idx=0):

        for i in range(len(self.sel_idx)):
            if i>idx:
                self.sel_idx[i]=-1
                comboboxes_name = [ item.objectName() for item in self.findChildren(QComboBox) ]
                if self.t1_objects[i].objectName() in comboboxes_name: 
                    self.t1_objects[i].clear()
            else:
                self.sel_idx[i]= 1

        self.update_progress(self.current_page_id, self.sel_idx.count(1), len(self.t1_objects)) 

    # -------------------------------------------------------------------------------

    """ 取得任務 並更新 模型 清單 """
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
        idx=0
        self.update_options_and_bar(idx)

        # 設定下一個 combo box ( model )
        self.setting_combo_box = True   # 由於設定的時候會造成遞迴，所以要透過 setting_combo_box 來防止遞迴
        self.ui.t1_combo_model.clear()
        self.ui.t1_combo_model.addItems(list(self.option[self.ui.t1_combo_task.currentText()].keys()))   # 加入元素的時候會導致 編號改變而跳到下一個 method
        self.ui.t1_combo_model.setCurrentIndex(-1)
        self.ui.t1_combo_model.setEnabled(True)
        self.setting_combo_box = False

    """ 取得模型 並更新 主幹 清單 """
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
            idx = 1
            self.update_options_and_bar(idx)

            # 更新 元素
            self.setting_combo_box = True
            self.ui.t1_combo_bone.clear()
            self.ui.t1_combo_bone.addItems(list(self.option[task][model]))
            self.ui.t1_combo_bone.setCurrentIndex(-1)
            self.ui.t1_combo_bone.setEnabled(True)
            self.setting_combo_box = False

    """ 取得主幹 並更新 層數 清單 """
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
            idx = 2
            self.update_options_and_bar(idx)
            
            # 這邊需要更新itao_env.json 環境變數，後面才能夠取得 spec 的檔案
            self.update_env()

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

    """ 取得層數 """
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

            # 更新 進度條
            idx = 3
            self.update_options_and_bar(idx)
            
            # 延續上一個動作
            self.setting_combo_box = False
            if self.debound[3]==2: 
                self.ui.t1_bt_download.setEnabled(True)
                self.insert_text("Press button to download model from NVIDIA NGC ... ")

    """ 按下 download 開始下載 """
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

    """ 顯示數據級結構 """
    def get_dset_format(self):
        for key, val in DSET_FMT_LS.items():
            if key.lower() in self.itao_env.get_env('NGC_TASK').lower():
                return val

    """ 下載模型的事件 """
    def download_model_event(self, data):

        if '[END]' in data or "[EXIST]" in data:
            
            # 從資料取得模型的路徑，詳情需要去看 download_tools.py
            model_path = data.split(":")[1]
            finish_info  = 'Pre-trained Model is downloaded. ({})'.format(model_path)
            self.itao_env.update2('TRAIN', 'LOCAL_PRETRAINED_MODEL', model_path)
            self.itao_env.update2('TRAIN', 'PRETRAINED_MODEL', self.itao_env.replace_docker_root(model_path))
            self.logger.info(finish_info)
            self.insert_text(finish_info)

            # 其他
            self.ui.t1_bt_dset.setEnabled(True)
            self.insert_text("Please select a dataset with correct format ... ", div=True)
            self.insert_text(self.get_dset_format(), t_fmt=False)

            # 更新 進度條
            idx = 4
            self.update_options_and_bar(idx)
            
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

    """ 取得資料夾路徑 """
    def get_dset(self):

        folder_path = None

        folder_path = QFileDialog.getExistingDirectory(self, "Open folder", "./tasks/data", options=QFileDialog.DontUseNativeDialog)
        trg_folder_path = self.itao_env.replace_docker_root(folder_path)
        
        self.itao_env.update('LOCAL_DATASET', folder_path)
        self.itao_env.update('DATASET', trg_folder_path)
        
        # 更新 進度條
        idx = 5
        self.update_options_and_bar(idx)

        self.ui.t1_combo_gpus.setEnabled(True)

        self.logger.info('Selected Folder: {}'.format(folder_path))

        # get gpus
        self.get_available_gpu()
        
    """ 選擇 GPU 的事件 """
    def sel_gpu_event(self):
        
        cur_gpu = self.ui.t1_combo_gpus.currentText()
        
        for gpu in self.gpus:
            if gpu.name==cur_gpu:
                self.logger.info('Select GPU:{} ({})'.format(gpu.id, gpu.name))
                self.gpu_idx=gpu.id
        
        # 更新 進度條
        idx = 6
        self.update_options_and_bar(idx)

    """ 當 T1 的所有選項都選完了就會進行 Mapping Spec 的動作 """      
    def finish_options(self):

        if int(self.ui.t1_progress.value())>=100:
            info = 'Page 1 is finished, mapping varible into spec'
            self.insert_text(info)
            self.logger.info(info)
            self.mapping_spec()
            
            self.mount_env()
            self.insert_text("Show config", config=self.itao_env.get_env('TRAIN'))
            self.swith_page_button(previous=0, next=1)
            
            if 'detection' in self.itao_env.get_env('NGC_TASK'):
                self.train_spec.set_label_for_detection(key='target_class_mapping')
            

    """ 對應 Spec 的動作事件 """   
    def mapping_spec(self):

        # 更新 spec 裡面的 arch
        self.train_spec.mapping('arch', '"{}"'.format(self.itao_env.get_env('BACKBONE').lower()))

        # 更新 spec 的 n_layers
        if 'classification' in self.itao_env.get_env('TASK'):
            key_nlayer = 'n_layers'
        elif 'detection' in self.itao_env.get_env('NGC_TASK'):
            key_nlayer = 'nlayers'

        if not self.itao_env.get_env('NLAYER').isdigit():

            self.train_spec.del_spec_item(scope='model_config', key=key_nlayer)
            # 更新 spec 裡面的 arch
            self.train_spec.mapping('arch', '"{}_{}"'.format(self.itao_env.get_env('BACKBONE').lower(), self.itao_env.get_env('NLAYER')))

        else:
            if self.train_spec.find_key(key_nlayer):
                self.train_spec.mapping(key_nlayer, self.itao_env.get_env('NLAYER'))
            else:
                self.train_spec.add_spec_item(scope='model_config', key=key_nlayer, val=self.itao_env.get_env('NLAYER'), level=2)

        # 更新dataset
        trg_folder_path = self.itao_env.get_env('DATASET')
        task = self.itao_env.get_env('TASK')
        if task=='classification':       
            self.train_spec.mapping('train_dataset_path', '"{}"'.format(os.path.join(trg_folder_path, 'train')))
            self.train_spec.mapping('val_dataset_path', '"{}"'.format(os.path.join(trg_folder_path, 'val')))
            self.train_spec.mapping('eval_dataset_path', '"{}"'.format(os.path.join(trg_folder_path, 'test')))

        elif task=='yolo_v4':
            train_folder = os.path.join(trg_folder_path, 'train')
            test_folder = os.path.join(trg_folder_path, 'test')
            val_folder = os.path.join(trg_folder_path, 'val')

            self.train_spec.mapping_with_scope('data_sources', 'image_directory_path', '"{}"'.format(os.path.join(train_folder, 'images')))
            self.train_spec.mapping_with_scope('data_sources', 'label_directory_path', '"{}"'.format(os.path.join(train_folder, 'labels')))
            
            self.train_spec.mapping_with_scope('validation_data_sources', 'image_directory_path', '"{}"'.format(os.path.join(val_folder, 'images')))
            self.train_spec.mapping_with_scope('validation_data_sources', 'label_directory_path', '"{}"'.format(os.path.join(val_folder, 'labels')))
        