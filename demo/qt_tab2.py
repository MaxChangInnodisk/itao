#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys, os
from typing import *
import pyqtgraph as pg
import importlib

""" 自己的函式庫自己撈"""
sys.path.append("../itao")

from itao.environ import SetupEnv
from demo.qt_init import Init

class Tab2(Init):

    def __init__(self):
        super().__init__()

        self.t2_var = { "avg_epoch":[],
                        "avg_loss":[],
                        "val_epoch":[],
                        "val_loss":[] }
        self.env = SetupEnv()
        self.worker, self.worker_eval = None, None
        self.ui.t2_bt_train.clicked.connect(self.train_event)
        self.ui.t2_bt_stop.clicked.connect(self.stop_event)
        self.ui.t2_epoch.textChanged.connect(self.update_epoch_event)
        self.backup = False
        self.kmeans_enable = False

        self.run_t2_option = {
            'kmeans':True,
            'train':True,
            'eval':True
        }
        if self.debug:
            for key in self.run_t2_option.keys():
                self.run_t2_option[key]=True if int(self.debug_page)==2 and key==self.debug_opt else False

    """ 當按下 train 按鈕的時候進行的事件 """
    def train_event(self):

        self.update_train_conf()
        self.init_plot()
        self.init_console()

        if 'yolo' in self.itao_env.get_env('TASK'):
        
            info = 'Calculate kmeans for yolo ...'
            self.logger.info(info)
            self.insert_text(info)

            if self.run_t2_option['kmeans']: self.get_anchor_event()

        info = "Start training ... "
        self.logger.info(info)
        self.insert_text(info, div=True)

        self.ui.t2_bt_train.setEnabled(False)
        self.ui.t2_bt_stop.setEnabled(True)
        
        cmd_args = {
            'task': self.itao_env.get_env('TASK'), 
            'spec': self.itao_env.get_env('TRAIN','SPECS'),
            'output_dir': self.itao_env.get_env('TRAIN', 'OUTPUT_DIR'), 
            'key': self.itao_env.get_env('KEY'),
            'num_gpus': self.itao_env.get_env('NUM_GPUS'),
            'gpu_index':self.gpu_idx
        }

        self.worker = self.train_cmd( args = cmd_args )

        if self.run_t2_option['train']:
            self.worker.trigger.connect(self.update_train_log)
            self.worker.start()
        else:
            self.train_finish()

    """ 按下 stop 的事件 """
    def stop_event(self):
        
        info = 'Stoping TAO container ... '
        self.logger.info(info)
        self.insert_text(info, endsym='')

        self.ui.t2_bt_train.setEnabled(True)
        self.ui.t2_bt_stop.setEnabled(True) 
        
        self.stop_tao.trigger.connect(self.tao_stop_event)
        self.stop_tao.start()
        
        if self.worker != None: 
            self.worker.terminate()
            self.worker = None

        if self.worker_eval != None:
            self.worker_eval.terminate()
            self.worker_eval = None

    """ 完成驗證後的動作 """
    def eval_finish(self):
        info = "Evaluating Model ... Done !\n"
        self.logger.info(info)
        self.insert_text(info)

        self.swith_page_button(previous=1, next=1)

    """ 更新 eval 的內容 """
    def update_eval_log(self, data):
        if data == 'end':
            self.eval_finish()
            self.worker_eval.quit()
            return

        self.consoles[self.current_page_id].insertPlainText(f"{data}\n")                                # 插入內容
        self.mv_cursor()
    
    """ YOLO 需要添加 Anchor """
    def get_anchor_event(self):
        args = {
            'task':self.itao_env.get_env('TASK'), 
            'train_image':self.train_spec.find_key('image_directory_path'), 
            'train_label':self.train_spec.find_key('label_directory_path'),
            'n_ancher':9, 
            'image_width':self.train_spec.find_key('output_width'), 
            'image_height':self.train_spec.find_key('output_height')
        }
        self.worker_kmeans = self.kmeans_cmd(args=args)
        self.kmeans_idx = 0
        self.kmeans_sub_idx = 0
        self.kmeans = ''
        self.worker_kmeans.trigger.connect(self.update_anchor)
        self.worker_kmeans.start()

        self.insert_text('Waiting for calculating kmeans ... ', endsym='')
        self.worker_kmeans.wait()    
        self.insert_text('Done', t_fmt=False)

    """ 更新 achor """
    def update_anchor(self, data):
        if bool(data):            
            self.train_spec.mapping('small_anchor_shape', f'"[{data[0]}]"' )
            self.train_spec.mapping('mid_anchor_shape', f'"[{data[1]}]"')
            self.train_spec.mapping('big_anchor_shape', f'"[{data[2]}]"')
            self.worker_kmeans.quit()
        else:
            self.logger.error("kmeans error: {}".format(data))
            self.worker_kmeans.quit()

    """ 更新 console 內容 """
    def update_train_log(self, data):
        
        if bool(data):

            if len(data.keys())>1:
                cur_epoch, avg_loss, val_loss, max_epoch = data['epoch'], data['avg_loss'], data['val_loss'], int(self.itao_env.get_env('TRAIN','EPOCH'))
                
                log=""
                self.t2_var["val_epoch"].append(cur_epoch)
                self.t2_var["val_loss"].append(val_loss)
                self.t2_var["avg_epoch"].append(cur_epoch)
                self.t2_var["avg_loss"].append(avg_loss)
                
                log = "{}  {} {} {}".format(  f'Epoch: {cur_epoch:03}/{max_epoch:03}',
                                            f'Loss: {avg_loss:7.3f}',
                                            f',  ' if val_loss is not None else ' ',
                                            f'Val Loss: {val_loss:7.3f}' if val_loss is not None else ' ')
                # 一些 資訊輸出
                self.logger.info(log)
                self.insert_text(log, t_fmt=False)
                self.mv_cursor()

                # 更新 圖表
                self.pws[self.current_page_id].clear()                                                  # 清除 Plot
                self.pws[self.current_page_id].plot(self.t2_var["avg_epoch"], self.t2_var["avg_loss"], pen=pg.mkPen(color='r', width=2), name="average loss")
                if val_loss is not None: 
                    self.pws[self.current_page_id].plot(self.t2_var["val_epoch"], self.t2_var["val_loss"], pen=pg.mkPen(color='b', width=2), name="validation loss")
                self.update_progress(self.current_page_id, cur_epoch, max_epoch)
            else:
                self.insert_text(data['INFO'], t_fmt=False)
        else:
            self.train_finish()
            self.worker.quit()
    
    """ 當訓練完成的時候要進行的動作 """
    def train_finish(self):
        info = "Training Model ... Finished !"
        self.logger.info(info)
        self.insert_text(info)

        self.eval_event()
        self.ui.t2_bt_train.setEnabled(True)
        self.swith_page_button(True)
        
    """ 驗證的事件 """
    def eval_event(self):
        info = "Start to evaluate model ... "
        self.logger.info(info)
        self.insert_text(info, div=True)
        self.init_plot()
        self.update_eval_conf()

        args = {
            'task': self.itao_env.get_env('TASK'), 
            'spec': self.itao_env.get_env('TRAIN', 'SPECS'),
            'key': self.itao_env.get_env('KEY'),
            'num_gpus': self.itao_env.get_env('NUM_GPUS'),
            'gpu_index':self.gpu_idx
        }
        
        # other case
        if 'yolo' in self.itao_env.get_env('TASK'): args['model']=self.itao_env.get_env('TRAIN', 'OUTPUT_MODEL')

        self.worker_eval = self.eval_cmd(args=args)

        if self.run_t2_option['eval']:    
            self.worker_eval.start()
            self.worker_eval.trigger.connect(self.update_eval_log)
        else:
            self.eval_finish()
    
    """ 更新 eval 的參數 """
    def update_eval_conf(self):
        local_model_list = self.get_trained_model()
        local_target_model = local_model_list[0]  # last model
        for model in local_model_list:
            cur_epoch = os.path.splitext(model)[0].split('_')[-1]
            if cur_epoch.isdigit(): 
                cur_epoch = int(cur_epoch)
                if cur_epoch == int(self.itao_env.get_env('TRAIN', 'EPOCH')):
                    local_target_model = model
        
        target_model = self.itao_env.replace_docker_root(local_target_model)
        self.itao_env.update2('TRAIN', 'OUTPUT_MODEL', target_model)
        self.train_spec.mapping('model_path', f'"{target_model}"')

    """ 即時更新與 epoch 相關的資訊 """
    def update_epoch_event(self):
        task = self.itao_env.get_env('TASK')
        epoch = self.ui.t2_epoch.text()
        
        if 'yolo' in self.itao_env.get_env('TASK'):
            output_model = "{}_{}{}_epoch_{:03}.tlt".format(
                self.itao_env.get_env('TASK').replace('_',""),
                self.itao_env.get_env('BACKBONE'), 
                self.itao_env.get_env('NLAYER'),
                int(epoch)
            )
        elif 'classi' in self.itao_env.get_env('TASK'):
            output_model = "{}_{:03}.tlt".format(
                self.itao_env.get_env('BACKBONE'), 
                int(epoch)
            )

        self.ui.t2_model_name.setText(output_model)

    """ 將 t2 的資訊映射到 self.train_conf 上 """
    def update_train_conf(self):
        
        self.logger.info("Updating self.train_conf ... ")

        # Update train spec to itao_env.json
        self.itao_env.update2('TRAIN', 'EPOCH', self.ui.t2_epoch.text())
        self.itao_env.update2('TRAIN', 'INPUT_SHAPE', self.ui.t2_input_shape.text())
        self.itao_env.update2('TRAIN', 'LR', self.ui.t2_lr.text())
        self.itao_env.update2('TRAIN', 'BATCH_SIZE', self.ui.t2_batch.text())
        self.itao_env.update2('TRAIN', 'CUSTOM', self.ui.t2_c1.text())

        self.update_epoch_event()

        if self.itao_env.get_env('NGC_TASK')=='classification':
            # epoch
            self.train_spec.mapping('n_epochs' , self.itao_env.get_env('TRAIN','EPOCH'))
            
            # input image size
            self.train_spec.mapping('input_image_size', '"{}"'.format(self.itao_env.get_env('TRAIN','INPUT_SHAPE')))

            # batch size
            self.train_spec.mapping('batch_size_per_gpu', self.itao_env.get_env('TRAIN','BATCH_SIZE'))

            # eval model path
            # output_model_dir = os.path.join(self.itao_env.get_env('TRAIN','OUTPUT_DIR'), 'weights')
            # output_model_path = os.path.join( output_model_dir, f"{self.itao_env.get_env('BACKBONE')}_{int(self.itao_env.get_env('TRAIN','EPOCH')):03}.tlt")
            # self.itao_env.update2('TRAIN', 'LOCAL_OUTPUT_MODEL', self.itao_env.replace_docker_root(output_model_path, mode='root'))
            # self.itao_env.update2('TRAIN', 'OUTPUT_MODEL', output_model_path)
            # self.train_spec.mapping('model_path', f'"{output_model_path}"')
        
        elif self.itao_env.get_env('TASK')=='yolo_v4':
            # epoch
            self.train_spec.mapping('num_epochs', self.itao_env.get_env('TRAIN','EPOCH'))

            # data augmentation's shape
            c, w, h = [ int(x) for x in self.itao_env.get_env('TRAIN','INPUT_SHAPE').split(',')]
            self.logger.debug('Get shape: {}, {}, {}'.format(c, w, h))
            self.train_spec.mapping('output_width', w)
            self.train_spec.mapping('output_height', h)
            self.train_spec.mapping('output_channel', c)
            
            # batch size
            self.train_spec.mapping('batch_size_per_gpu', self.itao_env.get_env('TRAIN','BATCH_SIZE'))
        
            # eval model path
            # output_model_dir = os.path.join(self.itao_env.get_env('TRAIN', 'OUTPUT_DIR'), 'weights')
            # output_model_path = os.path.join( output_model_dir, "{task}_{backbone}{nlayers}_epoch_{epoch:03}.tlt".format(
            #     task = self.itao_env.get_env('TASK').lower().replace('_', ''),
            #     backbone = self.itao_env.get_env('BACKBONE'),
            #     nlayers = self.itao_env.get_env('NLAYER'),
            #     epoch = int(self.itao_env.get_env('TRAIN','EPOCH'))
            # ))
            # self.itao_env.update2('TRAIN', 'LOCAL_OUTPUT_MODEL', self.itao_env.replace_docker_root(output_model_path, mode='root'))
            # self.itao_env.update2('TRAIN', 'OUTPUT_MODEL', output_model_path)