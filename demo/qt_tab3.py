#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys, os
from typing import *
import pyqtgraph as pg

""" 自己的函式庫自己撈"""
sys.path.append("../itao")
from demo.qt_init import Init

class Tab3(Init):
    def __init__(self):
        super().__init__()

        self.worker_prune, self.worker_retrain = None, None
        self.ui.t3_bt_pruned.clicked.connect(self.prune_event)
        self.ui.t3_bt_retrain.clicked.connect(self.retrain_event)
        self.ui.t3_bt_stop.clicked.connect(self.stop_event)
        self.ui.t3_retrain_epoch.textChanged.connect(self.update_t3_epoch_event)

        self.t3_var = { "avg_epoch":[],
                        "avg_loss":[],
                        "val_epoch":[],
                        "val_loss":[] }

        self.prune_log_key = [  "['nvcr.io']",
                                "Exploring graph for retainable indices",
                                "Pruning model and appending pruned nodes to new graph",
                                "Pruning ratio (pruned model / original model)",
                                "Stop container"]

        # if self.debug:
        #     self.t3_debug = True
        #     if int(self.debug_page)==3:
        #         self.t3_debug = False
        self.run_t3_option = {
            'prune':True,
            'retrain':True,
            'eval':True
        }
        if self.debug:
            for key in self.run_t3_option.keys():
                self.run_t3_option[key]=True if int(self.debug_page)==3 and key==self.debug_opt else False

    """ 按下停止的事件 """
    def stop_event(self):

        info = 'Stoping TAO container ... '
        self.logger.info(info)
        self.insert_text(info, endsym='')

        self.stop_tao.start()
        self.stop_tao.trigger.connect(self.tao_stop_event)

        # 如果 worker_prune 有在進行的話，就強置中斷
        if self.worker_prune != None:
            self.worker_prune.terminate()
            self.worker_prune = None
            self.ui.t3_bt_pruned.setEnabled(True)
        
        # 如果 worker_retrain 有在進行的話，就強置中斷
        if self.worker_retrain != None:
            self.worker_retrain.terminate()
            self.worker_retrain = None
            self.ui.t3_bt_retrain.setEnabled(True)
        
        # 都停止後將 page_button 開啟
        self.swith_page_button(True)
        self.ui.t3_bt_stop.setEnabled(False)


    """ Prune 的事件 """
    def prune_event(self):
        
        info = "Pruning model ... "
        self.logger.info(info)
        self.insert_text(info, div=True)

        self.swith_page_button(False)
        self.ui.t3_bt_pruned.setEnabled(False)
        self.ui.t3_bt_stop.setEnabled(True)
        self.ui.t3_group_retrain_spec.setEnabled(True)
        self.ui.t3_group_retrain_option.setEnabled(True)

        self.init_plot(xlabel="ID", ylabel="MB")
        self.init_console()
        self.update_prune_conf()

        cmd_args = {
            'task': self.itao_env.get_env('TASK'), 
            'input_model': self.itao_env.get_env('PRUNE', 'INPUT_MODEL'),
            'output_model': self.itao_env.get_env('PRUNE', 'OUTPUT_MODEL'),
            'key': self.itao_env.get_env('PRUNE', 'KEY'),
            'pth' : self.itao_env.get_env('PRUNE', 'THRES'), 
            'eq' : self.itao_env.get_env('PRUNE', 'EQ'),
            'num_gpus': self.itao_env.get_env('NUM_GPUS'),
            'gpu_index':self.gpu_idx
        }

        cmd_args['spec'] = self.itao_env.get_env('TRAIN', 'SPECS')

        self.worker_prune = self.prune_cmd( args = cmd_args )
   
        if self.run_t3_option['prune']:
            self.worker_prune.start()
            self.worker_prune.trigger.connect(self.update_prune_log)
        else:
            self.pruned_compare()

    """ 更新 Prune 的資訊 """
    def update_prune_log(self, data):
        
        # 顯示相關資訊
        self.logger.debug(f"{data}\n")
        self.consoles[self.current_page_id].insertPlainText(f"{data}\n")
        self.mv_cursor()

        # 確認進度
        for key in self.prune_log_key:
            if key in data:
                self.update_progress(self.current_page_id, self.prune_log_key.index(key)+1, len(self.prune_log_key))    

        # 當完成的時候
        if data=='end':
            self.worker_prune.quit()
            self.pruned_compare()

    """ 當 prune 完成之後 """
    def prune_finish(self):
        self.ui.t3_bt_pruned.setEnabled(True)  
        self.ui.t3_bt_retrain.setEnabled(True)
        self.ui.t3_bt_stop.setEnabled(False)
        self.swith_page_button(1,0)
        self.consoles[self.current_page_id].insertPlainText("Pruning Model ... Done !\n")
    
    """ 剪枝後計算模型大小以及顯示直方圖 """
    def pruned_compare(self):
        self.logger.info("Comparing Model ... ")
        
        # Get each name of input and output model
        input_model = self.itao_env.replace_docker_root(self.itao_env.get_env('PRUNE', 'INPUT_MODEL') , mode='root')
        output_model = self.itao_env.replace_docker_root(self.itao_env.get_env('PRUNE', 'OUTPUT_MODEL'), mode='root')        
        
        # Get each size
        if os.path.exists(input_model) and os.path.exists(output_model): fake_val = False

        org_size = float(os.path.getsize( input_model ))/1024/1024 if not fake_val else 243
        aft_size = float(os.path.getsize( output_model ))/1024/1024 if not fake_val else 91.2

        # Show some info
        self.consoles[self.current_page_id].insertPlainText(f"Unpruned Model Size : {(org_size):.3f} MB\n")
        self.consoles[self.current_page_id].insertPlainText(f"Pruned Model Size : {(aft_size):.3f} MB\n")
        self.insert_text(f"Pruning Rate : {(aft_size/org_size)*100:.3f}%")
        self.mv_cursor()
        
        # Add plot
        self.pws[self.current_page_id].addItem(pg.BarGraphItem(x=[1], height=[org_size], width=0.5, brush='b', name="Unpruned"))
        self.pws[self.current_page_id].addItem(pg.BarGraphItem(x=[2], height=[aft_size], width=0.5, brush='g', name="Pruned"))
        
        # Update status
        self.update_progress(self.current_page_id, len(self.prune_log_key), len(self.prune_log_key))  
        self.prune_finish()  

    def update_t3_epoch_event(self):
        # output_model = self.ui.t3_retrain_out_model.text()
        # self.retrain_conf['epoch'] = self.ui.t3_retrain_epoch.text()
        epoch = self.ui.t3_retrain_epoch.text()
        self.itao_env.update2('PRUNE', 'EPOCH', epoch)
        
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
        self.ui.t3_retrain_out_model.setText(output_model)
        return output_model

    """ add retrain variable into env """
    def update_retrain_info(self):
        # self.env.update()
        pass

    """ Retrain Event"""
    def retrain_event(self):

        self.init_console()
        self.update_retrain_conf()
        self.init_plot(clean=True)

        info = "Start retraining ... "
        self.logger.info(info)
        self.insert_text(info, div=True)
        
        self.swith_page_button(False)
        self.ui.t3_bt_pruned.setEnabled(False)
        self.ui.t3_bt_retrain.setEnabled(False)
        self.ui.t3_bt_stop.setEnabled(True)
        
        cmd_args = {
            'task': self.itao_env.get_env('TASK'), 
            'key': self.itao_env.get_env('KEY'),
            'spec': self.itao_env.get_env('RETRAIN','SPECS'),
            'output_dir': self.itao_env.get_env('RETRAIN', 'OUTPUT_DIR'), 
            'num_gpus': self.itao_env.get_env('NUM_GPUS'),
            'gpu_index':self.gpu_idx
        }

        self.worker_retrain = self.retrain_cmd( args = cmd_args )

        if self.itao_env.get_env('RETRAIN', 'EPOCH').isdigit:
            self.update_progress(self.current_page_id, 0, int(self.itao_env.get_env('RETRAIN', 'EPOCH')))
        else:
            self.logger.error('Value Error: retrain_conf["epoch"] -> {}'.format(self.itao_env.get_env('RETRAIN', 'EPOCH')))

        
        if self.run_t3_option['retrain']:
            self.worker_retrain.start()
            self.worker_retrain.trigger.connect(self.update_retrain_log)
        else:
            self.retrain_finish()

    """ 更新 Retrain 的相關資訊 """
    def update_retrain_log(self, data):
        if bool(data):

            if len(data.keys())>1:
                cur_epoch, avg_loss, val_loss, max_epoch = data['epoch'], data['avg_loss'], data['val_loss'], int(self.itao_env.get_env('RETRAIN', 'EPOCH'))
                log = "{} {} {}\n".format(  f'[{cur_epoch:03}/{max_epoch:03}]',
                                            f'AVG_LOSS: {avg_loss:7.3f}',
                                            f'VAL_LOSS: {val_loss:7.3f}' if val_loss is not None else ' ')

                self.t3_var["val_epoch"].append(cur_epoch)
                self.t3_var["val_loss"].append(val_loss)        
                self.t3_var["avg_epoch"].append(cur_epoch)
                self.t3_var["avg_loss"].append(avg_loss)

                self.logger.info(log)
                self.insert_text(log, t_fmt=False)
                self.mv_cursor()

                self.pws[self.current_page_id].clear()                                                  # 清除 Plot
                self.pws[self.current_page_id].plot(self.t3_var["avg_epoch"], self.t3_var["avg_loss"], pen=pg.mkPen('r', width=2), name="average loss")
                if val_loss is not None: 
                    self.pws[self.current_page_id].plot(self.t3_var["val_epoch"], self.t3_var["val_loss"], pen=pg.mkPen('b', width=2), name="validation loss")
                    
                self.update_progress(self.current_page_id, cur_epoch, max_epoch)            
            else:
                self.insert_text(data['INFO'], t_fmt=False)
        else:
            self.retrain_finish()
            self.worker_retrain.quit()
    
    """ 當 retrain 完成 """
    def retrain_finish(self):
        
        info = "Re-Train Model ... Done !\n"
        self.logger.info(info)
        self.consoles[self.current_page_id].insertPlainText(info)

        self.ui.t3_bt_retrain.setEnabled(True)  
        self.ui.t3_bt_pruned.setEnabled(True)
        self.ui.t3_bt_stop.setEnabled(False)
        self.swith_page_button(1)

    """ 將QT中的PRUNE配置內容映射到PRUNE_CONF """
    def update_prune_conf(self):
        
        self.logger.info("Updating PRUNE_CONF ... ")

        # create prune folder
        backbone = self.itao_env.get_env('BACKBONE')
        nlayer = self.itao_env.get_env('NLAYER')
        local_prune_dir = os.path.join(self.itao_env.get_env('TRAIN', 'LOCAL_OUTPUT_DIR'), f"{backbone}{nlayer}_pruned")

        if not os.path.exists(local_prune_dir):
            print('Create directory of pruned model {}'.format(local_prune_dir))
            os.mkdir(local_prune_dir)
        
        # setup path of prune_model
        prune_dir = self.itao_env.replace_docker_root(local_prune_dir)
        prune_model_name = f'{backbone}{nlayer}_pruned.tlt' # if self.ui.t3_pruned_out_name.text() == "" else self.ui.t3_pruned_out_name.text()
        # update information on itao
        # self.ui.t3_pruned_out_name.setText(prune_model_name)
        self.ui.t3_retrain_pretrain.setText(prune_model_name)
        
        # setup path of output_model and input model  
        prune_model = os.path.join(prune_dir, prune_model_name )
        self.prune_conf['output_model']=prune_model
        self.itao_env.update2('PRUNE', 'OUTPUT_MODEL', prune_model)
        self.itao_env.update2('PRUNE', 'LOCAL_OUTPUT_MODEL', self.itao_env.replace_docker_root(prune_model, mode='root'))
        self.itao_env.update2('PRUNE', 'INPUT_MODEL', self.itao_env.get_env('TRAIN', 'OUTPUT_MODEL'))
        
        # setup key, thres, eq
        self.itao_env.update2('PRUNE', 'KEY', self.itao_env.get_env('TRAIN', 'KEY'))
        # self.ui.t3_pruned_key.setText(self.itao_env.get_env('PRUNE', 'KEY'))
        self.itao_env.update2('PRUNE', 'THRES', round(float(self.ui.t3_pruned_threshold.value()), 2))
        
        if 'yolo' in self.itao_env.get_env('TASK'):
            eq = 'intersection'   
        else: 
            eq = 'union'

        self.prune_conf['eq'] = eq
        self.itao_env.update2('PRUNE','EQ', eq)

        # show conf
        self.insert_text("Show pruned config", config=self.itao_env.get_env('PRUNE'))

    """ 將QT中的RETRAIN配置內容映射到RETRAIN_CONF """
    def update_retrain_conf(self):

        self.logger.info('Updating self.retrain_conf ... ')
        
        # 從 GUI 中取得參數
        input_model = self.itao_env.get_env('PRUNE', 'OUTPUT_MODEL')
        backbone = self.itao_env.get_env('BACKBONE').lower()
        nlayer = self.itao_env.get_env('NLAYER')
        epoch = self.ui.t3_retrain_epoch.text()
        batch_size = self.ui.t3_retrain_bsize.text()
        lr = self.ui.t3_retrain_lr.text()

        # 更新到 env 方便 debug以及取用
        self.ui.t3_retrain_key.setText(self.itao_env.get_env('KEY'))
        self.itao_env.update2('RETRAIN', 'INPUT_MODEL', input_model)
        self.itao_env.update2('RETRAIN', 'EPOCH', epoch)
        self.itao_env.update2('RETRAIN', 'BATCH_SIZE', batch_size)
        self.itao_env.update2('RETRAIN', 'LR', lr)
        
        # 更新
        output_model = self.update_t3_epoch_event()
        
        output_model_dir = os.path.join(self.itao_env.get_env('USER_EXPERIMENT_DIR'), 'output_retrain')
        output_weight_dir = os.path.join(output_model_dir, 'weights')
        
        local_weight_model_dir = self.itao_env.replace_docker_root(output_weight_dir,mode='root')
        if not os.path.exists(local_weight_model_dir): os.makedirs(local_weight_model_dir)

        output_model_path = os.path.join(output_weight_dir, output_model )
        self.itao_env.update2('RETRAIN', 'OUTPUT_DIR', output_model_dir)
        self.itao_env.update2('RETRAIN', 'OUTPUT_MODEL', output_model_path)
        
        if 'classification' in self.itao_env.get_env('TASK'):
            self.retrain_spec.mapping('arch', '"{}"'.format(backbone))            
            self.retrain_spec.mapping('batch_size_per_gpu', int(batch_size))

            self.retrain_spec.mapping('n_layer', nlayer)
            self.retrain_spec.mapping('n_epochs', epoch)
            self.retrain_spec.mapping('input_image_size', '"{}"'.format(self.train_spec.find_key('input_image_size')))

            self.retrain_spec.mapping('train_dataset_path', '"{}"'.format(self.train_spec.find_key('train_dataset_path')))
            self.retrain_spec.mapping('val_dataset_path', '"{}"'.format( self.train_spec.find_key('val_dataset_path')))

            self.retrain_spec.mapping('pretrained_model_path', '"{}"'.format(input_model))
            self.retrain_spec.mapping('eval_dataset_path', '"{}"'.format( self.train_spec.find_key('eval_dataset_path')))

            self.retrain_spec.mapping('model_path', f'"{output_model_path}"')
            self.retrain_spec.mapping('pretrained_model_path', '"{}"'.format(input_model))

        elif 'yolo' in self.itao_env.get_env('TASK'):

            # anchor
            self.retrain_spec.mapping('small_anchor_shape', f'"{self.train_spec.find_key("small_anchor_shape")}"' )
            self.retrain_spec.mapping('mid_anchor_shape', f'"{self.train_spec.find_key("mid_anchor_shape")}"' )
            self.retrain_spec.mapping('big_anchor_shape', f'"{self.train_spec.find_key("big_anchor_shape")}"' )

            # epoch, arch, nlayers, batch_size_per_gpu
            self.retrain_spec.mapping('num_epochs', epoch)
            self.retrain_spec.mapping('arch', '"{}"'.format(backbone))            
            self.retrain_spec.mapping('nlayers', nlayer)
            self.retrain_spec.mapping('batch_size_per_gpu', int(batch_size))

            # data augmentation's shape
            c, w, h = [ int(x) for x in self.itao_env.get_env('TRAIN','INPUT_SHAPE').split(',')]
            self.retrain_spec.mapping('output_width', w)
            self.retrain_spec.mapping('output_height', h)
            self.retrain_spec.mapping('output_channel', c)

            # set dataset's label
            self.retrain_spec.set_label_for_detection(key='target_class_mapping')

            # dataset
            self.retrain_spec.mapping('image_directory_path', '"{}"'.format( self.train_spec.find_key('image_directory_path') ))
            self.retrain_spec.mapping('label_directory_path', '"{}"'.format( self.train_spec.find_key('label_directory_path') ))
                
            # model path
            self.retrain_spec.mapping('pruned_model_path', '"{}"'.format(input_model))

        self.insert_text("Show Retrain Conifg", config=self.itao_env.get_env('RETRAIN'))

            