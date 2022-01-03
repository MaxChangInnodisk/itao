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

        if self.debug:
            self.t3_debug = True
            if int(self.debug_page)==3:
                self.t3_debug = False


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
            'input_model': self.itao_env.get_env('UNPRUNED_MODEL'),
            'output_model': self.itao_env.get_env('PRUNED_MODEL'),
            'key': self.itao_env.get_env('KEY'),
            'pth' : self.itao_env.get_env('PRUNED_THRES'), 
            'eq' : self.itao_env.get_env('PRUNED_EQ')
        }

        cmd_args['spec'] = self.itao_env.replace_docker_root(self.train_spec.get_spec_path(), mode='docker')

        self.worker_prune = self.prune_cmd( args = cmd_args )
   
        if not self.t3_debug:
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
        input_model = self.itao_env.replace_docker_root(self.prune_conf['input_model'], mode='root')
        output_model = self.itao_env.replace_docker_root(self.prune_conf['output_model'], mode='root')        
        
        # Get each size
        org_size = float(os.path.getsize( input_model ))/1024/1024 if not self.t3_debug else 243
        aft_size = float(os.path.getsize( output_model ))/1024/1024 if not self.t3_debug else 91.2
        
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
        # output_model = self.ui.t3_retrain_out_model.toPlainText()
        self.retrain_conf['epoch'] = self.ui.t3_retrain_epoch.toPlainText()
        output_model = "{}{}_{:03}.tlt".format(self.train_conf['backbone'].lower(), int(self.retrain_conf['epoch']))
        self.ui.t3_retrain_out_model.setPlainText(output_model)

    """ add retrain variable into env """
    def update_retrain_info(self):
        # self.env.update()
        pass

    """ Retrain Event"""
    def retrain_event(self):
        
        info = "Start retraining ... "
        self.logger.info(info)
        self.insert_text(info, div=True)
        
        self.swith_page_button(False)
        self.ui.t3_bt_pruned.setEnabled(False)
        self.ui.t3_bt_retrain.setEnabled(False)
        self.ui.t3_bt_stop.setEnabled(True)
        
        self.init_console()
        self.update_retrain_conf()
        self.init_plot(clean=True)
        
        self.worker_retrain = self.retrain_cmd(
            task= self.itao_env.get_env('TASK'), 
            spec= self.itao_env.replace_docker_root(self.retrain_spec.get_spec_path()), 
            output_dir= self.itao_env.get_env('RETRAIN_OUTPUT_DIR'), 
            key= self.itao_env.get_env('KEY'),
            num_gpus= self.itao_env.get_env('NUM_GPUS')
        )

        if self.retrain_conf['epoch'].isdigit:
            self.update_progress(self.current_page_id, 0, self.retrain_conf['epoch'])
        else:
            self.logger.error('Value Error: retrain_conf["epoch"] -> {}'.format(self.retrain_conf['epoch']))

        
        if not self.t3_debug:
            self.worker_retrain.start()
            self.worker_retrain.trigger.connect(self.update_retrain_log)
        else:
            self.retrain_finish()

    """ 更新 Retrain 的相關資訊 """
    def update_retrain_log(self, data):
        if bool(data):
            cur_epoch, avg_loss, val_loss, max_epoch = data['epoch'], data['avg_loss'], data['val_loss'], int(self.retrain_conf['epoch'])
            log = "{} {} {}\n".format(  f'[{cur_epoch:03}/{max_epoch:03}]',
                                        f'AVG_LOSS: {avg_loss:06.3f}',
                                        f'VAL_LOSS: {val_loss:06.3f}' if val_loss is not None else ' ')

            self.t3_var["val_epoch"].append(cur_epoch)
            self.t3_var["val_loss"].append(val_loss)        
            self.t3_var["avg_epoch"].append(cur_epoch)
            self.t3_var["avg_loss"].append(avg_loss)

            self.pws[self.current_page_id].clear()                                                  # 清除 Plot
            self.consoles[self.current_page_id].insertPlainText(log)                                # 插入內容
            self.mv_cursor()

            self.pws[self.current_page_id].plot(self.t3_var["avg_epoch"], self.t3_var["avg_loss"], pen=pg.mkPen('r', width=2), name="average loss")
            self.pws[self.current_page_id].plot(self.t3_var["val_epoch"], self.t3_var["val_loss"], pen=pg.mkPen('b', width=2), name="validation loss")
            self.update_progress(self.current_page_id, cur_epoch, max_epoch)            

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
        backbone = self.itao_env.get_env('backbone')
        nlayer = self.itao_env.get_env('nlayer')

        local_prune_dir = os.path.join(self.itao_env.get_env('LOCAL_OUTPUT_DIR'), f"{backbone}{nlayer}_pruned")
        if not os.path.exists(local_prune_dir):
            print('Create directory of pruned model {}'.format(local_prune_dir))
            os.mkdir(local_prune_dir)
        # self.prune_conf['local_prune_dir']=local_prune_dir
        
        # setup path of prune_model
        prune_dir = self.itao_env.replace_docker_root(local_prune_dir)
        
        # check output name is set.
        prune_model_name = f'{backbone}{nlayer}_pruned.tlt' if self.ui.t3_pruned_out_name.toPlainText() == "" else self.ui.t3_pruned_out_name.toPlainText()
        # update information on itao
        self.ui.t3_pruned_out_name.setPlainText(prune_model_name)
        self.ui.t3_retrain_pretrain.setPlainText(prune_model_name)
        
        # setup path of input model and output_model
        prune_model = os.path.join(prune_dir, prune_model_name )
        self.prune_conf['output_model']=prune_model
        self.itao_env.update('PRUNED_MODEL', prune_model)
        self.itao_env.update('LOCAL_PRUNED_MODEL', self.itao_env.replace_docker_root(prune_model, mode='root'))
        
        input_model = os.path.join(self.itao_env.get_env('OUTPUT_DIR'), os.path.join('weights', self.ui.t3_pruned_in_model.toPlainText()))
        self.prune_conf['input_model']=self.itao_env.get_env('UNPRUNED_MODEL')
        # = self.itao_env.get_env('UNPRUNED_MODEL')
        
        # setup key, thres, eq
        self.retrain_conf['key'] = self.prune_conf['key'] = self.train_conf['key']
        self.ui.t3_pruned_key.setText(self.train_conf['key'])

        self.prune_conf['thres'] = round(float(self.ui.t3_pruned_threshold.value()), 2)
        self.itao_env.update('PRUNED_THRES', round(float(self.ui.t3_pruned_threshold.value()), 2))
        
        if 'yolo' in self.itao_env.get_env('TASK'):
            eq = 'intersection'   
        else: 
            eq = 'union'

        self.prune_conf['eq'] = eq
        self.itao_env.update('PRUNED_EQ', eq)

        # show conf
        self.insert_text("Show pruned config", config=self.prune_conf)

    """ 將QT中的RETRAIN配置內容映射到RETRAIN_CONF """
    def update_retrain_conf(self):

        self.logger.info('Updating self.retrain_conf ... ')

        self.ui.t3_retrain_key.setText(self.train_conf['key'])
        self.retrain_conf['backbone'] = self.train_conf['backbone']
        self.retrain_conf['pretrain'] = self.prune_conf['output_model']
        self.retrain_conf['epoch'] = self.ui.t3_retrain_epoch.toPlainText()
        self.retrain_conf['batch_size'] = self.ui.t3_retrain_bsize.toPlainText()
        self.retrain_conf['learning_rate'] = self.ui.t3_retrain_lr.toPlainText()

        self.retrain_spec.mapping('n_epochs', self.retrain_conf['epoch'])
        self.retrain_spec.mapping('input_image_size', '"{}"'.format(self.train_spec.find_key('input_image_size')))
        # self.retrain_spec.mapping('input_image_size', '"3,224,224"')
        self.retrain_spec.mapping('arch', '"{}"'.format(self.train_conf['backbone'].lower()))
        self.retrain_spec.mapping('n_layer', self.train_conf['nlayer'])
        
        self.retrain_spec.mapping('train_dataset_path', '"{}"'.format(self.train_spec.find_key('train_dataset_path')))
        self.retrain_spec.mapping('val_dataset_path', '"{}"'.format( self.train_spec.find_key('val_dataset_path')))
        
        self.retrain_spec.mapping('batch_size_per_gpu', int(self.ui.t3_retrain_bsize.toPlainText()))
        self.retrain_spec.mapping('pretrained_model_path', '"{}"'.format(self.retrain_conf['pretrain']))
        self.retrain_spec.mapping('eval_dataset_path', '"{}"'.format( self.train_spec.find_key('eval_dataset_path')))
        
        self.update_t3_epoch_event()
        
        if 'classification' in self.itao_env.get_env('TASK'):

            output_model = self.ui.t3_retrain_out_model.toPlainText()
            output_model = "{}_{:03}.tlt".format(self.retrain_spec.find_key('arch').replace('"', "").replace(" ", ""), int(self.retrain_spec.find_key('n_epochs').rstrip()))

            self.retrain_conf['output_model'] = output_model
            self.ui.t3_retrain_out_model.setPlainText(self.retrain_conf['output_model'])
            
            retrain_model_dir = os.path.join(self.itao_env.get_env('USER_EXPERIMENT_DIR'), 'output_retrain')
            self.itao_env.update('RETRAIN_OUTPUT_DIR', retrain_model_dir)

            _retrain_model_path = os.path.join(retrain_model_dir, 'weights')
            retrain_model_path = os.path.join(_retrain_model_path, self.retrain_conf['output_model'] )
            self.retrain_spec.mapping('model_path', f'"{retrain_model_path}"')

            self.retrain_spec.mapping('pretrained_model_path', '"{}"'.format(self.retrain_conf['pretrain']))
        elif 'yolo' in self.itao_env.get_env('TASK'):
            
            pass


        self.insert_text("Show Retrain Conifg", config=self.retrain_conf)

            