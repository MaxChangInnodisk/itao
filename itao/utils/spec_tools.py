import json, os, shutil, datetime
import glob, time
from itao.utils.qt_logger import CustomLogger

if __name__ == '__main__':
    import environ
    import configs
else:
    from itao import environ

# MAP_KEY = {
#     "train":[
#         "arch",
#         "n_layer",
#         "input_image_size",
#         "train_dataset_path",
#         "val_dataset_path",
#         "pretrained_model_path",
#         "batch_size_per_gpu",
#         "n_epochs",
#         "eval_dataset_path",
#         "model_path",
#     ],
#     "retrain":[
#         "arch",
#         "n_layer",
#         "input_image_size",
#         "train_dataset_path",
#         "val_dataset_path",
#         "pretrained_model_path",
#         "batch_size_per_gpu",
#         "n_epochs",
#         "eval_dataset_path",
#         "model_path",
#     ]
# }

class DefineSpec():
    """ initialize """
    def __init__(self, mode='train'):
        self.mode = mode # or 'retrain'
        self.env = environ.SetupEnv()
        self.logger = CustomLogger().get_logger('dev')
        self.include_fmt, self.exclude_fmt = [], []
        self.spec_path = None
        self.spec_cnt, self.new_sepc_cnt = [],[]
        self.write_spec = False

        self.define_include_fmt()
        self.define_exclude_fmt()
        # self.keys = MAP_KEY[self.mode]
        
        # Get path of spec and backup.
        # self.spec_path = self.get_spec_path()        
        
    def define_include_fmt(self):
        if 'yolo_v4' in self.env.get_env('TASK'):
            self.include_fmt = ['seq']
        
    def define_exclude_fmt(self):
        if self.mode=='train':
            self.logger.info('Create train spec')
            self.exclude_fmt = ['retrain', 'backup', 'old']
        else:
            self.logger.info('Create retrain spec')
            self.exclude_fmt = ['backup', 'old']
                    
    """ backup target spec file """
    def backup_spec(self):
        print("Backup Spec ... ", end="")
        self.spec_path = self.get_spec_path()
        cur_time = datetime.datetime.now()            
        backup_spec_path = '{}_backup_{}{}'.format(
            os.path.splitext(self.spec_path)[0],
            f'{cur_time.year}{cur_time.month:02}{cur_time.day:02}{cur_time.hour:02}{cur_time.minute:02}',
            os.path.splitext(self.spec_path)[1]
        )
        
        if self.spec_path!=None:    
            shutil.copy2(self.spec_path, backup_spec_path) 
            print(f"Success! \nSave to {backup_spec_path}")
        else:
            print("Failed! Spec is not setup.")

    """ convert spec's content to python dict """
    def spec_to_dict(self):
        self.spec_path = self.get_spec_path()
        if self.spec_path != None:
            cnt = []
            with open(self.spec_path, 'r') as spec:
                cnt = [ line for line in spec.readlines() ]
            return cnt
        else:
            return None

    """ write `new_cnt` into spec file """    
    def dict_to_spec(self, new_cnt):
        self.spec_path = self.get_spec_path()
        with open(self.spec_path, 'w') as spec:
            for c in new_cnt:
                spec.write(c)    

    """ check is include format in path and exclude not in path """
    def check_fmt(self, spec_path):
        
        parts = os.path.splitext(spec_path)[0].split('_')
        for part in parts:
            if part not in self.exclude_fmt:
                if self.include_fmt == []:
                    return 1
                else:
                    if part in self.include_fmt:
                        return 1
        return 0 

    """ get train or retrain spec """
    def get_spec_path(self):        
        for spec in os.listdir(self.env.get_env('LOCAL_SPECS_DIR')):
            if self.check_fmt(spec):
                spec_path = os.path.join( self.env.get_env('LOCAL_SPECS_DIR'), spec ) 
                if self.mode=='train':   
                    self.env.update('LOCAL_SPECS', spec_path, log=0)
                    self.env.update('SPECS', self.env.replace_docker_root(spec_path), log=0)
                elif self.mode=='retrain':   
                    self.env.update('LOCAL_RETRAIN_SPECS', spec_path)
                    self.env.update('RETRAIN_SPECS', self.env.replace_docker_root(spec_path))

                return spec_path

        self.logger.error('Can not find spec file!!!', spec)

    """ find value of `key` in spec """
    def find_key(self, key):
        self.spec_path = self.get_spec_path()
        with open(self.spec_path, 'r') as spec:
            for line in spec.readlines():
                if key in line: 
                    if key=='model_path' and 'pretrained_model_path' in line:
                        continue
                    trg = line.split(':')[1].rstrip("\n").replace('"','').replace(" ", "")
                    # print(line, trg)
                    return trg

    """ mapping val of key """
    def mapping(self, key, val=""):
        self.spec_cnt = self.spec_to_dict()
        for idx, cnt in enumerate(self.spec_cnt):
            if key in cnt:
                org_key, org_val = self.spec_cnt[idx].split(":")

                # make sure key is correct. ( e.g. problem with model_path and pretrained_model_path )
                if key == org_key.replace(" ", ""):
                    self.spec_cnt[idx] = f"{org_key}: {val}\n"
                else:
                    print('Mapping Error')
            else:
                continue

        self.dict_to_spec(self.spec_cnt)

    """ return list of label """
    def get_label_list(self, label_dir:str) -> list:
        self.logger.info('Calculating number of labels ...')
        t_start = time.time()
        # check is path exists

        if not os.path.exists(label_dir):
            self.logger.error('Can not find label directory')
            return

        # find all txt file
        classes = []
        all_label_path = glob.glob( os.path.join(label_dir, '*.txt'))

        # double check
        if len(all_label_path)==0:
            self.logger.error(f'No such .txt file in {label_dir}')
            return

        # figure out all file and search how many labels in this dataset
        for label_path in all_label_path:
            with open(label_path) as label:
                contents = [ line for line in label.readlines() ]
                for content in contents:
                    class_name = content.split(' ')[0]
                    if class_name not in classes:
                        classes.append(class_name)

        # print out some information
        t_end = time.time()
        self.logger.info('Find {} classes ({}s)'.format(len(classes), round(t_end-t_start, 2)))
        self.logger.debug('-'*20)
        [ self.logger.debug('[{}] {}'.format(idx, name)) for idx, name in enumerate(classes) ]
        
        # return list include all classes
        return classes

    """ return scope position """
    def get_scope(self, spec:list, key:str) -> list:

        start, end = '{', '}'
        scp_start, scp_end = 0, 0 
        in_scope = False
        scopes = []
        temp_scope = ''

        for idx, cnt in enumerate(spec):
            if key in cnt and start in cnt:
                scp_start = idx
                in_scope = True
            if in_scope and end in cnt:
                in_scope = False
                scp_end = idx
                
                scp_idx = len(scopes)
                scp_range = '{}:{}'.format(scp_start, scp_end)
                
                print('Find scope -> id: {}, range: {}'.format(scp_idx, scp_range))
                scopes.append(scp_range)

        return scopes    

    """ setup object detection's label in spec """
    def set_label_for_detection(self, key):
        
        # load spec
        self.spec_cnt = self.spec_to_dict()
        
        # get labels and seting format (args)
        labels = self.get_label_list(label_dir=os.path.join(self.env.get_env('LOCAL_DATASET'), 'labels'))
        args = []
        [ args.append({'key':f'"{lbl}"', 'value':f'"{lbl}"'}) for lbl in labels ]
            
        # find scope
        scopes = self.get_scope(self.spec_cnt, key)

        # backup scope
        cur_start, cur_end = map(lambda x:int(x), scopes[0].split(':'))
        backup_scope = self.spec_cnt[cur_start:cur_end+1]

        # replace or add new scope content
        for arg_idx, arg in enumerate(args):

            # if scope nums bigger than arguments than just do replace        
            if len(scopes)>=arg_idx+1:

                for key, val in arg.items():
                    cur_start, cur_end = map(lambda x:int(x), scopes[arg_idx].split(':'))
        
                    for cnt, idx in zip(self.spec_cnt[cur_start:cur_end], range(cur_start, cur_end)): 
                        if key in cnt:
                            org_key = cnt.split(':')[0]
                            self.spec_cnt[idx] = '{key}: {value}\n'.format(key=org_key, value=val) 
                            self.logger.debug('[{:03}] {:20} -> {:20}'.format(idx, cnt.rstrip(), self.spec_cnt[idx]))

            # if scope nums smaller than arguments, we will add new scope into spec which format is base on the first scope
            else:
                # get the correct position
                insert_idx = cur_end+1

                # gain a new backup 
                for key, val in arg.items():
                    for idx, cnt in enumerate(backup_scope):
                        if key in cnt:
                            backup_scope[idx] = '{key}: {value}\n'.format(key=cnt.split(':')[0], value=val) 

                # add backup into spec
                for cnt in backup_scope:
                    self.spec_cnt.insert(insert_idx, cnt)            
                    self.logger.debug('[{:03}] {:20}'.format(insert_idx, self.spec_cnt[insert_idx]))
                    insert_idx+=1

        # remove unused scope
        if len(scopes)>len(args):  
            pop_nums, cur_scope = 0, 0
            for scope in scopes[len(args):]:
                start, end = map(lambda x:int(x), scope.split(':'))
                for i in range(start, end+1):
                    self.spec_cnt.pop(i-pop_nums)
                    pop_nums += 1
                self.logger.debug('Remove scope -> id: {}, range: {}'.format(len(scopes)-len(args)+cur_scope, scope.split(':')))
                cur_scope +=1

        self.logger.debug('Done')
        
        # overwrite spec
        self.dict_to_spec(self.spec_cnt)
        
        return 1