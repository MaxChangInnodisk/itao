import json, os, shutil, datetime
import glob, time

try:
    from itao.utils.qt_logger import CustomLogger
except:
    from qt_logger import CustomLogger

if __name__ == '__main__':
    import environ
else:
    from itao import environ

"""
Version 0.2:

Provide fixed specs

./classification/
└── specs
    ├── classification_retrain_spec.cfg
    └── classification_spec.cfg

./yolo_v4/
└── specs
    ├── yolo_v4_kitti_seq_spec.txt
    └── yolo_v4_retrain_kitti_seq_spec.txt

"""

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

        if self.spec_path == None:
            self.spec_path = self.get_spec_path()
                    
    """ backup target spec file """
    def backup_spec(self):
        print("Backup Spec ... ", end="")

        if self.spec_path == None: 
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
    def spec_to_list(self):
        if self.spec_path == None: 
            self.spec_path = self.get_spec_path()
            
        if self.spec_path != None:
            cnt = []
            with open(self.spec_path, 'r') as spec:
                cnt = [ line for line in spec.readlines() ]
            return cnt
        else:
            return None

    """ write `new_cnt` into spec file """    
    def list_to_spec(self, new_cnt):
        if self.spec_path == None: 
            self.spec_path = self.get_spec_path()

        with open(self.spec_path, 'w') as spec:
            for c in new_cnt:
                spec.write(c)    

    """ check is include format in path and exclude not in path """
    def check_fmt(self, spec_path):
        parts = os.path.splitext(spec_path)[0].split('_')
        for part in parts:
            if part.lower() == self.mode.lower():
                return 1
        return 0 

    """ get train or retrain spec """
    def get_spec_path(self):     
        self.logger.info('Get specification ... ')   
        for spec in os.listdir(self.env.get_env('LOCAL_SPECS_DIR')):
            if self.check_fmt(spec):
                spec_path = os.path.join( self.env.get_env('LOCAL_SPECS_DIR'), spec ) 
                if self.mode=='train':   
                    self.env.update2('TRAIN', 'LOCAL_SPECS', spec_path)
                    self.env.update2('TRAIN', 'SPECS', self.env.replace_docker_root(spec_path))
                elif self.mode=='retrain':   
                    self.env.update2('RETRAIN', 'LOCAL_SPECS', spec_path)
                    self.env.update2('RETRAIN', 'SPECS', self.env.replace_docker_root(spec_path))
                return spec_path
        self.logger.error('Can not find spec file!!!', spec)
        return None

    """ find value of `key` in spec """
    def find_key(self, key, rm_space=True):

        if self.spec_path == None: 
            self.spec_path = self.get_spec_path()
            
        with open(self.spec_path, 'r') as spec:
            for line in spec.readlines():
                if key in line: 
                    if key=='model_path' and 'pretrained_model_path' in line:
                        continue
                    trg = line.split(':')[1].rstrip("\n").replace('"','')
                    if rm_space:
                        trg = trg.replace(" ", "")
                    return trg
        return None

    """ find value of `key` in spec """
    def find_key_with_scope(self, scope_key , key, rm_space=True):

        if self.spec_path == None: 
            self.spec_path = self.get_spec_path()

        scope_range = self.get_scope(self.spec_cnt, scope_key)
        cur_start, cur_end = map(lambda x:int(x), scope_range[0].split(':'))

        self.spec_cnt = self.spec_to_list()

        for idx, cnt in enumerate(self.spec_cnt):
            if ':' in cnt:
                if key in cnt:
                    org_key, org_val = self.spec_cnt[idx].split(":")
                    if key == org_key.replace(" ", "") and (idx>=cur_start and idx<=cur_end):
                        self.logger.debug('Find {}:{}'.format(key, org_val))
                        val = org_val.rstrip("\n").replace('"','')
                        if rm_space:
                            val = val.replace(" ", "")
                        return val

        return None

    """ mapping val of key """
    def mapping(self, key, val=""):
        self.spec_cnt = self.spec_to_list()
        found_key = False
        for idx, cnt in enumerate(self.spec_cnt):
            if ':' in cnt:
                if key in cnt:
                    org_key, org_val = self.spec_cnt[idx].split(":")
                    if key == org_key.replace(" ", ""):
                        self.logger.info('Upd spec: {} -> {}'.format(org_key, val))
                        self.spec_cnt[idx] = f"{org_key}: {val}\n"
                        found_key=True
                    else:
                        self.logger.warning('Original key: {}, Mapping key: {}'.format(org_key, key))
                        continue
                
        if found_key:
            self.list_to_spec(self.spec_cnt)
        else:
            self.logger.warning('Not found any key ({}) in spec ({}).'.format(key, self.mode))

    """ mapping val of key """
    def mapping_with_scope(self, scope_key, key, val=""):
        self.spec_cnt = self.spec_to_list()

        scope_range = self.get_scope(self.spec_cnt, scope_key)
        cur_start, cur_end = map(lambda x:int(x), scope_range[0].split(':'))

        found_key = False
        for idx, cnt in enumerate(self.spec_cnt):
            if ':' in cnt:
                if key in cnt:
                    org_key, org_val = self.spec_cnt[idx].split(":")
                    if key == org_key.replace(" ", "") and (idx>=cur_start and idx<=cur_end):
                        self.logger.info('Upd spec: {} -> {}'.format(org_key, val))
                        self.spec_cnt[idx] = f"{org_key}: {val}\n"
                        found_key=True
                    else:
                        self.logger.warning('Original key: {}, Mapping key: {}'.format(org_key, key))
                        continue
                
        if found_key:
            self.list_to_spec(self.spec_cnt)
        else:
            self.logger.warning('Not found any key ({}) in spec ({}).'.format(key, self.mode))

    
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
        self.spec_cnt = self.spec_to_list()
        
        # get labels and seting format (args)
        train_data_path = os.path.join(self.env.get_env('LOCAL_DATASET'), 'train')
        labels = self.get_label_list(label_dir=os.path.join(train_data_path, 'labels'))

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
            for scope in scopes[len(args):-1]:
                start, end = map(lambda x:int(x), scope.split(':'))
                for i in range(start, end+1):
                    self.spec_cnt.pop(i-pop_nums)
                    pop_nums += 1
                self.logger.debug('Remove scope -> id: {}, range: {}'.format(len(scopes)-len(args)+cur_scope, scope.split(':')))
                cur_scope +=1

        self.logger.debug('Done')
        
        # overwrite spec
        self.list_to_spec(self.spec_cnt)
        
        return 1
    
    def del_spec_item(self,scope=None, key=None):
        
        self.logger.info('Del key: [{}][{}]'.format(scope, key))
        
        self.spec_cnt = self.spec_to_list()
        
        # find first scope
        idx_start, idx_end = tuple([int(x) for x in self.get_scope(self.spec_cnt, scope)[0].split(':')]) if scope != None else (0, len(self.spec_cnt)-1)
            
        # delete spec item and update
        for idx, cnt in enumerate(self.spec_cnt):

            if ':' in cnt:
                org_key = cnt.split(':')[0].replace(" ", "")
                if key == org_key and idx in [i for i in range(idx_start, idx_end+1)]:
                    self.spec_cnt.pop( idx )
        # overwrite
        self.list_to_spec(self.spec_cnt)

    def add_spec_item(self, scope, key, val, level=1):
        self.logger.info('Add key: [{}][{}]={}'.format(scope, key, val))
        # load spec
        self.spec_cnt = self.spec_to_list()

        # find first scope
        idx_start, idx_end = tuple([int(x) for x in self.get_scope(self.spec_cnt, scope)[0].split(':')]) if scope != None else (0, len(self.spec_cnt)-1)
        
        # add spec item
        space = ' '*2*(level-1)
        self.spec_cnt.insert(idx_end, '{}{}: {}\n'.format(space, key, val))

        self.list_to_spec(self.spec_cnt)


        