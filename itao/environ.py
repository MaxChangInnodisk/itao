import os, json
from itao.utils.qt_logger import CustomLogger

class SetupEnv:
    """ initialize """
    def __init__(self, env_file='./configs/itao_env.json', debug=True) -> None:
        
        self.env = {}
        self.json_file_path = env_file
        self.debug = debug
        self.cwd = os.getcwd()
        self.env_cfg = os.path.realpath(env_file)
        
        self.mounts_file = os.path.expanduser("~/.tao_mounts.json")
        self.logger = CustomLogger().get_logger('dev')
        # self.get_newest_env()

    """ replace base path """
    def replace_docker_root(self, path, mode='docker'):
        if mode=='docker':
            return path.replace(self.env['LOCAL_PROJECT_DIR'], self.env['PROJECT_DIR']) if self.env['LOCAL_PROJECT_DIR'] in path else path
        if mode=='root':
            return path.replace(self.env['PROJECT_DIR'], self.env['LOCAL_PROJECT_DIR']) if self.env['PROJECT_DIR'] in path else path

    """ create environ file ( defualt: ./configs/itao_env.json )"""
    def create_env_file(self, is_docker=False):

        local_project_dir = os.path.join(self.cwd, 'tasks')
        local_data_dir = os.path.join(local_project_dir, 'data')

        content = { "KEY": "nvidia_tlt",
                    "NUM_GPUS": "1",
                    "LOCAL_PROJECT_DIR": f"{local_project_dir}",
                    "LOCAL_DATA_DIR": f"{local_data_dir}",
                    "LOCAL_EXPERIMENT_DIR": "",
                    "PROJECT_DIR": "/workspace/tao-experiments",
                    "USER_EXPERIMENT_DIR": "",
                    "DATA_DOWNLOAD_DIR": "",
                    "CLI": "ngccli_cat_linux.zip" }

        if is_docker:
            self.logger.warning('Running in docker ...')
            content["PROJECT_DIR"]=f"{local_project_dir}"

        if not os.path.exists(os.path.dirname(self.env_cfg)):
            os.makedirs(os.path.dirname(self.env_cfg))
            
        with open(self.env_cfg, "w") as env:
            json.dump(content, env, indent=4)
        
        self.logger.info("Created Environment File")

    """ update environ file ( itao_env.json ) """
    def update(self, key, val, log=True):
        self.get_newest_env()
        self.env[key]=val
        with open(self.json_file_path, 'w') as file:
            json.dump(self.env, file, indent=2)
        if log: self.logger.info(f'Upd env: "{key}" -> {val}')


    """ update environ file ( itao_env.json ) """
    def update2(self,prim_key , key, val, log=True):
        self.get_newest_env()
        
        if prim_key not in self.env.keys():
            self.env[prim_key] = dict()

        self.env[prim_key][key] = val
        # self.env[key]=val
        with open(self.json_file_path, 'w') as file:
            json.dump(self.env, file, indent=2)

        if log: self.logger.info(f'Upd env: [{prim_key}][{key}]={val}')


    """ get environ from environ file """
    def get_env(self, key, key2=None):
        self.get_newest_env()
        return self.env[str(key)] if key2 == None else self.env[str(key)][str(key2)]

    """ just get path of workspace """
    def get_workspace_path(self):
        return self.env["PROJECT_DIR"]

    """ update newest environ """
    def get_newest_env(self):
        with open(self.json_file_path, 'r') as file:
            self.env = json.load(file)
        return(self.env)

    """ Important: create mount file for TAO Tookit which identify local path to docker path """
    def create_mount_json(self, is_docker=True):
        self.logger.info("Creating json file for mount path into docker ... ")

        if self.env["LOCAL_PROJECT_DIR"] == "":
            self.logger.info(f"Please update LOCAL_PROJECT_DIR in {self.json_file_path}")
            return 0
        if self.env["LOCAL_SPECS_DIR"] == "":
            self.logger.info(f"Please update LOCAL_SPECS_DIR in {self.json_file_path}")
            return 0
        if self.env["SPECS_DIR"] == "":
            self.logger.info(f"Please update SPECS_DIR in {self.json_file_path}")
            return 0

        drive_map = {
            "Mounts": [
                # Mapping the data directory
                {
                    "source": self.env["LOCAL_PROJECT_DIR"],
                    "destination": self.env["PROJECT_DIR"]
                },
                # Mapping the specs directory.
                {
                    "source": self.env["LOCAL_SPECS_DIR"],
                    "destination": self.env["SPECS_DIR"]
                },
            ],
            "DockerOptions":{
                "user": "{}:{}".format(os.getuid(), os.getgid())
            }
        }

        if is_docker:
            drive_map["Mounts"][0]["destination"]=self.env["LOCAL_PROJECT_DIR"]
            drive_map["Mounts"][1]["destination"]=self.env["LOCAL_SPECS_DIR"]

        # Writing the mounts file.
        with open(self.mounts_file, "w") as mfile:
            json.dump(drive_map, mfile, indent=4)
        return 1

if __name__=='__main__':
    pass
    # env = SetupEnv()
    # print(env.get_newest_env())

    # print('-'*50)
    # env.update('TEST','only for test')
    # print(env.get_newest_env())

    # print('-'*50)
    # env.create_mount_json()
