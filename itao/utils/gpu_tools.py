import sys
from itao.utils.qt_logger import CustomLogger

try:
    import GPUtil
except ModuleNotFoundError:
    sys.exit()    

def get_available_device(max_memory=0.8, logger=None):
    '''
    select available device based on the memory utilization status of the device
    :param max_memory: the maximum memory utilization ratio that is considered available
    :return: GPU id that is available, -1 means no GPU is available/uses CPU, if GPUtil package is not installed, will
    return 0 
    '''
    GPUs = GPUtil.getGPUs()
    freeMemory = 0
    available=[]
    for GPU in GPUs:
        if GPU.memoryUtil > max_memory:
            continue
        if GPU.memoryFree >= freeMemory:
            # freeMemory = GPU.memoryFree
            available.append(GPU)

    return available

def get_gpu_name(logger=None):
    """Returns the model name of the first available GPU"""
    try:
        gpus = GPUtil.getGPUs()
    except:
        if logger != None:
            logger.warning("Unable to detect GPU model. Is your GPU configured? Are you running with nvidia-docker?")
        return "UNKNOWN"
    if len(gpus) == 0:
        raise ValueError("No GPUs detected in the system")
    return gpus[0].name 