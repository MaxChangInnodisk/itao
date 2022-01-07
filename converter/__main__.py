"""
converter is use to converter .etlt model to .engine ( TensorRT ) model
"""
import sys,os,time


try:
    from itao.utils.qt_logger import CustomLogger
except ModuleNotFoundError:
    sys.path.append("../itao")
    from itao.utils.qt_logger import CustomLogger

logger = CustomLogger().get_logger('dev', write_mode='a')

logger.info('Start converter ... ')

# Do somthing

logger.info('Convert finish')