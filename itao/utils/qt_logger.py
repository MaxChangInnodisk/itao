import logging

class CustomLogger:
    def __init__(self):
        pass
    
    """ Create logger which name is 'dev' """
    def create_logger(self, name='dev'):

        logger = logging.getLogger(name)
        # setup LEVEL
        logger.setLevel(logging.DEBUG)
        # setup formatter
        formatter = logging.Formatter(
                        "%(asctime)s %(levelname)-.4s %(message)s",
                        "%m-%d %H:%M:%S")
        # setup handler
        stream_handler = logging.StreamHandler()
        file_handler = logging.FileHandler('itao.log', 'w', 'utf-8')
        # add formatter into handler
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # add handler into logger
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

        logger.info('Create Logger: {}'.format(name))
        return logger

    """ get logger """
    def get_logger(self, name='dev'):
        logger = logging.getLogger(name)
        return logger if logger.hasHandlers() else self.create_logger(name)