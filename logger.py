import logging
logging.basicConfig(level=logging.INFO,
                    filename='./service.log',
                    format='%(asctime)s - %(levelname)s\n%(message)s\n',
                    filemode='a')

def log_debug(str):
    logging.debug(str)
    
def log_info(str):
    logging.info(str)
    
def log_warning(str):
    logging.warn(str)