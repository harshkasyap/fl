import logging
import json

logger = logging.getLogger(__name__)
format = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d) - %(message)s"

def init(file, arg = None):
  file = './logs/' + file.split(".")[0] + ".log"
  if arg == 'info' or arg == 'INFO':
    logging.basicConfig(filename=file, format=format, level=logging.INFO)
  elif arg == 'debug' or arg == 'DEBUG':
    logging.basicConfig(filename=file, format=format, level=logging.DEBUG)
  else:
    logging.basicConfig(format=format, level=logging.INFO)

def modeldebug(nn_model, info):
  logger.debug(info + " ")
  for params in nn_model.parameters():
    logger.debug(str(params))

def jsoninfo(_json, info):
  logger.info(info + " " + json.dumps(_json, indent=4, sort_keys=True))

def jsondebug(_json, info):
  logger.debug(info + " " + json.dumps(_json, indent=4, sort_keys=True))