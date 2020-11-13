import logging

logger = logging.getLogger(__name__)
format = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d) - %(message)s"

def init(arg, file):
  if arg == 'info' or arg == 'INFO':
    logging.basicConfig(filename=file, format=format, level=logging.INFO)
  elif arg == 'debug' or arg == 'DEBUG':
    logging.basicConfig(filename=file, format=format, level=logging.DEBUG)
  else:
    logging.basicConfig(format=format, level=logging.DEBUG)