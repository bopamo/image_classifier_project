#########################################################################
#########################################################################
#########################################################################
#########################################################################

import logging

spacing_string = "\n"


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
# fh = logging.FileHandler('logs.log')
# fh.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create a logging format
F = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
LOG_FORMAT = "[%(levelname)s]  \t%(filename)s : %(funcName)s() : line #: %(lineno)d | %(message)s"


# create formatter and add it to the handlers

formatter = logging.Formatter(LOG_FORMAT)
# fh.setFormatter(formatter)
ch.setFormatter(formatter)

# add the handlers to the logger
# logger.addHandler(fh)
logger.addHandler(ch)


def log_program_step(debug_string):

    print(spacing_string)
    logger.info(debug_string)
    print(spacing_string)

    return False
