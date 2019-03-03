
import logging
# import coloredlogs

from coloredlogs import ColoredFormatter

# Create a logger object.
# logger = logging.getLogger(__name__)

# By default the install() function installs a handler on the root logger,
# this means that log messages from your code and log messages from the
# libraries that you use will all show up on the terminal.
# coloredlogs.install(level='DEBUG')

# If you don't want to see log messages from libraries, you can pass a
# specific logger object to the install() function. In this case only log
# messages originating from that logger will show up on the terminal.
# coloredlogs.install(level='DEBUG', logger=logger)

# coloredlogs.install()

# Add console handler using our custom ColoredFormatter
# LOGFORMAT = "[%(levelname)s]  \t%(filename)s : %(funcName)s() : line #: %(lineno)d | %(message)s"
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# cf = ColoredFormatter(LOGFORMAT)
# ch.setFormatter(cf)
# logger.addHandler(ch)


#
# import logging
# LOG_LEVEL = logging.DEBUG
#
# F =  "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
# LOG_FORMAT = "[%(levelname)s]  \t%(filename)s : %(funcName)s() : line #: %(lineno)d | %(message)s"
# from colorlog import ColoredFormatter
# logging.root.setLevel(LOG_LEVEL)
# formatter = ColoredFormatter(LOG_FORMAT)
# stream = logging.StreamHandler()
# stream.setLevel(LOG_LEVEL)
# stream.setFormatter(formatter)
# logger = logging.getLogger('pythonConfig')
# logger.setLevel(LOG_LEVEL)
# logger.addHandler(stream)
#
# spacing_string = "\n"


# import logging
# from colored_log import ColoredFormatter
#
# # Create top level logger
# log = logging.getLogger("main")
#
# # Add console handler using our custom ColoredFormatter
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# cf = ColoredFormatter("[%(name)s][%(levelname)s]  %(message)s (%(filename)s:%(lineno)d)")
# ch.setFormatter(cf)
# log.addHandler(ch)
#
# # Add file handler
# fh = logging.FileHandler('app.log')
# fh.setLevel(logging.DEBUG)
# ff = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh.setFormatter(ff)
# log.addHandler(fh)
#
# # Set log level
# log.setLevel(logging.DEBUG)
#
# # Log some stuff
# log.debug("app has started")
# log.info("Logging to 'app.log' in the script dir")
# log.warning("This is my last warning, take heed")
# log.error("This is an error")
# log.critical("He's dead, Jim")



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
