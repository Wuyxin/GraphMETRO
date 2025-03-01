import logging
import os

class Logger:
    logger = None

    @staticmethod
    def get_logger(name: str = None, fname=None):
        if not Logger.logger:
            Logger.init_logger(name=name, fname=fname)
        return Logger.logger

    @staticmethod
    def init_logger(level=logging.INFO,
                    fmt='%(asctime)s - %(levelname)s: %(message)s',
                    fname: str = None, name: str = None):
        logger = logging.getLogger(name)
        logger.setLevel(level)

        fmt = logging.Formatter(fmt)

        if os.path.exists(fname):
            os.remove(fname)
        if fname:
            # file handler
            fh = logging.FileHandler(fname)
            fh.setLevel(level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

        logger.setLevel(level)
        Logger.logger = logger
        return logger

    @staticmethod
    def deinit_logger():
        if Logger.logger is not None:
            for h in Logger.logger.handlers:
                Logger.logger.removeHandler(h)
            Logger.logger = None