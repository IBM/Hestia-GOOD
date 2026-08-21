import logging

from tqdm import tqdm


def welcome_autohestia() -> str:
    from hestia import __version__

    mssg = f"  AutoHestia v.{__version__}\n"
    mssg += "By Raul Fernandez-Diaz"
    max_width = max([len(line) for line in mssg.split('\n')])
    out = "-" * (max_width + 4) + "\n"
    for line in mssg.split('\n'):
        out += "| " + line + " " * (max_width - len(line)) + " |" + "\n"
    out += "-" * (max_width + 4) + "\n"
    return out


class TqdmHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)  # , file=sys.stderr)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            sys.exit(0)
            raise KeyboardInterrupt
        except:
            self.handleError(record)


def define_logger(logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    console_handler = logging.StreamHandler()
    logger_formatter = logging.Formatter(
        '{message}',
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(logger_formatter)
    # logger.addHandler(console_handler)
    logger.addHandler(TqdmHandler())
    return logger
