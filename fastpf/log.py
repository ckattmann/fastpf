import sys
from loguru import logger


def scale_time(time_s):
    """
    Transform a time in seconds to a string with the appropriate unit
    """
    if time_s >= 60:  # >= 1 minute
        minutes = round(time_s // 60)
        seconds = round(time_s % 60)
        return f"{minutes}min {seconds}s"
    if time_s >= 1:  # >= 1 second
        seconds = round(time_s % 60)
        milliseconds = round((time_s % 1) * 1000)
        return f"{seconds}.{milliseconds}s"
    if time_s < 1:  # < 1 second
        milliseconds = round(time_s * 1000)
        return f"{milliseconds}ms"


def set_loglevel(loglevel):

    if loglevel.upper() not in ["DEBUG", "INFO", "ERROR", "OFF"]:
        raise ValueError(
            f"loglevel must be 'DEBUG', 'INFO', 'ERROR', or 'OFF', not {loglevel}"
        )

    logger.remove()

    if loglevel.upper() == "DEBUG":
        logger.add(
            sink=sys.stdout,
            level="DEBUG",
            format="{time:HH:mm:ss} | {elapsed} | <cyan>{module}:{function}</cyan> | <lvl>{message}</lvl>",
        )

    if loglevel.upper() == "INFO":
        logger.add(
            sink=sys.stdout,
            level="INFO",
            format="{time:HH:mm:ss} | {elapsed} | <lvl>{message}</lvl>",
        )

    if loglevel.upper() == "ERROR":
        logger.add(
            sink=sys.stdout,
            level="ERROR",
            format="{time:HH:mm:ss} | {elapsed} | <lvl>{message}</lvl>",
        )

    if loglevel.upper() == "OFF":
        pass


# set_loglevel("INFO")
