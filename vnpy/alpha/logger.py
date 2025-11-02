import sys

from loguru import logger


# Remove default output
logger.remove()


# Add terminal output
fmt: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
logger.add(sys.stdout, colorize=True, format=fmt)

import time
import psutil
import functools
import logging

# 配置 logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def log_time_memory(func):
    """
    装饰器：打印函数执行时间和内存变化
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()  # 当前进程
        rss_before = process.memory_info().rss  # 执行前内存

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        rss_after = process.memory_info().rss  # 执行后内存
        elapsed = end_time - start_time

        logging.info(
            f"{func.__name__} 耗时: {elapsed:.3f}s | "
            f"内存变化: {(rss_after - rss_before) / 1024 / 1024:.1f}MB | "
            f"RSS: {rss_after / 1024 / 1024:.1f}MB"
        )
        return result
    return wrapper
