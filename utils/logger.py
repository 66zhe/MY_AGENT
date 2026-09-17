import logging
import os

def get_logger(name="agent"):
    """
    获取一个配置好的 Logger 实例。
    同时输出到控制台和 agent.log 文件。
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加 handler (防止日志打印多次)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # 1. 控制台输出 (INFO 级别以上)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_fmt)

    # 2. 文件输出 (DEBUG 级别以上，写入 agent.log)
    # 确保日志文件生成在项目根目录，而不是 utils 目录下
    log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger