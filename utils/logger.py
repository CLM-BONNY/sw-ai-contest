import logging
import os
from datetime import datetime


try:
    import wandb
except ImportError:
    wandb = None


# 로깅 설정 반환 함수
def get_logger(name, save_dir="logs"):
    # 로그 파일 경로 설정
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"{name}_{timestamp}.log")

    # 로거 객체 생성 및 설정
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 기존 핸들러 제거 (중복 방지)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 콘솔 핸들러 생성
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)

    # 파일 핸들러 생성
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_formatter)

    # 핸들러 등록
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# 콘솔 + 파일 + W&B metric 기록 함수
def log_metrics(logger, metrics: dict, step: int = None):
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")
    if wandb and wandb.run is not None:
        wandb.log(metrics, step=step)
