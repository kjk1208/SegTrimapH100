import logging
import os
import sys
import torch
import torch.distributed as dist
from tqdm import tqdm

def setup_logger(name="ddp_debug_logger"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 핸들러 중복 방지
    if not logger.hasHandlers():
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(levelname)s][%(asctime)s][%(name)s] %(message)s", "%H:%M:%S")
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def is_master_process():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True  # single GPU


def init_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

        return rank, world_size, local_rank
    return 0, 1, 0


if __name__ == "__main__":
    rank, world_size, local_rank = init_ddp()
    logger = setup_logger()

    logger.info("Logger created.")
    print("Raw print works too.")
    sys.stdout.flush()

    logger.info(f"RANK={rank}, WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")
    logger.info(f"is_master_process={is_master_process()}")
    logger.info(f"Logger Handlers = {logger.handlers}")
    logger.info(f"Logger Level = {logger.level} ({logging.getLevelName(logger.level)})")

    # tqdm 테스트
    tbar = tqdm(range(10), file=sys.stdout, ncols=80)
    for i in tbar:
        tbar.set_description(f"[Rank {rank}] Progress")
        torch.cuda.synchronize()
