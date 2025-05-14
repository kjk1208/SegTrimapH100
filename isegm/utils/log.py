# #isegm/utils/log.py

import io
import logging
import sys
import time
from datetime import datetime

import numpy as np
from torch.utils.tensorboard import SummaryWriter

LOGGER_NAME = 'root'
LOGGER_DATEFMT = '%Y-%m-%d %H:%M:%S'

# === Custom stdout logger handler (DDP-safe) ===
class StdoutPrintHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            print(msg, file=sys.stdout, flush=True)  # 강제로 stdout 출력
        except Exception:
            self.handleError(record)

# === Logger 초기화 ===
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)

# 기존 핸들러 제거 후 print 기반 핸들러만 추가
for h in logger.handlers[:]:
    logger.removeHandler(h)

stdout_handler = StdoutPrintHandler()
stdout_formatter = logging.Formatter('[%(levelname)s] %(message)s')
stdout_handler.setFormatter(stdout_formatter)
logger.addHandler(stdout_handler)

# === 로그 파일 저장용 핸들러 추가 함수 ===
def add_logging(logs_path, prefix):
    log_name = prefix + datetime.strftime(datetime.today(), '%Y-%m-%d_%H-%M-%S') + '.log'
    log_path = logs_path / log_name

    file_handler = logging.FileHandler(str(log_path))
    file_formatter = logging.Formatter(fmt='(%(levelname)s) %(asctime)s: %(message)s',
                                       datefmt=LOGGER_DATEFMT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


class TqdmToLogger(io.StringIO):
    """tqdm 출력 → logging으로 전달"""
    def __init__(self, logger, level=logging.INFO, mininterval=1.0):
        super().__init__()
        self.logger = logger
        self.level = level
        self.mininterval = mininterval
        self.last_time = 0
        self._buffer = ''

    def write(self, buf):
        self._buffer = buf.strip()

    def flush(self):
        if len(self._buffer) > 0 and time.time() - self.last_time > self.mininterval:
            self.logger.log(self.level, self._buffer)
            self._buffer = ''
            self.last_time = time.time()

    def isatty(self):
        return True


class SummaryWriterAvg(SummaryWriter):
    def __init__(self, *args, dump_period=20, **kwargs):
        super().__init__(*args, **kwargs)
        self._dump_period = dump_period
        self._avg_scalars = {}

    def add_scalar(self, tag, value, global_step=None, disable_avg=False):
        if disable_avg or isinstance(value, (tuple, list, dict)):
            super().add_scalar(tag, np.array(value), global_step=global_step)
            return

        if tag not in self._avg_scalars:
            self._avg_scalars[tag] = ScalarAccumulator(self._dump_period)

        acc = self._avg_scalars[tag]
        acc.add(value)

        if acc.is_full():
            super().add_scalar(tag, acc.value, global_step=global_step)
            acc.reset()


class ScalarAccumulator:
    def __init__(self, period):
        self.period = period
        self.reset()

    def add(self, value):
        self.sum += value
        self.cnt += 1

    def reset(self):
        self.sum = 0
        self.cnt = 0

    def is_full(self):
        return self.cnt >= self.period

    @property
    def value(self):
        return self.sum / self.cnt if self.cnt > 0 else 0

    def __len__(self):
        return self.cnt






# import io
# import time
# import logging
# from datetime import datetime

# import numpy as np
# from torch.utils.tensorboard import SummaryWriter

# LOGGER_NAME = 'root'
# LOGGER_DATEFMT = '%Y-%m-%d %H:%M:%S'

# handler = logging.StreamHandler()

# logger = logging.getLogger(LOGGER_NAME)
# logger.setLevel(logging.INFO)
# logger.addHandler(handler)


# def add_logging(logs_path, prefix):
#     log_name = prefix + datetime.strftime(datetime.today(), '%Y-%m-%d_%H-%M-%S') + '.log'
#     stdout_log_path = logs_path / log_name

#     fh = logging.FileHandler(str(stdout_log_path))
#     formatter = logging.Formatter(fmt='(%(levelname)s) %(asctime)s: %(message)s',
#                                   datefmt=LOGGER_DATEFMT)
#     fh.setFormatter(formatter)
#     logger.addHandler(fh)


# class TqdmToLogger(io.StringIO):
#     logger = None
#     level = None
#     buf = ''

#     def __init__(self, logger, level=None, mininterval=5):
#         super(TqdmToLogger, self).__init__()
#         self.logger = logger
#         self.level = level or logging.INFO
#         self.mininterval = mininterval
#         self.last_time = 0

#     def write(self, buf):
#         self.buf = buf.strip('\r\n\t ')
 
#     def flush(self):
#         print(f"[TqdmToLogger.flush] CALLED with: {self.buf}")  # 디버깅 출력
#         #kjk
#         # if len(self.buf) > 0 and time.time() - self.last_time > self.mininterval:
#         #kjk
#         if len(self.buf) > 0:
#             self.logger.log(self.level, self.buf)
#             self.last_time = time.time()


# class SummaryWriterAvg(SummaryWriter):
#     def __init__(self, *args, dump_period=20, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._dump_period = dump_period
#         self._avg_scalars = dict()

#     def add_scalar(self, tag, value, global_step=None, disable_avg=False):
#         if disable_avg or isinstance(value, (tuple, list, dict)):
#             super().add_scalar(tag, np.array(value), global_step=global_step)
#         else:
#             if tag not in self._avg_scalars:
#                 self._avg_scalars[tag] = ScalarAccumulator(self._dump_period)
#             avg_scalar = self._avg_scalars[tag]
#             avg_scalar.add(value)

#             if avg_scalar.is_full():
#                 super().add_scalar(tag, avg_scalar.value,
#                                    global_step=global_step)
#                 avg_scalar.reset()


# class ScalarAccumulator(object):
#     def __init__(self, period):
#         self.sum = 0
#         self.cnt = 0
#         self.period = period

#     def add(self, value):
#         self.sum += value
#         self.cnt += 1

#     @property
#     def value(self):
#         if self.cnt > 0:
#             return self.sum / self.cnt
#         else:
#             return 0

#     def reset(self):
#         self.cnt = 0
#         self.sum = 0

#     def is_full(self):
#         return self.cnt >= self.period

#     def __len__(self):
#         return self.cnt
