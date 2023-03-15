# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys

import paddle.distributed as dist

__all__ = ['setup_logger']

logger_initialized = []


def setup_logger(name="ppdet", output=None):
    """
    Initialize logger and set its verbosity level to INFO.
    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger

    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger

    logger.setLevel(logging.INFO)   # 设置日志器将会处理的日志INFO及以上级别的消息
    logger.propagate = False

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%m/%d %H:%M:%S")
    # stdout logging: master only
    local_rank = dist.get_rank()
    if local_rank == 0:
        # 将日志消息发送到输出到Stream，如std.out, std.err或任何file-like对象
        ch = logging.StreamHandler(stream=sys.stdout)
        # 设置handler将会处理的日志消息的最低严重级别
        ch.setLevel(logging.DEBUG)
        # 为handler设置一个格式器对象
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if local_rank > 0:
            filename = filename + ".rank{}".format(local_rank)
        os.makedirs(os.path.dirname(filename))
        # 将日志消息发送到磁盘文件，默认情况下文件大小会无限增长
        fh = logging.FileHandler(filename, mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter())
        logger.addHandler(fh)
    logger_initialized.append(name)
    return logger
