import os
import cv2
import torch
import warnings
import logging
import logging.config
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

LOGGING_NAME = "Traffic Light Color Detection"

def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level, }},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False, }}})


set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)

class GenericLogger:
    """
    YOLOv5 General purpose logger for non-task specific logging
    Usage: from utils.loggers import GenericLogger; logger = GenericLogger(...)
    Arguments
        opt:             Run arguments
        console_logger:  Console logger
        include:         loggers to include
    """

    def __init__(self, save_dir, console_logger, include=('tb')):
        # init default loggers
        self.save_dir = Path(save_dir)
        self.include = include
        self.console_logger = console_logger
        self.csv = self.save_dir / 'results.csv'  # CSV logger
        
        prefix = 'TensorBoard: '
        self.console_logger.info(
            f"{prefix}Start with 'tensorboard --logdir {self.save_dir.parent}', view at http://localhost:6006/")
        self.tb = SummaryWriter(str(self.save_dir))
        
    def log_metrics(self, metrics, epoch):
        # Log metrics dictionary to all loggers
        if self.csv:
            keys, vals = list(metrics.keys()), list(metrics.values())
            n = len(metrics) + 1  # number of cols
            s = '' if self.csv.exists() else (('%23s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # header
            with open(self.csv, 'a') as f:
                f.write(s + ('%23.5g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')

        if self.tb:
            for k, v in metrics.items():
                self.tb.add_scalar(k, v, epoch)

    def log_images(self, files, name='Images', epoch=0):
        # Log images to all loggers
        files = [Path(f) for f in (files if isinstance(files, (tuple, list)) else [files])]  # to Path
        files = [f for f in files if f.exists()]  # filter by exists

        if self.tb:
            for f in files:
                self.tb.add_image(f.stem, cv2.imread(str(f))[..., ::-1], epoch, dataformats='HWC')

    # def log_graph(self, model, imgsz=(640, 640)):
    #     # Log model graph to all loggers
    #     if self.tb:
    #         log_tensorboard_graph(self.tb, model, imgsz)

    # def log_model(self, model_path, epoch=0, metadata={}):
    #     # Log model to all loggers
    #     if self.wandb:
    #         art = wandb.Artifact(name=f'run_{wandb.run.id}_model', type='model', metadata=metadata)
    #         art.add_file(str(model_path))
    #         wandb.log_artifact(art)

    # def update_params(self, params):
    #     # Update the parameters logged
    #     if self.wandb:
    #         wandb.run.config.update(params, allow_val_change=True)
