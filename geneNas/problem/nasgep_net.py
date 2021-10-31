from argparse import ArgumentParser
from datetime import datetime
from typing import Optional, Union, Dict

import numpy as np
import datasets
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup

from network.recurrent_net import RecurrentNet
from util.logger import ChromosomeLogger
from util.exception import NanException

class NasgepNet(pl.LightningModule):
    pass