# Tensorboard log creation
from torch.utils.tensorboard import SummaryWriter
from utils.md_report import create_md_summary
from collections import deque
from parameters import *
import gymnasium as gym
import torch
import numpy as np
import random
from time import time

class InfoPlot:
    def __init__(self, gym_id: str,name: str, folder: str = "logs/", rnd: bool=False) -> SummaryWriter:
        
        print(f"Experiment name: {name}")

        self.loss_index = 0
        self.train_index = 0
        self.test_index = 0

        self.timer = time()
        self.buff = deque(maxlen=100)

        # Handle seed
        seed = self.set_seed(rnd)

        print(f"Using seed: {seed}") 
        if name is not None:

            # Create tensorboard logger
            self.logger = SummaryWriter(folder + name)
            # Create a md file for hyperparam
            create_md_summary(gym_id, name, folder, seed=seed)

        else:
            self.logger = None

    def add_train_rew(self, reward: int, tag: str = "Train/Episode Reward") -> None:

        if self.logger is not None:
            self.logger.add_scalar(tag, reward, self.train_index)
            self.train_index += 1
        
    def add_loss(self, loss: int, tag: str = "Train/Loss") -> None:
        
        if self.logger is not None:
            if type(loss) == float:
                self.logger.add_scalar(tag, loss, self.loss_index)
            else: 
                self.logger.add_scalar(tag, loss.item(), self.loss_index)
            self.loss_index += 1

    def add_test(self, reward: int, tag: str = "Test/Reward") -> None:

        if self.logger is not None:
            self.logger.add_scalar(tag, reward, self.test_index)
            self.test_index +=1
    
    def close(self):
        if self.logger is not None:
            self.logger.flush()
            self.logger.close()

    def show_progress(self, update) -> None: 
        
        if update != 1:
            dt = time()-self.timer
            epoch_speed = 1/dt
        else: 
            epoch_speed = 0
        self.timer = time()
        self.buff.append(epoch_speed)
        avg = sum(self.buff)/len(self.buff)
        progress = f"\rProgress: {update/MAX_EPOCH*100:2.2f} %"
        speed = f"\t Epoch/s: {epoch_speed:2.2f}"
        avg_string = f"\t Average speed: {avg:2.2f}"
        remaning_time = (MAX_EPOCH-update)/avg
        time_to_go = f"\t Remaning time: {remaning_time/60:3.0f} min {remaning_time%60:2.0f} s"
        print(progress + speed + avg_string + time_to_go, end="")

    def set_seed(self, rnd: bool = False) -> float:
        """
        Function to set seed on all packages exept for gymansium
        :param rnd: Flag for random seed, if true use time() as seed
        """
        if rnd:
            random.seed(time())
            np.random.seed(time())
            torch.manual_seed(time())
            torch.backends.cudnn.deterministic = False
            return time()
        else:
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.backends.cudnn.deterministic = True
            return SEED
        
