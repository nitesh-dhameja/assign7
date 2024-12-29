import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

class CutMix:
    def __init__(self, beta=1.0):
        self.beta = beta
        
    def __call__(self, batch, targets):
        lam = np.random.beta(self.beta, self.beta)
        batch_size = batch.size()[0]
        index = torch.randperm(batch_size)
        
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(batch.size(), lam)
        batch[:, :, bbx1:bbx2, bby1:bby2] = batch[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch.size()[-1] * batch.size()[-2]))
        
        targets_a = targets
        targets_b = targets[index]
        
        return batch, targets_a, targets_b, lam
    
    def _rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def setup_distributed():
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(torch.distributed.get_rank())

def get_learning_rate(epoch, max_epochs):
    # Cosine learning rate schedule
    return 0.5 * (1 + np.cos(np.pi * epoch / max_epochs)) 