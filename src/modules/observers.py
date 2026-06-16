import torch
from torch import nn

# ********************* observers *********************
class ObserverBase(nn.Module):
    def __init__(self, q_level):
        super(ObserverBase, self).__init__()
        self.q_level = q_level

    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        self.update_range(torch.min(input), torch.max(input))


class MinMaxObserver(ObserverBase):
    def __init__(self, q_level, out_channels):
        super(MinMaxObserver, self).__init__(q_level)
        self.num_flag = 0
        
    def update_range(self, min_val_cur, max_val_cur):
        if self.num_flag == 0:
            self.num_flag += 1
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = torch.min(min_val_cur, self.min_val)
            self.max_val = torch.max(max_val_cur, self.max_val)


class MovingAverageMinMaxObserver(ObserverBase):
    def __init__(self, q_level, out_channels, momentum=0.1):
        super(MovingAverageMinMaxObserver, self).__init__(q_level)
        self.momentum = momentum
        self.num_flag = 0

    def update_range(self, min_val_cur, max_val_cur):
        if self.num_flag == 0:
            self.num_flag += 1
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = (1 - self.momentum) * self.min_val + self.momentum * min_val_cur
            self.max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur