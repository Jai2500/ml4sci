import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""

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

def get_optimizer(model, lr, lr_step, lr_gamma):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=lr_step, gamma=lr_gamma)

    return optimizer, scheduler

def get_criterion(criterion_type, beta=None):
    if criterion_type == 'mse' or criterion_type == 'l2':
        return torch.nn.MSELoss()
    elif criterion_type == 'l1':
        return torch.nn.L1Loss()
    elif criterion_type == 'smoothl1':
        return torch.nn.SmoothL1Loss(beta=beta)

def get_test_metric():
    return torch.nn.L1Loss()
