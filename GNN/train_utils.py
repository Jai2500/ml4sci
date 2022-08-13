import torch


def get_optimizer(model, lr, lr_step, lr_gamma):
    '''
        Returns the optimizer and the scheduler for the model
        Args:
            model: The model 
            lr: The learning rate for the optimizer
            lr_step: The step after which lr will be reduced
            lr_gamma: The amount by which the lr will be reduced

        Returns:
            optimizer: The optimizer for the model
            scheduler: The scheduler for the optimizer
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=lr_step, gamma=lr_gamma)

    return optimizer, scheduler

def get_criterion(criterion_type, beta=None):
    '''
        Returns the criterion based on the criterion_type
        Args:
            criterion_type: Which criterion (loss_fn) to return
            beta: Additional paramter required for the SmoothL1Loss

        Returns:
            The criterion based on the criterion_type
    '''
    if criterion_type == 'mse' or criterion_type == 'l2':
        return torch.nn.MSELoss()
    elif criterion_type == 'l1':
        return torch.nn.L1Loss()
    elif criterion_type == 'smoothl1':
        return torch.nn.SmoothL1Loss(beta=beta)

def get_test_metric():
    '''
        Returns the test metric
    '''
    return torch.nn.L1Loss()
