from numpy import require
import torch
import torchvision.transforms as T
import numpy as np


def get_transforms():
    required_transform = [
        # T.Resize(224),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        # T.Normalize(mean=[0.5] * 8,
        # std=[0.5] * 8)
        # T.Normalize(mean=[0.01037084, 0.0103173, 0.01052679, 0.01034378, 0.01097225, 0.01024814, 0.01037642, 0.01058754],
        #             std=[10.278656283775618, 7.64753320751208, 16.912319597559645, 9.005579923580713, 21.367327333103688, 7.489890622699373, 12.977402491253788, 24.50774893130742])
        # T.Normalize(mean=[0.485, 0.456, 0.406],
        #             std=[0.229, 0.224, 0.225]),
        # T.RandomAdjustSharpness(0.5, p=0.1),
    ]
    return required_transform


def get_loaders(train_dset, val_dset, test_dset, train_batch_size, val_batch_size, test_batch_size):
    train_loader = torch.utils.data.DataLoader(
        train_dset, shuffle=True, batch_size=train_batch_size, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dset, shuffle=False, batch_size=val_batch_size, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dset, shuffle=False, batch_size=test_batch_size
    )
    return train_loader, val_loader, test_loader


def positional_encoding(data, pe_scales):
    pe_cos = torch.cat([torch.cos(2**i * np.pi * torch.as_tensor(data))
                       for i in range(pe_scales)])
    pe_sin = torch.cat([torch.sin(2**i * np.pi * torch.as_tensor(data))
                       for i in range(pe_scales)])

    return torch.cat([data, pe_cos, pe_sin])


def zero_suppression(X_jets, min_threshold):
    return np.where(np.abs(X_jets) >= min_threshold, X_jets, 0.)
