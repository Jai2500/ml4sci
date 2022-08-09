import torch
import numpy as np
import torch_geometric

def get_loaders(train_dset, val_dset, test_dset, train_batch_size, val_batch_size, test_batch_size):
    train_loader = torch_geometric.data.DataLoader(
        train_dset, shuffle=True, batch_size=train_batch_size, pin_memory=True
    )
    val_loader = torch_geometric.data.DataLoader(
        val_dset, shuffle=False, batch_size=val_batch_size, pin_memory=True
    )
    test_loader = torch_geometric.data.DataLoader(
        test_dset, shuffle=False, batch_size=test_batch_size
    )
    return train_loader, val_loader, test_loader


def positional_encoding(data, pe_scales):
    pe_cos = torch.cat([torch.cos(2**i * np.pi * torch.as_tensor(data))
                       for i in range(pe_scales)], dim=1)
    pe_sin = torch.cat([torch.sin(2**i * np.pi * torch.as_tensor(data))
                       for i in range(pe_scales)], dim=1)

    output= torch.cat([torch.as_tensor(data), pe_cos, pe_sin], dim=1)
    return output


def normalize_x(x):
    x = x - np.array([0.01037084, 0.0103173, 0.01052679, 0.01034378, 0.01097225, 0.01024814, 0.01037642, 0.01058754])
    x = x / np.array([10.278656283775618, 7.64753320751208, 16.912319597559645, 9.005579923580713, 21.367327333103688, 7.489890622699373, 12.977402491253788, 24.50774893130742])

    return x