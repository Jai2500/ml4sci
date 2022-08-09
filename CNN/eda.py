import glob
import os
from dataset import ImageDatasetFromParquet
from tqdm.auto import tqdm
import torch
from train_utils import AverageMeter

root_dir = './data'

paths = list(glob.glob(os.path.join(root_dir, "*.parquet")))

dsets = []
for path in tqdm(paths[0:1]):
    dsets.append(ImageDatasetFromParquet(
        path, transforms=[]))

combined_dset = torch.utils.data.ConcatDataset(dsets)
loader = torch.utils.data.DataLoader(combined_dset, shuffle=True, batch_size=128)

avg_meter = AverageMeter()
sq_avg_meter = AverageMeter()
rel_count = 0

pbar = tqdm(loader, total=len(loader))
for data in pbar:
    avg_meter.update(
        data['X_jets'].mean(dim=[0, 2, 3]),
        n=data['m'].shape[0]
    )
    sq_avg_meter.update(
        (data['X_jets'] ** 2).mean(dim=[0,2,3]),
        n=data['m'].shape[0]
    )

    rel_count += data['m'].shape[0]

    if rel_count >= 1000:
        print(f"current_mean_estimate={avg_meter.avg.numpy()}, current_std_estimate={(sq_avg_meter.avg - avg_meter.avg**2).numpy()}")
        rel_count -= 1000

print(avg_meter.avg.numpy(), avg_meter.count)

