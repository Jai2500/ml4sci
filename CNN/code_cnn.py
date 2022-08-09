import pyarrow.parquet as pq
import numpy as np
import torch
import wandb
from tqdm.auto import tqdm
from torchmetrics import AUROC, ROC, Accuracy
import torchvision.transforms as T
import torchvision
import copy
import glob


class ImageDatasetFromParquet(torch.utils.data.Dataset):
    def __init__(self, filename, transforms=[]) -> None:
        super().__init__()

        self.file = pq.ParquetFile(filename)

        self.transforms = T.Compose([T.ToTensor(), *transforms])

    def __getitem__(
        self,
        idx,
    ):
        row = self.file.read_row_group(idx).to_pydict()
        to_return = {
            "X_jets": self.transforms(np.array(row["X_jet"][0]).reshape(125, 125, 8)),
            "m": row["m"][0],
            "pt": row["pt"][0],
            "ieta": row["ieta"][0],
            "iphi": row["iphi"][0],
        }

        return to_return

    def __len__(self):
        return self.file.num_row_groups


required_transform = [
    # T.Resize(224),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    # T.Normalize(mean=[0.485, 0.456, 0.406],
    #             std=[0.229, 0.224, 0.225]),
    # T.RandomAdjustSharpness(0.5, p=0.1),
]

paths = list(glob.glob("./*.parquet"))

dsets = []
for path in tqdm(paths[0:1]):
    dsets.append(ImageDatasetFromParquet(path, transforms=required_transform))


combined_dset = torch.utils.data.ConcatDataset(dsets)

TEST_SIZE = 0.2
VAL_SIZE = 0.15

test_size = int(len(combined_dset) * TEST_SIZE)
val_size = int(len(combined_dset) * VAL_SIZE)
train_size = len(combined_dset) - val_size - test_size

train_dset, val_dset, test_dset = torch.utils.data.random_split(
    combined_dset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42),
)

print(train_size, val_size, test_size)

test_dset.required_transforms = []

TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64

train_loader = torch.utils.data.DataLoader(
    train_dset, shuffle=True, batch_size=TRAIN_BATCH_SIZE, pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    val_dset, shuffle=False, batch_size=VAL_BATCH_SIZE, pin_memory=True
)
test_loader = torch.utils.data.DataLoader(
    test_dset, shuffle=False, batch_size=TEST_BATCH_SIZE
)


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


class RegressModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Identity()

        self.out_lin = torch.nn.Sequential(
            torch.nn.Linear(in_features + 3, in_features // 2, bias=True),
            torch.nn.BatchNorm1d(in_features // 2),
            torch.nn.SiLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(in_features // 2, in_features // 4, bias=True),
            torch.nn.BatchNorm1d(in_features // 4),
            torch.nn.SiLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(in_features // 4, 1, bias=True),
        )

    def forward(self, X, pt, ieta, iphi):
        out = self.model(X)
        out = torch.cat(
            [out, pt.unsqueeze(-1), ieta.unsqueeze(-1), iphi.unsqueeze(-1)], dim=1
        )
        return self.out_lin(out)


def get_model(device, pretrained=False):
    regress_model = RegressModel(
        model=torchvision.models.resnet18(pretrained=pretrained)
    )
    regress_model.model.conv1 = torch.nn.Conv2d(
        8, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    regress_model = regress_model.to(device)

    return regress_model


def get_optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr)


def train(num_epochs, model, criterion, optimizer, train_loader, val_loader, device):
    best_model = copy.deepcopy(model).to("cpu", non_blocking=True)
    best_val_loss = float("inf")
    val_loss_avg_meter = AverageMeter()

    for epoch in range(num_epochs):
        model.train()
        tqdm_iter = tqdm(train_loader, total=len(train_loader))
        tqdm_iter.set_description(f"Epoch {epoch}")

        for it, batch in enumerate(tqdm_iter):
            optimizer.zero_grad()

            X, pt, ieta, iphi, m = (
                batch["X_jets"].float(),
                batch["pt"].float(),
                batch["ieta"].float(),
                batch["iphi"].float(),
                batch["m"].float(),
            )

            X = X.to(device, non_blocking=True)
            pt = pt.to(device, non_blocking=True)
            ieta = ieta.to(device, non_blocking=True)
            iphi = iphi.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)

            out = model(X, pt, ieta, iphi)

            loss = criterion(out, m.unsqueeze(-1))

            tqdm_iter.set_postfix(loss=loss.item())
            wandb.log(
                {
                    "train_mse_loss": loss.item(),
                    "train_step": (it * TRAIN_BATCH_SIZE) + epoch * train_size,
                }
            )

            loss.backward()
            optimizer.step()

        model.eval()
        val_tqdm_iter = tqdm(val_loader, total=len(val_loader))
        val_tqdm_iter.set_description(f"Validation Epoch {epoch}")
        val_loss_avg_meter.reset()

        for it, batch in enumerate(val_tqdm_iter):
            with torch.no_grad():
                X, pt, ieta, iphi, m = (
                    batch["X_jets"].float(),
                    batch["pt"].float(),
                    batch["ieta"].float(),
                    batch["iphi"].float(),
                    batch["m"].float(),
                )

                X = X.to(device, non_blocking=True)
                pt = pt.to(device, non_blocking=True)
                ieta = ieta.to(device, non_blocking=True)
                iphi = iphi.to(device, non_blocking=True)
                m = m.to(device, non_blocking=True)

                out = model(X, pt, ieta, iphi)

                loss = criterion(out, m.unsqueeze(-1))

                val_tqdm_iter.set_postfix(loss=loss.item())
                wandb.log(
                    {
                        "val_mse_loss": loss.item(),
                        "val_step": (it * VAL_BATCH_SIZE) + epoch * val_size,
                    }
                )
                val_loss_avg_meter.update(loss.item(), out.size(0))

        if val_loss_avg_meter.avg < best_val_loss:
            best_model = copy.deepcopy(model).to("cpu", non_blocking=True)
            best_val_loss = val_loss_avg_meter.avg

    del model

    return best_model.to(device, non_blocking=True)


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss_avg_meter = AverageMeter()
    tqdm_iter = tqdm(test_loader, total=len(test_loader))

    pred_list = []
    ground_truth_list = []

    for it, batch in enumerate(tqdm_iter):
        with torch.no_grad():
            X, pt, ieta, iphi, m = (
                batch["X_jets"].float(),
                batch["pt"].float(),
                batch["ieta"].float(),
                batch["iphi"].float(),
                batch["m"].float(),
            )

            X = X.to(device, non_blocking=True)
            pt = pt.to(device, non_blocking=True)
            ieta = ieta.to(device, non_blocking=True)
            iphi = iphi.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)

            out = model(X, pt, ieta, iphi)

            loss = criterion(out, m.unsqueeze(-1))

            tqdm_iter.set_postfix(loss=loss.item())

            test_loss_avg_meter.update(loss.item(), out.size(0))

    return test_loss_avg_meter.avg


def main(run_name, num_epochs, device):
    wandb.init(name=run_name, project="gsoc-cnn-runs")

    model = get_model(device)

    opt = get_optimizer(model, lr=1e-3)
    criterion = torch.nn.MSELoss()

    model = train(num_epochs, model, criterion, opt, train_loader, val_loader, device)
    test_loss = test(model, test_loader, criterion, device)
    print(f"Model on Test dataset: Loss: {test_loss}")

    wandb.finish()

    return model


NUM_EPOCHS = 5
DEVICE = "cuda"
model = main("test_regress_local", NUM_EPOCHS, DEVICE)
