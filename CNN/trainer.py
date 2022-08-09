import copy
from train_utils import AverageMeter
import wandb
from tqdm.auto import tqdm

def train(num_epochs, model, criterion, optimizer, scheduler, train_loader, train_batch_size, train_size, val_loader, val_batch_size, val_size, device):
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
                    "train_step": (it * train_batch_size) + epoch * train_size,
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
                        "val_step": (it * val_batch_size) + epoch * val_size,
                    }
                )
                val_loss_avg_meter.update(loss.item(), out.size(0))

        if val_loss_avg_meter.avg < best_val_loss:
            best_model = copy.deepcopy(model).to("cpu", non_blocking=True)
            best_val_loss = val_loss_avg_meter.avg

        scheduler.step()

    del model

    return best_model.to(device, non_blocking=True)