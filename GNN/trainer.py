import copy
from train_utils import AverageMeter
import wandb
from tqdm.auto import tqdm
import torch

def train(args, num_epochs, model, criterion, optimizer, scheduler, train_loader, train_batch_size, train_size, val_loader, val_batch_size, val_size, device):
    best_model = copy.deepcopy(model).to("cpu", non_blocking=True)
    best_val_loss = float("inf")
    val_loss_avg_meter = AverageMeter()
    val_mae_avg_meter = AverageMeter()

    metric = torch.nn.L1Loss()

    for epoch in range(num_epochs):
        model.train()
        tqdm_iter = tqdm(train_loader, total=len(train_loader))
        tqdm_iter.set_description(f"Epoch {epoch}")

        for it, batch in enumerate(tqdm_iter):
            optimizer.zero_grad()
            
            batch = batch.to(device, non_blocking=True)
            m = batch.y

            out = model(batch)

            loss = criterion(out, m.unsqueeze(-1))

            if args.output_norm_scaling:
                m = m * args.output_norm_value
                out = out * args.output_norm_value

            if args.output_mean_scaling:
                m = m + args.output_mean_value
                out = out + args.output_mean_value

            mae = metric(out.detach(), m.unsqueeze(-1))
            
            tqdm_iter.set_postfix(loss=loss.item(), mae=mae.item())
            wandb.log(
                {
                    "train_loss": loss.item(),
                    "train_mae": mae.item(),
                    "train_step": (it * train_batch_size) + epoch * train_size,
                }
            )

            loss.backward()
            optimizer.step()

        model.eval()
        val_tqdm_iter = tqdm(val_loader, total=len(val_loader))
        val_tqdm_iter.set_description(f"Validation Epoch {epoch}")
        val_loss_avg_meter.reset()
        val_mae_avg_meter.reset()

        for it, batch in enumerate(val_tqdm_iter):
            with torch.no_grad():
                
                batch = batch.to(device, non_blocking=True)
                m = batch.y

                out = model(batch)

                loss = criterion(out, m.unsqueeze(-1))

                if args.output_norm_scaling:
                    m = m * args.output_norm_value
                    out = out * args.output_norm_value

                if args.output_mean_scaling:
                    m = m + args.output_mean_value
                    out = out + args.output_mean_value

                mae = metric(out, m.unsqueeze(-1))

                val_loss_avg_meter.update(loss.item(), out.size(0))
                val_mae_avg_meter.update(mae.item(), out.size(0))
                val_tqdm_iter.set_postfix(loss=val_loss_avg_meter.avg, mae=val_mae_avg_meter.avg)
                wandb.log(
                    {
                        "val_loss": loss.item(),
                        "val_mae": mae.item(),
                        "val_step": (it * val_batch_size) + epoch * val_size,
                    }
                )

        if val_loss_avg_meter.avg < best_val_loss:
            best_model = copy.deepcopy(model).to("cpu", non_blocking=True)
            best_val_loss = val_loss_avg_meter.avg

        scheduler.step()

    del model

    return best_model.to(device, non_blocking=True)