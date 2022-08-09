from tqdm.auto import tqdm
from train_utils import AverageMeter
import torch

def test(model, test_loader, criterion, device, output_norm_scaling=False, output_norm_value=1.):
    model.eval()
    test_loss_avg_meter = AverageMeter()
    tqdm_iter = tqdm(test_loader, total=len(test_loader))

    pred_list = []
    ground_truth_list = []

    for it, batch in enumerate(tqdm_iter):
        with torch.no_grad():
            batch = batch.to(device, non_blocking=True)
            m = batch.y

            out = model(batch)

            if output_norm_scaling:
                out *= output_norm_value
                m *= output_norm_value

            loss = criterion(out, m.unsqueeze(-1))

            tqdm_iter.set_postfix(loss=loss.item())

            test_loss_avg_meter.update(loss.item(), out.size(0))

    return test_loss_avg_meter.avg