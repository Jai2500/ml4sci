from tqdm.auto import tqdm
from train_utils import AverageMeter

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