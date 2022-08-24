from tqdm.auto import tqdm
from train_utils import AverageMeter
import torch

m0_scale    = 85
m1_scale    = 415

def test(args, model, test_loader, criterion, device, output_norm_scaling=False, output_norm_value=1.):
    '''
        Performs testing of the model on the test dataset
        Args:
            model: The model to test on the test dataset
            test_loader: The test dataset data loader
            criterion: The criterion to test the model on
            device: The device on which to perform the testing
            output_norm_scaling[Optional]: Whether the outputs have been normalized
            output_norm_value[Optional]: The amount by which the outputs have been normalized


        Returns:
            Average loss on the test dataset. 
    '''
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
            out = out['regress']

            if output_norm_scaling:
                out *= output_norm_value
                m *= output_norm_value
            elif args.scale_histogram:
                m = m * m1_scale + m0_scale
                out = out * m1_scale + m0_scale

            loss = criterion(out, m.unsqueeze(-1))

            tqdm_iter.set_postfix(loss=loss.item())

            test_loss_avg_meter.update(loss.item(), out.size(0))

    return test_loss_avg_meter.avg