from dataset_utils import get_loaders
from dataset import get_datasets
import wandb
import torch
from model import get_model
from train_utils import get_criterion, get_optimizer, get_test_metric
from trainer import train
from tester import test
import argparse
import os


def save_model(model, save_path):
    '''
        Saves the model to the path provided.
        Args:
            model: The model weights to be saved
            save_path: The root dir path to save the model in
        
        Returns:
            None
    '''
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))


def main(
    args,
    run_name,
    num_epochs,

    model,
    criterion,
    test_metric,

    optimizer,
    scheduler,

    train_loader,
    val_loader,
    test_loader,
    train_size,
    val_size,
    test_size,
    device,

    save_path
):
    ''''
        Runs the training and testing of the model given the arguments and the configurations.
        Args:
            args: The Argparse parsed arguments
            run_name: The name to log the run on WandB
            num_epochs: The number of epochs to run the training for
            model: The model to train
            criterion: The criterion to use for training and validation
            test_metric: The metric for evaluating the test results
            optimizer: The optimizer of the model
            scheduler: The epoch based scheduler for the optimizer
            train_loader: The train dataset data loader
            val_loader: The validation dataset data loader
            test_loader: The test dataset data loader
            train_size: The size of the training dataset
            val_size: The size of the validation dataset
            test_size: The size of the test dataset
            device: The device to run the training on
            save_path: Path on the disk to the folder where to save the best model

        Returns:
            The best model
    '''
    if not args.debug:
        wandb.init(name=run_name, project="gsoc-gnn-runs")
        wandb.config.update(args)

    if device == 'cuda':
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available on the device. Falling back to CPU")
            device = 'cpu'

    model = train(args, num_epochs, model, criterion, optimizer, scheduler,
                  train_loader, args.train_batch_size, train_size,
                  val_loader, args.val_batch_size, val_size, device)

    test_error = test(model, test_loader, test_metric, device, output_norm_scaling=args.output_norm_scaling,
                      output_norm_value=args.output_norm_value)
    print(f"Model on Test dataset - Error: {test_error}")

    if not args.debug:
        wandb.log({
            "test_error": test_error
        })

        wandb.finish()

    save_model(model, save_path)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of epochs to train')
    parser.add_argument('--device', type=str, choices=[
                        'cpu', 'cuda'], default='cuda', help='Which device to run the models on')
    parser.add_argument('--name', type=str,
                        default='cnn-model', help='Name of the run')
    parser.add_argument('--save_path', type=str,
                        default='./ckpt', help='Path to save the final model')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to the root dir of the dataset')
    parser.add_argument('--train_batch_size', type=int,
                        default=32, help='Train Batch Size')
    parser.add_argument('--val_batch_size', type=int,
                        default=32, help='Validation Batch Size')
    parser.add_argument('--test_batch_size', type=int,
                        default=32, help='Test Batch Size')
    parser.add_argument('--num_files', type=int, default=7,
                        help='Number of dataset files to load')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Ratio of the dataset to take as the test set')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Ratio of the dataset to take as the validation set')
    parser.add_argument('--pretrained', action='store_true',
                        help='Whether to use the pretrained network')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='The learning rate of the model')
    parser.add_argument('--lr_step', type=int, default=5,
                        help='The number of steps to reduce the LR of the optimizer')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='The factor by which to reduce the LR of the optimizer')
    parser.add_argument('--criterion_type', type=str, default='mse', choices=[
                        'mse', 'l2', 'l1', 'smoothl1'], help='Which criterion to use for training')
    parser.add_argument('--criterion_beta', type=float, default=20,
                        help='Beta for the specific criterion if applicable')
    parser.add_argument('--use_pe', action='store_true',
                        help='Whether to use Positional Encoding')
    parser.add_argument('--num_pe_scales', type=int,
                        default=5, help='Number of PE scales to use')
    parser.add_argument('--min_threshold', type=float, default=1e-3,
                        help='The min threshold for the zero suppression')
    parser.add_argument('--output_mean_scaling', action='store_true',
                        help='Whether to perform mean scaling on the output')
    parser.add_argument('--output_mean_value', type=float,
                        default=293.2899, help='The mean to subtract from the mean')
    parser.add_argument('--output_norm_scaling', action='store_true',
                        help='Whether to divide the output by normalizing constant')
    parser.add_argument('--output_norm_value', type=float, default=119.904,
                        help='The the normalizing constant to divide the output by')
    parser.add_argument('--model', type=str, default='gat', choices=['dgcnn', 'gat', 'pna'], help='The backbone GNN to use')
    parser.add_argument('--point_fn', type=str, default='total', choices=['total', 'channel_wise'], help='How to obtain points from the image')
    parser.add_argument('--plot', action='store_true', help='Whether to plot the predicted vs ground truth results')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    train_dset, val_dset, test_dset, train_size, val_size, test_size = get_datasets(
        args.data_dir,
        args.num_files,
        args.test_ratio,
        args.val_ratio,
        point_fn=args.point_fn,
        use_pe=args.use_pe,
        pe_scales=args.num_pe_scales,
        min_threshold=args.min_threshold,
        output_mean_scaling=args.output_mean_scaling,
        output_mean_value=args.output_mean_value,
        output_norm_scaling=args.output_norm_scaling,
        output_norm_value=args.output_norm_value
    )

    train_loader, val_loader, test_loader = get_loaders(
        train_dset, val_dset, test_dset, args.train_batch_size, args.val_batch_size, args.test_batch_size)

    model = get_model(args.device, model=args.model, train_loader=train_loader, point_fn=args.point_fn, pretrained=args.pretrained,
                      use_pe=args.use_pe, pe_scales=args.num_pe_scales)
    optimizer, scheduler = get_optimizer(
        model, args.lr, args.lr_step, args.lr_gamma)

    criterion = get_criterion(args.criterion_type, beta=args.criterion_beta)
    test_metric = get_test_metric()

    model = main(
        args=args,
        run_name=args.name,
        num_epochs=args.num_epochs,
        model=model,
        criterion=criterion,
        test_metric=test_metric,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        device=args.device,
        save_path=args.save_path
    )
