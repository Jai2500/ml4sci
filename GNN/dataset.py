import pyarrow.parquet as pq
import numpy as np
import torch 
import torch_geometric
import glob
import os
from tqdm.auto import tqdm
from dataset_utils import normalize_x, positional_encoding

class PointCloudFromParquetDataset(torch.utils.data.Dataset):
    '''
        Dataset to extract Point Cloud from the Parquet File. This does not load
        the entire Parquet Dataset on memory but reads from it every time
        an image is queried.
    '''
    def __init__(
        self,
        filename,
        suppresion_thresh=0,
        use_pe=False,
        pe_scales=0,
        output_mean_scaling=False,
        output_mean_value=None,
        output_norm_scaling=False,
        output_norm_value=None
    ) -> None:
        '''
            Init fn. of the dataset
            Args:
                filename: The path to the parquet file
                suppression_thresh: The minimum threshold for converting to point cloud 
                use_pe: Whether to use sin/cos positional encoding on the features
                pe_scales: The scales for the positional encoding
                output_mean_scaling: Whether to subtract the ground truth with a mean value
                output_mean_value: The mean value to subtract the ground truth with
                output_norm_scaling: Whether to scale the ground truth with a norm value
                output_norm_value: The norm value to scale the ground truth with

            Returns:
                None
        '''
        super().__init__()

        self.file = pq.ParquetFile(filename)
        self.suppression_thresh = suppresion_thresh

        self.use_pe = use_pe
        self.pe_scales = pe_scales
        self.output_mean_scaling = output_mean_scaling
        self.output_mean_value = output_mean_value
        self.output_norm_scaling = output_norm_scaling
        self.output_norm_value = output_norm_value

    def __getitem__(self, idx, ):
        '''
            __getitem__ function of a Pytorch dataset. 
            Returns the traning element. 
        '''
        row = self.file.read_row_group(idx).to_pydict()
        
        arr = np.array(row['X_jet'][0])
        idx = np.where(arr.sum(axis=0) > self.suppression_thresh)
        pos = np.array(idx).T / arr.shape[1]
        x = arr[:, idx[0], idx[1]].T
        x = normalize_x(x)
        x = np.concatenate([x, pos], axis=1)
        
        pt = row['pt'][0]
        ieta = row['ieta'][0]
        iphi = row['iphi'][0]
        m = row['m'][0]

        if self.output_mean_scaling:
            m = m - self.output_mean_value
        
        if self.output_norm_scaling:
            m  = m / self.output_norm_value

        x = torch.as_tensor(x) if not self.use_pe else positional_encoding(x, self.pe_scales)

        data = torch_geometric.data.Data(
            pos=torch.as_tensor(pos).float(),
            x=x.float(),
            pt=torch.as_tensor(pt).unsqueeze(-1),
            ieta=torch.as_tensor(ieta).unsqueeze(-1),
            iphi=torch.as_tensor(iphi).unsqueeze(-1),
            y=torch.as_tensor(m),
        )

        return data

    def __len__(self):
        return self.file.num_row_groups


def get_datasets(
    root_dir,
    num_files,
    test_ratio,
    val_ratio,
    use_pe=False,
    pe_scales=0,
    min_threshold=0.,
    output_mean_scaling=False,
    output_mean_value=0,
    output_norm_scaling=False,
    output_norm_value=1.,
):
    '''
        Returns the datasets provided the root directory of the multiple parquet files.
        Args:
            root_dir: The root directory containing all the parquet files
            num_files: The number of files to be read
            test_ratio: The ratio of the dataset to be used as the test dataset
            val_ratio: The ratio of the dataset to be used as the validation dataset
            use_pe: Whether to use sin/cos positional encoding
            pe_scales: The scales for the positional encoding
            min_threshold: The minimum threshold for the zero suppression
            output_mean_scaling: Whether to subtract the ground truth with a mean value
            output_mean_value: The mean value to subtract from the ground truth
            output_norm_scaling: Whether to scale the ground truth with a norm value
            output_norm_value: The value with which to scale the ground truth

        Returns:
            train_dset: The training dataset
            val_dset: The validation dataset
            test_dset: The test dataset
            train_size: The size of the training dataset
            val_size: The size of the validation dataset
            test_size: The size of the test dataset
    '''
    paths = list(glob.glob(os.path.join(root_dir, "*.parquet")))

    dsets = []
    for path in tqdm(paths[0:num_files]):
        dsets.append(
            PointCloudFromParquetDataset(
                path,
                use_pe=use_pe, pe_scales=pe_scales,
                suppresion_thresh=min_threshold,
                output_mean_scaling=output_mean_scaling, output_mean_value=output_mean_value,
                output_norm_scaling=output_norm_scaling, output_norm_value=output_norm_value
            )
        )

    combined_dset = torch.utils.data.ConcatDataset(dsets)

    val_size = int(len(combined_dset) * val_ratio)
    test_size = int(len(combined_dset) * test_ratio)
    train_size = len(combined_dset) - val_size - test_size

    train_dset, val_dset, test_dset = torch.utils.data.random_split(
        combined_dset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    return train_dset, val_dset, test_dset, train_size, val_size, test_size
