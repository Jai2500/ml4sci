import torch_geometric
import torch
from tqdm.auto import tqdm

class MLPStack(torch.nn.Module):
    '''
        A simple MLP stack that stacks multiple linear-bn-act layers
    '''
    def __init__(self, layers, bn=True, act=True):
        super().__init__()
        assert len(layers) > 1, "At least input and output channels must be provided"

        modules = []
        for i in range(1, len(layers)):
            modules.append(
                torch.nn.Linear(layers[i-1], layers[i])
            )
            modules.append(
                torch.nn.BatchNorm1d(layers[i]) if bn == True else torch.nn.Identity()
            )
            modules.append(
                torch.nn.SiLU() if bn == True else torch.nn.Identity()
            )

        self.mlp_stack = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.mlp_stack(x)

class DynamicEdgeConvPN(torch.nn.Module):
    '''
        Internal convolution DynamicEdgeConv block inspired from ParticleNet
    '''
    def __init__(self, edge_nn, nn, k=7, aggr='max', flow='source_to_target') -> None:
        super().__init__()
        self.nn = nn
        self.k = k
        self.edge_conv = torch_geometric.nn.EdgeConv(nn=edge_nn, aggr=aggr)
        self.flow = flow

    def forward(self, x, pos, batch):
        edge_index = torch_geometric.nn.knn_graph(x=pos, k=self.k, batch=batch, flow=self.flow)

        edge_out = self.edge_conv(x, edge_index)

        x_out = self.nn(x)

        return edge_out + x_out


class DGCNN(torch.nn.Module):
    '''
        DGCNN network that is similar to the ParticleNet architecture
    '''
    def __init__(self, x_size, pos_size, k=7, use_pe=False, pe_scales=0):
        super().__init__()
        self.dynamic_conv_1 = DynamicEdgeConvPN(
            edge_nn=MLPStack(
                [x_size * 2 if not use_pe else x_size * 2 * (pe_scales * 2 + 1), 32, 32, 32], bn=True, act=True
            ),
            nn=MLPStack(
                [x_size, 32, 32, 32], bn=True, act=True
            ),
            k=k
        )

        self.dynamic_conv_2 = DynamicEdgeConvPN(
            edge_nn=MLPStack(
                [64, 64, 64, 128], bn=True, act=True
            ),
            nn=MLPStack(
                [32, 64, 64, 128], bn=True, act=True
            ),
            k=k
        )

    def forward(self, data):
        x = data.x
        pos = data.pos
        batch = data.batch

        x_out = self.dynamic_conv_1(
            x, pos, batch
        )
        x_out = self.dynamic_conv_2(
            x_out, x_out, batch
        )

        x_out = torch_geometric.nn.global_mean_pool(x_out, batch)

        return x_out

def compute_degree(train_dset, k=7):
    max_degree = -1
    for data in tqdm(train_dset, desc='Max Degree'):
        edge_index = torch_geometric.nn.knn_graph(data.pos, k=k, num_workers=1)
        d = torch_geometric.utils.degree(edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, d.max())
    
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in tqdm(train_dset, desc='Degree Distribution'):
        edge_index = torch_geometric.nn.knn_graph(data.pos, k=k, num_workers=1)
        d = torch_geometric.utils.degree(edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=d.numel())

    return deg

class PNANet(torch.nn.Module):
    def __init__(self, x_size, pos_size, deg, k=7, use_pe=False, pe_scales=0):
        super().__init__()
        self.k = k
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.pna_conv_1 = torch_geometric.nn.PNAConv(
            in_channels=x_size if not use_pe else x_size * (pe_scales * 2 + 1),
            out_channels=64,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            towers=5,
            pre_layers=1,
            post_layers=1,
            divide_input=False
        )
        self.bn1 = torch_geometric.nn.BatchNorm(64)

        self.pna_conv_2 = torch_geometric.nn.PNAConv(
            in_channels=64,
            out_channels=128,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            towers=5,
            pre_layers=1,
            post_layers=1,
            divide_input=False
        )
        self.bn2 = torch_geometric.nn.BatchNorm(128)

        self.act = torch.nn.ReLU()

    def forward(self, data):
        pos = data.pos
        batch = data.batch
        x = data.x

        edge_index = torch_geometric.nn.knn_graph(x=pos, k=self.k, batch=batch)

        x_out = self.act(self.bn1(self.pna_conv_1(x, edge_index)))
        x_out = self.act(self.bn2(self.pna_conv_2(x_out, edge_index)))

        x_out = torch_geometric.nn.global_mean_pool(x_out, batch) # Consider global add pool
        
        return x_out


class SimpleGAT(torch.nn.Module):
    '''
        Simple 2 layered GAT GNN
    '''
    def __init__(self, x_size, pos_size, k=7, use_pe=False, pe_scales=0):
        super().__init__()
        self.k = k
        self.gat_conv_1 = torch_geometric.nn.GATv2Conv(
            in_channels=x_size if not use_pe else x_size * (pe_scales * 2 + 1),
            out_channels=16,
            heads=4
        )
        self.gat_conv_2 = torch_geometric.nn.GATv2Conv(
            in_channels=16 * 4,
            out_channels=32,
            heads=4
        )
        self.act = torch.nn.ReLU()

    def forward(self, data):
        pos = data.pos
        batch = data.batch
        x = data.x

        edge_index = torch_geometric.nn.knn_graph(x=pos, k=self.k, batch=batch)

        x_out = self.act(self.gat_conv_1(
            x, edge_index))
        x_out = self.act(self.gat_conv_2(
            x_out, edge_index))

        x_out = torch_geometric.nn.global_mean_pool(x_out, batch)

        return x_out


class RegressModel(torch.nn.Module):
    '''
        Model to perform the regression on the data. 
        Builds a small MLP network on a provided backbone network.
    '''
    def __init__(self, model, in_features, use_pe=False, pe_scales=0):
        '''
            Init fn. of the RegressModel.
            Args:
                model: The backbone model to operate on the images
                in_features: The size of the output of the backbone
                use_pe: Whether positional encoding is being used
                pe_scales: The scales of the positional encoding
        '''
        super().__init__()
        self.model = model


        self.out_mlp = MLPStack(
            [in_features + 3, in_features * 2, in_features * 2, in_features, in_features // 2],
            bn=True, act=True
        )

        self.out_lin = torch.nn.Linear(in_features//2, 1)

    def forward(self, data):
        out = self.model(data)
        out = torch.cat(
            [out, data.pt.unsqueeze(-1), data.ieta.unsqueeze(-1), data.iphi.unsqueeze(-1)], dim=1
        )
        return self.out_lin(self.out_mlp(out))


def get_model(device, model, point_fn, train_dset, pretrained=False, use_pe=False, pe_scales=0):
    '''
        Returns the model based on the arguments
        Args:
            device: The device to run the model on
            model: The backbone model choice
            pretrained: Whether to use the pretrained backbone
            use_pe: Whether positional encoding is being used
            pe_scales: The scale of the positional encoding

        Returns:
            regress_model: Model that is used to perform regression
    '''
    if point_fn == 'total':
        x_size = 10
        pos_size = 2
    elif point_fn == 'channel_wise':
        x_size = 11
        pos_size = 10
    else:
        raise NotImplementedError()

    if model == 'dgcnn':
        input_model = DGCNN(x_size=x_size, pos_size=pos_size, use_pe=use_pe, pe_scales=pe_scales)
    elif model == 'gat':
        input_model = SimpleGAT(x_size=x_size, pos_size=pos_size,use_pe=use_pe, pe_scales=pe_scales)
    elif model == 'pna':
        deg = compute_degree(train_dset)
        input_model = PNANet(x_size, pos_size, deg, use_pe=use_pe, pe_scales=pe_scales)
    else:
        raise NotImplementedError(f"Model type {model} not implemented")
    regress_model = RegressModel(
        model=input_model,
        in_features=128,
        use_pe=use_pe,
        pe_scales=pe_scales
    )

    regress_model = regress_model.to(device)

    return regress_model