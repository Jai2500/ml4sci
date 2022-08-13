import torch_geometric
import torch

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
    def __init__(self, k=7):
        super().__init__()
        self.dynamic_conv_1 = DynamicEdgeConvPN(
            edge_nn=MLPStack(
                [20, 32, 32, 32], bn=True, act=True
            ),
            nn=MLPStack(
                [10, 32, 32, 32], bn=True, act=True
            ),
            k=k
        )

        self.dynamic_conv_2 = DynamicEdgeConvPN(
            edge_nn=MLPStack(
                [64, 64, 64, 64], bn=True, act=True
            ),
            nn=MLPStack(
                [32, 64, 64, 64], bn=True, act=True
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

class SimpleGAT(torch.nn.Module):
    '''
        Simple 2 layered GAT GNN
    '''
    def __init__(self, k=7, use_pe=False, pe_scales=0):
        super().__init__()
        self.k = k
        self.gat_conv_1 = torch_geometric.nn.GATv2Conv(
            in_channels=10 if not use_pe else 10 * (pe_scales * 2 + 1),
            out_channels=16,
            heads=4
        )
        self.gat_conv_2 = torch_geometric.nn.GATv2Conv(
            in_channels=16 * 4,
            out_channels=32,
            heads=4
        )

    def forward(self, data):
        pos = data.pos
        batch = data.batch
        x = data.x

        edge_index = torch_geometric.nn.knn_graph(x=pos, k=self.k, batch=batch)

        x_out = self.gat_conv_1(
            x, edge_index
        )
        x_out = self.gat_conv_2(
            x_out, edge_index
        )

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

    def forward(self, data):
        out = self.model(data)
        out = torch.cat(
            [out, data.pt.unsqueeze(-1), data.ieta.unsqueeze(-1), data.iphi.unsqueeze(-1)], dim=1
        )
        return self.out_lin(out)


def get_model(device, model, pretrained=False, use_pe=False, pe_scales=0):
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
    input_model = DGCNN() if model == 'dgcnn' else SimpleGAT(use_pe=use_pe, pe_scales=pe_scales)
    regress_model = RegressModel(
        model=input_model,
        in_features=128,
        use_pe=use_pe,
        pe_scales=pe_scales
    )

    regress_model = regress_model.to(device)

    return regress_model