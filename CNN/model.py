import torch
import torchvision

class RegressModel(torch.nn.Module):
    def __init__(self, model, use_pe=False, pe_scales=0):
        super().__init__()
        self.model = model
        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Identity()

        self.out_lin = torch.nn.Sequential(
            torch.nn.Linear(in_features + 3 if not use_pe else in_features + 3 * (pe_scales * 2 + 1), in_features // 2, bias=True),
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
            [out, pt, ieta, iphi], dim=1
        )
        return self.out_lin(out)


def get_model(device, pretrained=False, use_pe=False, pe_scales=0):
    regress_model = RegressModel(
        model=torchvision.models.resnet34(pretrained=pretrained),
        use_pe=use_pe,
        pe_scales=pe_scales
    )
    regress_model.model.conv1 = torch.nn.Conv2d(
        8 if not use_pe else 8 * (pe_scales * 2 + 1), 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    regress_model = regress_model.to(device)

    return regress_model