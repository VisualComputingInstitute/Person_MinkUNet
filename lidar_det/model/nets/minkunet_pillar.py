import torch
import torch.nn as nn

import torchsparse
import torchsparse.nn as spnn
import torchsparse.nn.functional as spf

__all__ = ["MinkPillarUNet"]


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class PillarBlock(nn.Module):
    def __init__(self, channel, pdim=2):
        super().__init__()
        self.pdim = pdim
        l = []
        for i in range(2):
            l.append(
                nn.Sequential(
                    nn.Linear(channel, channel), nn.BatchNorm1d(channel), nn.ReLU(True),
                ),
            )
        self.mlp = nn.Sequential(*l)

    def forward(self, x):
        # average pooling over all voxels within a pillar
        # https://github.com/mit-han-lab/torchsparse/blob/master/torchsparse/nn/modules/conv.py#L89 # noqa
        coords, feats = x.C.clone(), x.F
        coords[:, self.pdim] = 0
        coords -= coords.min(dim=0, keepdim=True)[0]
        feats = torch.cat([torch.ones_like(feats[:, :1]), feats], axis=1)
        sp_tensor = torch.cuda.sparse.FloatTensor(coords.t().long(), feats).coalesce()
        coords_coalesced = sp_tensor.indices().t().int()
        feats_coalesced = sp_tensor.values()[:, 1:] / sp_tensor.values()[:, :1].detach()

        feats_coalesced = self.mlp(feats_coalesced)

        # add pillar feature back to each voxel
        # https://github.com/mit-han-lab/e3d/blob/d58b12877d73f812b3f0b99ee77b72c4aad7e8da/spvnas/core/models/utils.py#L49 # noqa
        hash_coalesced = spf.sphash(coords_coalesced)
        hash_raw = spf.sphash(coords)
        idx = spf.sphashquery(hash_raw, hash_coalesced)
        x.F = x.F + feats_coalesced[idx]

        return x


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, stride=stride, transpose=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation, stride=1),
            spnn.BatchNorm(outc),
        )

        self.downsample = (
            nn.Sequential()
            if (inc == outc and stride == 1)
            else nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc),
            )
        )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class MinkPillarUNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get("cr", 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.run_up = kwargs.get("run_up", True)

        input_dim = kwargs.get("input_dim", 3)

        self.stem = nn.Sequential(
            spnn.Conv3d(input_dim, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),
            # PillarBlock(cs[0]),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            PillarBlock(cs[0]),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            # PillarBlock(cs[1]),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            PillarBlock(cs[1]),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            # PillarBlock(cs[2]),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1),
            PillarBlock(cs[2]),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1),
            # PillarBlock(cs[3]),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1),
            PillarBlock(cs[3]),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1),
            # PillarBlock(cs[4]),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1, dilation=1),
                    ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1),
                ),
            ]
        )

        self.up2 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1, dilation=1),
                    ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1),
                ),
            ]
        )

        self.up3 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1, dilation=1),
                    ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1),
                ),
            ]
        )

        self.up4 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1, dilation=1),
                    ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1),
                ),
            ]
        )

        self.classifier = nn.Sequential(nn.Linear(cs[8], kwargs["num_classes"]))

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        y1 = self.up1[0](x4)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)

        out = self.classifier(y4.F)

        return out
