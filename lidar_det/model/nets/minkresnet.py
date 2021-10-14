import numpy as np
import torch
import torch.nn as nn
import torchsparse.nn as spnn

__all__ = ["MinkResNet"]


def _ASSERT_EQUAL(arr_a, arr_b, tolerance):
    assert (arr_a - arr_b).abs().max() <= tolerance


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


class ResNetBlock(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        midc = int(inc / 4)
        self.net = nn.Sequential(
            spnn.Conv3d(inc, midc, kernel_size=1),
            spnn.BatchNorm(midc),
            spnn.ReLU(True),
            spnn.Conv3d(midc, midc, kernel_size=3),
            spnn.BatchNorm(midc),
            spnn.ReLU(True),
            spnn.Conv3d(midc, outc, kernel_size=1),
            spnn.BatchNorm(outc),
        )
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + x)
        return out


class MinkResNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get("cr", 1.0)
        cs = [64, 128, 256, 512, 512]  # channel size
        cs = [int(cr * x) for x in cs]
        br = [3, 4, 6, 3]  # blocks repeat

        input_dim = kwargs.get("input_dim", 3)

        self.stem = nn.Sequential(
            spnn.Conv3d(input_dim, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),
        )

        blocks = []
        blocks_ds = []
        for i in range(len(cs) - 1):
            blocks_ds.append(
                BasicConvolutionBlock(cs[i], cs[i], ks=2, stride=2, dilation=1),
            )
            blocks.append(
                nn.Sequential(
                    *[
                        ResidualBlock(cs[i], cs[i], ks=3, stride=1, dilation=1)
                        for _ in range(br[i] - 1)
                    ],
                    ResidualBlock(cs[i], cs[i + 1], ks=3, stride=1, dilation=1)
                )
            )

        self.blocks = nn.ModuleList(blocks)
        self.blocks_ds = nn.ModuleList(blocks_ds)

        self._fpn = kwargs.get("fpn", False)
        if self._fpn:
            self.classifier = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(cs[i + 1], kwargs["num_classes"]))
                    for i in range(len(cs) - 1)
                ]
            )
        else:
            self.classifier = nn.Sequential(nn.Linear(cs[-1], kwargs["num_classes"]))

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.stem(x)

        inds_maps = []
        voxel_centers = []
        x_fpn = [x0]
        for i in range(len(self.blocks)):
            x_ds = self.blocks_ds[i](x_fpn[-1])
            x_fpn.append(self.blocks[i](x_ds))

            # get a mapping from downsampled voxels to the original
            # NOTE only downsampling changes the order of voxels
            kernel_map = (
                x_ds.kernel_maps["k2_os%d_s2_d1" % 2 ** i][0].data.cpu().numpy()
            )
            _, uq_inds = np.unique(kernel_map[:, 1], return_index=True)
            inds_map = kernel_map[uq_inds]  # NOTE inds_map[:, 1] is sorted by np.unique
            if len(inds_maps) > 0:
                # kernel map gives mapping to the previous stage, we want mapping
                # to the original voxels
                inds_map[:, 0] = inds_maps[-1][inds_map[:, 0], 0]
            inds_maps.append(inds_map)

            C_center = x_ds.C[:, :3].clone().float()
            C_center += 0.5 * x_ds.s
            voxel_centers.append(C_center)

            # # for debug
            # km = kernel_map[uq_inds]
            # _ASSERT_EQUAL(x_fpn[-2].C[km[:, 0]], x_fpn[-1].C, 2 ** i)
            # _ASSERT_EQUAL(x.C[inds_maps[-1][:, 0]], x_fpn[-1].C, 2 ** (i + 1) - 1)
            # print((x.C[inds_maps[-1][:, 0]] - x_fpn[-1].C).abs().max())

        if self._fpn:
            out = torch.cat(
                [
                    self.classifier[i](x_fpn[i + 1].F)
                    for i in range(len(self.classifier))
                ],
                dim=0,
            )

            inds_map = np.concatenate([imap[:, 0] for imap in inds_maps])
            voxel_center = torch.cat(voxel_centers, dim=0)

            return out, inds_map, voxel_center
        else:
            out = self.classifier(x_fpn[-1].F)
            inds_map = inds_maps[-1][:, 0]
            voxel_center = voxel_centers[-1]

            return out, inds_map, voxel_center
