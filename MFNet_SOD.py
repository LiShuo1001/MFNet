import os
import cv2
import glob
import torch
import random
import skimage
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from skimage import segmentation
from alisuretool.Tools import Tools
import torchvision.transforms as transforms
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, SAGEConv


"""
torch==1.4.0+cu100
pip uninstall torch-sparse
pip install torch-scatter==latest+cu100 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+cu100 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==latest+cu100 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+cu100 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric=1.4.3
"""


def gpu_setup(use_gpu, gpu_id):
    if torch.cuda.is_available() and use_gpu:
        Tools.print()
        Tools.print('Cuda available with GPU: {}'.format(torch.cuda.get_device_name(0)))
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device("cuda:{}".format(gpu_id))
    else:
        Tools.print()
        Tools.print('Cuda not available')
        device = torch.device("cpu")
    return device


class DealSuperPixel(object):

    def __init__(self, image_data, label_data, super_pixel_size=14, slic_sigma=1, slic_max_iter=5):
        image_size = image_data.shape[0: 2]
        self.super_pixel_num = (image_size[0] * image_size[1]) // (super_pixel_size * super_pixel_size)
        self.image_data = image_data
        self.label_data = label_data
        try:
            self.segment = segmentation.slic(self.image_data, n_segments=self.super_pixel_num,
                                             sigma=slic_sigma, max_iter=slic_max_iter, start_label=0)
        except TypeError:
            self.segment = segmentation.slic(self.image_data, n_segments=self.super_pixel_num,
                                             sigma=slic_sigma, max_iter=slic_max_iter)
            pass

        _measure_region_props = skimage.measure.regionprops(self.segment + 1)
        self.region_props = [[region_props.centroid, region_props.coords] for region_props in _measure_region_props]
        pass

    def run(self):
        edge_index, sp_label, pixel_adj = [], [], []
        for i in range(self.segment.max() + 1):
            where = self.segment == i
            # 计算标签
            label = np.mean(self.label_data[where])
            sp_label.append(label)

            # 计算邻接矩阵
            _now_adj = skimage.morphology.dilation(where, selem=skimage.morphology.square(3))
            edge_index.extend([[i, sp_id] for sp_id in np.unique(self.segment[_now_adj]) if sp_id != i])

            # 计算单个超像素中的邻接矩阵
            _now_where = self.region_props[i][1]
            pixel_data_where = np.concatenate([[[0]] * len(_now_where), _now_where], axis=-1)
            _a = np.tile([_now_where], (len(_now_where), 1, 1))
            _dis = np.sum(np.power(_a - np.transpose(_a, (1, 0, 2)), 2), axis=-1)
            _dis[_dis == 0] = 111
            pixel_edge_index = np.argwhere(_dis <= 2)
            pixel_edge_w = np.ones(len(pixel_edge_index))
            pixel_adj.append([pixel_data_where, pixel_edge_index, pixel_edge_w, label])
            pass

        sp_adj = np.asarray(edge_index)
        sp_label = np.asarray(sp_label)
        return self.segment, sp_adj, pixel_adj, sp_label

    pass


class FixedResize(object):

    def __init__(self, size):
        self.size = (size, size)
        pass

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        return {'image': img, 'label': mask}

    pass


class RandomHorizontalFlip(object):

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return {'image': img, 'label': mask}

    pass


class RandomCrop(transforms.RandomCrop):

    def __init__(self, size):
        self.size = (int(size), int(size))
        pass

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        i, j, h, w = self.get_params(img, self.size)
        img = transforms.functional.crop(img, i, j, h, w)
        mask = transforms.functional.crop(mask, i, j, h, w)
        return {'image': img, 'label': mask}

    pass


class MyDataset(Dataset):

    def __init__(self, data_root_path, down_ratio=4, down_ratio2=1, is_train=True, sp_size=4, min_size=256):
        super().__init__()
        self.sp_size = sp_size
        self.is_train = is_train
        self.down_ratio_for_sp = down_ratio
        self.down_ratio_for_sod = down_ratio2
        self.data_root_path = data_root_path
        self.min_size = min_size

        # 路径
        self.data_image_path = os.path.join(data_root_path, "DUTS-TR" if self.is_train else "DUTS-TE",
                                            "DUTS-TR-Image" if self.is_train else "DUTS-TE-Image")
        self.data_label_path = os.path.join(data_root_path, "DUTS-TR" if self.is_train else "DUTS-TE",
                                            "DUTS-TR-Mask" if self.is_train else "DUTS-TE-Mask")

        # 数据增强
        self.transform_train = transforms.Compose([RandomHorizontalFlip()])

        # 准备数据
        self.image_name_list, self.label_name_list = self.get_image_label_name()
        pass

    def get_image_label_name(self):
        tra_img_name_list = glob.glob(os.path.join(self.data_image_path, '*.jpg'))
        tra_lbl_name_list = [os.path.join(self.data_label_path, '{}.png'.format(
            os.path.splitext(os.path.basename(img_path))[0])) for img_path in tra_img_name_list]
        return tra_img_name_list, tra_lbl_name_list

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        # 读数据
        label = Image.open(self.label_name_list[idx])
        image = Image.open(self.image_name_list[idx]).convert("RGB")

        image_name = self.image_name_list[idx]
        if image.size == label.size:
            # 限制最小大小
            if image.size[0] < self.min_size or image.size[1] < self.min_size:
                if image.size[0] < image.size[1]:
                    image = image.resize((self.min_size, int(self.min_size / image.size[0] * image.size[1])))
                    label = label.resize((self.min_size, int(self.min_size / image.size[0] * image.size[1])))
                else:
                    image = image.resize((int(self.min_size / image.size[1] * image.size[0]), self.min_size))
                    label = label.resize((int(self.min_size / image.size[1] * image.size[0]), self.min_size))
                pass

            w, h = label.size
            # 数据增强
            sample = {'image': image, 'label': label}
            sample = self.transform_train(sample) if self.is_train else sample
            image, label = sample['image'], sample['label']
            label_for_sod = np.asarray(label.resize((w // self.down_ratio_for_sod, h // self.down_ratio_for_sod))) / 255

            # 归一化
            _normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img_data = transforms.Compose([transforms.ToTensor(), _normalize])(image).unsqueeze(dim=0)

            # 超像素
            image_small_data = np.asarray(image.resize((w // self.down_ratio_for_sp, h // self.down_ratio_for_sp)))
            label_for_sp = np.asarray(label.resize((w // self.down_ratio_for_sp, h // self.down_ratio_for_sp))) / 255
            graph, pixel_graph, segment = self.get_sp_info(image_small_data, label_for_sp, sp_size=self.sp_size)
        else:
            Tools.print('IMAGE ERROR, PASSING {}'.format(image_name))
            graph, pixel_graph, img_data, label_for_sp, label_for_sod, segment, image_small_data, image_name = \
                self.__getitem__(np.random.randint(0, len(self.image_name_list)))
            pass
        # 返回
        return graph, pixel_graph, img_data, label_for_sp, label_for_sod, segment, image_small_data, image_name

    @staticmethod
    def get_sp_info(image, label, sp_size):
        # Super Pixel
        #################################################################################
        deal_super_pixel = DealSuperPixel(image_data=image, label_data=label, super_pixel_size=sp_size)
        segment, sp_adj, pixel_adj, sp_label = deal_super_pixel.run()
        #################################################################################
        # Graph
        #################################################################################
        graph = Data(edge_index=torch.from_numpy(np.transpose(sp_adj, axes=(1, 0))),
                     num_nodes=len(pixel_adj), y=torch.from_numpy(sp_label).float(), num_sp=len(pixel_adj))
        #################################################################################
        # Small Graph
        #################################################################################
        pixel_graph = []
        for super_pixel in pixel_adj:
            small_graph = Data(edge_index=torch.from_numpy(np.transpose(super_pixel[1], axes=(1, 0))),
                               data_where=torch.from_numpy(super_pixel[0]).long(),
                               num_nodes=len(super_pixel[0]), y=torch.tensor([super_pixel[3]]),
                               edge_w=torch.from_numpy(super_pixel[2]).unsqueeze(1).float())
            pixel_graph.append(small_graph)
            pass
        #################################################################################
        return graph, pixel_graph, segment

    @staticmethod
    def collate_fn(samples):
        graphs, pixel_graphs, images, labels_sp, labels_sod, segments, images_small, image_name = map(list,
                                                                                                      zip(*samples))

        images = torch.cat(images)
        images_small = torch.tensor(images_small)

        # 超像素图
        batched_graph = Batch.from_data_list(graphs)

        # 像素图
        _pixel_graphs = []
        for super_pixel_i, pixel_graph in enumerate(pixel_graphs):
            for now_graph in pixel_graph:
                now_graph.data_where[:, 0] = super_pixel_i
                _pixel_graphs.append(now_graph)
            pass
        batched_pixel_graph = Batch.from_data_list(_pixel_graphs)

        return images, labels_sp, labels_sod, batched_graph, batched_pixel_graph, segments, images_small, image_name

    pass


class ConvBlock(nn.Module):

    def __init__(self, cin, cout, stride=1, ks=3, has_relu=True, has_bn=False, bias=True):
        super().__init__()
        self.has_relu = has_relu
        self.has_bn = has_bn

        self.conv = nn.Conv2d(cin, cout, kernel_size=ks, stride=stride, padding=ks // 2, bias=bias)
        if self.has_bn:
            self.bn = nn.BatchNorm2d(cout)
        if self.has_relu:
            self.relu = nn.ReLU(inplace=True)
        pass

    def forward(self, x):
        out = self.conv(x)
        if self.has_bn:
            out = self.bn(out)
        if self.has_relu:
            out = self.relu(out)
        return out

    pass


class SAGENet1(nn.Module):

    def __init__(self, in_dim=128, hidden_dims=[128, 128, 128, 128],
                 has_bn=False, normalize=False, residual=False, concat=False):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.residual = residual
        self.normalize = normalize
        self.has_bn = has_bn
        self.concat = concat
        self.out_num = self.hidden_dims[-1]

        # self.embedding_h = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU()

        _in_dim = in_dim
        self.gcn_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        for hidden_dim in self.hidden_dims:
            self.gcn_list.append(SAGEConv(_in_dim, hidden_dim, normalize=self.normalize, concat=self.concat))
            self.bn_list.append(nn.BatchNorm1d(hidden_dim))
            _in_dim = hidden_dim
            pass
        pass

    def forward(self, data):
        # hidden_nodes_feat = self.embedding_h(data.x)
        hidden_nodes_feat = data.x
        for gcn, bn in zip(self.gcn_list, self.bn_list):
            h_in = hidden_nodes_feat

            # Conv
            hidden_nodes_feat = gcn(h_in, data.edge_index)
            if self.has_bn:
                hidden_nodes_feat = bn(hidden_nodes_feat)
            hidden_nodes_feat = self.relu(hidden_nodes_feat)

            # Res
            if self.residual and h_in.size()[-1] == hidden_nodes_feat.size()[-1]:
                hidden_nodes_feat = h_in + hidden_nodes_feat
            pass

        hg = global_mean_pool(hidden_nodes_feat, data.batch)
        return hg

    pass


class SAGENet2(nn.Module):

    def __init__(self, in_dim=128, hidden_dims=[128, 128, 128, 128], skip_which=[1, 2, 3],
                 skip_dim=128, sout=1, has_bn=False, normalize=False, residual=False, concat=False):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.normalize = normalize
        self.residual = residual
        self.has_bn = has_bn
        self.concat = concat
        self.out_num = len(skip_which) * skip_dim

        self.embedding_h = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU()

        _in_dim = in_dim
        self.gcn_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        for hidden_dim in self.hidden_dims:
            self.gcn_list.append(SAGEConv(_in_dim, hidden_dim, normalize=self.normalize, concat=self.concat))
            self.bn_list.append(nn.BatchNorm1d(hidden_dim))
            _in_dim = hidden_dim
            pass

        # skip
        self.skip_connect_index = skip_which
        self.skip_connect_list = nn.ModuleList()
        self.skip_connect_bn_list = nn.ModuleList()
        for hidden_dim in [self.hidden_dims[which - 1] for which in skip_which]:
            self.skip_connect_list.append(nn.Linear(hidden_dim, skip_dim, bias=False))
            self.skip_connect_bn_list.append(nn.BatchNorm1d(skip_dim))
            pass

        self.readout_mlp = nn.Linear(self.out_num, sout, bias=False)
        pass

    def forward(self, data):
        hidden_nodes_feat = self.embedding_h(data.x)

        gcn_hidden_nodes_feat = [hidden_nodes_feat]
        for gcn, bn in zip(self.gcn_list, self.bn_list):
            h_in = hidden_nodes_feat

            # Conv
            hidden_nodes_feat = gcn(h_in, data.edge_index)
            if self.has_bn:
                hidden_nodes_feat = bn(hidden_nodes_feat)
            hidden_nodes_feat = self.relu(hidden_nodes_feat)

            # Res
            if self.residual and h_in.size()[-1] == hidden_nodes_feat.size()[-1]:
                hidden_nodes_feat = h_in + hidden_nodes_feat

            gcn_hidden_nodes_feat.append(hidden_nodes_feat)
            pass

        skip_connect = []
        for sc, index, bn in zip(self.skip_connect_list, self.skip_connect_index, self.skip_connect_bn_list):
            # Conv
            sc_feat = sc(gcn_hidden_nodes_feat[index])
            if self.has_bn:
                sc_feat = bn(sc_feat)
            sc_feat = self.relu(sc_feat)

            skip_connect.append(sc_feat)
            pass

        out_feat = torch.cat(skip_connect, dim=1)
        logits = self.readout_mlp(out_feat).view(-1)
        return out_feat, logits, torch.sigmoid(logits)

    pass


class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.extract = [8, 15, 22, 29]  # [3, 8, 15, 22, 29]
        self.features = self.vgg(self.cfg)

        self.weight_init(self.modules())

        self.out_num1 = 128
        self.out_num2 = 256
        self.out_num3 = 512
        self.out_num4 = 512
        pass

    def forward(self, x):
        tmp_x = []
        for k in range(len(self.features)):
            x = self.features[k](x)
            if k in self.extract:
                tmp_x.append(x)
        return tmp_x

    @staticmethod
    def vgg(cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                # layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
            pass
        return nn.Sequential(*layers)

    @staticmethod
    def weight_init(modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            pass
        pass

    def load_pretrained_model(self, pretrained_model="./pretrained/vgg16-397923af.pth"):
        if pretrained_model is None:
            raise Exception("pretrained_model is None, please set pretrained_model!")
        self.load_state_dict(torch.load(pretrained_model), strict=False)
        pass

    pass


class DeepPoolLayer(nn.Module):

    def __init__(self, k, k_out, is_not_last, has_gcn=False, gcn_in=None):
        super(DeepPoolLayer, self).__init__()
        self.is_not_last = is_not_last
        self.has_gcn = has_gcn

        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.pool6 = nn.AvgPool2d(kernel_size=6, stride=6)
        self.pool8 = nn.AvgPool2d(kernel_size=8, stride=8)
        self.conv11 = nn.Conv2d(k, k // 4, 3, 1, 1, bias=False)
        self.conv21 = nn.Conv2d(k, k // 4, 3, 1, 1, bias=False)
        self.conv31 = nn.Conv2d(k, k // 4, 3, 1, 1, bias=False)
        self.conv41 = nn.Conv2d(k, k // 4, 3, 1, 1, bias=False)
        self.conv12 = nn.Conv2d(k // 4, k // 4, 3, 1, 1, bias=False)
        self.conv22 = nn.Conv2d(k // 4, k // 4, 3, 1, 1, bias=False)
        self.conv32 = nn.Conv2d(k // 4, k // 4, 3, 1, 1, bias=False)
        self.conv42 = nn.Conv2d(k // 4, k // 4, 3, 1, 1, bias=False)
        self.conv13 = nn.Conv2d(k, k // 4, 3, 1, 1, bias=False)
        self.conv23 = nn.Conv2d(k, k // 4, 3, 1, 1, bias=False)
        self.conv33 = nn.Conv2d(k, k // 4, 3, 1, 1, bias=False)
        self.conv43 = nn.Conv2d(k, k // 4, 3, 1, 1, bias=False)

        self.relu = nn.ReLU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)
        if self.has_gcn:
            self.conv_gcn = nn.Conv2d(gcn_in, k_out, 3, 1, 1, bias=False)
            self.conv_att = nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False)
        if self.is_not_last:
            self.conv_sum_c = nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False)
        pass

    def forward(self, x, x2=None, x_gcn=None):
        x_size = x.size()

        y1 = self.conv12(self.relu(self.conv11(self.pool2(x))))
        y2 = self.conv22(self.relu(self.conv21(self.pool4(x))))
        y3 = self.conv32(self.relu(self.conv31(self.pool6(x))))
        y4 = self.conv42(self.relu(self.conv41(self.pool8(x))))

        y1 = F.interpolate(y1, x_size[2:], mode='bilinear', align_corners=True)
        y2 = F.interpolate(y2, x_size[2:], mode='bilinear', align_corners=True)
        y3 = F.interpolate(y3, x_size[2:], mode='bilinear', align_corners=True)
        y4 = F.interpolate(y4, x_size[2:], mode='bilinear', align_corners=True)

        y1 = torch.sigmoid(y1)
        y2 = torch.sigmoid(y2)
        y3 = torch.sigmoid(y3)
        y4 = torch.sigmoid(y4)

        _x1 = self.conv13(x)
        _x2 = self.conv23(x)
        _x3 = self.conv33(x)
        _x4 = self.conv43(x)

        c1 = y1 * _x1
        c2 = y2 * _x2
        c3 = y3 * _x3
        c4 = y4 * _x4

        res = torch.cat([c1, c2, c3, c4], dim=1)

        if self.is_not_last:
            res = F.interpolate(res, x2.size()[2:], mode='bilinear', align_corners=True)
            pass

        res = self.conv_sum(res)

        if self.has_gcn:
            x_gcn = F.interpolate(x_gcn, res.size()[2:], mode='bilinear', align_corners=True)
            x_gcn = self.conv_gcn(x_gcn)
            x_gcn = torch.sigmoid(x_gcn)
            # res = x_gcn * res + res
            res = x_gcn * res
            res = self.conv_att(res)
            pass

        if self.is_not_last:
            res = torch.add(res, x2)
            res = self.conv_sum_c(res)
            pass

        return res

    pass


class DeepPoolLayer2(nn.Module):

    def __init__(self, k, k_out, is_not_last, has_gcn=False, gcn_in=None):
        super(DeepPoolLayer2, self).__init__()
        self.is_not_last = is_not_last
        self.has_gcn = has_gcn

        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.pool6 = nn.AvgPool2d(kernel_size=6, stride=6)
        self.pool8 = nn.AvgPool2d(kernel_size=8, stride=8)

        self.conv11 = nn.Conv2d(k, k, 3, 1, 1, bias=False)
        self.conv21 = nn.Conv2d(k, k, 3, 1, 1, bias=False)
        self.conv31 = nn.Conv2d(k, k, 3, 1, 1, bias=False)
        self.conv41 = nn.Conv2d(k, k, 3, 1, 1, bias=False)

        self.conv12 = nn.Conv2d(k, k, 3, 1, 1, bias=False)
        self.conv22 = nn.Conv2d(k, k, 3, 1, 1, bias=False)
        self.conv32 = nn.Conv2d(k, k, 3, 1, 1, bias=False)
        self.conv42 = nn.Conv2d(k, k, 3, 1, 1, bias=False)

        self.relu = nn.ReLU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)
        if self.has_gcn:
            self.conv_gcn = nn.Conv2d(gcn_in, k_out, 3, 1, 1, bias=False)
            self.conv_att = nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False)
        if self.is_not_last:
            self.conv_sum_c = nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False)
        pass

    def forward(self, x, x2=None, x_gcn=None):
        x_size = x.size()

        y1 = self.conv11(self.pool2(x))
        y2 = self.conv21(self.pool4(x))
        y3 = self.conv31(self.pool6(x))
        y4 = self.conv41(self.pool8(x))

        y1 = torch.sigmoid(y1)
        y2 = torch.sigmoid(y2)
        y3 = torch.sigmoid(y3)
        y4 = torch.sigmoid(y4)

        y1 = F.interpolate(y1, x_size[2:], mode='bilinear', align_corners=True)
        y2 = F.interpolate(y2, x_size[2:], mode='bilinear', align_corners=True)
        y3 = F.interpolate(y3, x_size[2:], mode='bilinear', align_corners=True)
        y4 = F.interpolate(y4, x_size[2:], mode='bilinear', align_corners=True)

        y1 = y1 * x
        y2 = y2 * x
        y3 = y3 * x
        y4 = y4 * x

        y1 = self.conv12(y1)
        y2 = self.conv22(y2)
        y3 = self.conv32(y3)
        y4 = self.conv42(y4)

        res = torch.add(x, y1)
        res = torch.add(res, y2)
        res = torch.add(res, y3)
        res = torch.add(res, y4)
        res = self.relu(res)

        if self.is_not_last:
            res = F.interpolate(res, x2.size()[2:], mode='bilinear', align_corners=True)
            pass

        res = self.conv_sum(res)

        if self.has_gcn:
            x_gcn = F.interpolate(x_gcn, res.size()[2:], mode='bilinear', align_corners=True)
            x_gcn = self.conv_gcn(x_gcn)
            x_gcn = torch.sigmoid(x_gcn)
            # res = x_gcn * res + res
            res = x_gcn * res
            res = self.conv_att(res)
            pass

        if self.is_not_last:
            res = torch.add(res, x2)
            res = self.conv_sum_c(res)
            pass

        return res

    pass


class MyGCNNet(nn.Module):

    def __init__(self, has_bn=False, normalize=False, residual=False, concat=True):
        super(MyGCNNet, self).__init__()
        # BASE
        self.vgg16 = VGG16()

        # GCN
        self.model_gnn1 = SAGENet1(in_dim=self.vgg16.out_num2, hidden_dims=[512, 512],
                                   has_bn=has_bn, normalize=normalize, residual=residual, concat=concat)
        self.model_gnn2 = SAGENet2(in_dim=self.model_gnn1.hidden_dims[-1], hidden_dims=[512, 512, 512, 512],
                                   skip_which=[2, 4], skip_dim=256, has_bn=has_bn,
                                   normalize=normalize, residual=residual, concat=concat)

        # DEEP POOL
        which_deep_pool_layer = DeepPoolLayer
        # deep_pool_layer = DeepPoolLayer2
        deep_pool = [[512, 512, 256, 128], [512, 256, 128, 128]]
        self.deep_pool4 = which_deep_pool_layer(deep_pool[0][0], deep_pool[1][0], True, True, 512)
        self.deep_pool3 = which_deep_pool_layer(deep_pool[0][1], deep_pool[1][1], True, True, 512)
        self.deep_pool2 = which_deep_pool_layer(deep_pool[0][2], deep_pool[1][2], True, False)
        self.deep_pool1 = which_deep_pool_layer(deep_pool[0][3], deep_pool[1][3], False, False)

        # ScoreLayer
        score = 128
        self.score = nn.Conv2d(score, 1, 1, 1)

        VGG16.weight_init(self.modules())
        pass

    def forward(self, x, batched_graph, batched_pixel_graph):
        # BASE
        feature1, feature2, feature3, feature4 = self.vgg16(x)

        # SIZE
        x_size = x.size()[2:]

        # GCN 1
        data_where = batched_pixel_graph.data_where
        pixel_nodes_feat = feature2[data_where[:, 0], :, data_where[:, 1], data_where[:, 2]]
        batched_pixel_graph.x = pixel_nodes_feat
        gcn1_feature = self.model_gnn1.forward(batched_pixel_graph)
        sod_gcn1_feature = self.sod_feature(data_where, gcn1_feature, batched_pixel_graph=batched_pixel_graph)

        # GCN 2
        batched_graph.x = gcn1_feature
        gcn2_feature, gcn2_logits, gcn2_logits_sigmoid = self.model_gnn2.forward(batched_graph)
        sod_gcn2_feature = self.sod_feature(data_where, gcn2_feature, batched_pixel_graph=batched_pixel_graph)
        # For Eval
        sod_gcn2_sigmoid = self.sod_feature(data_where, gcn2_logits_sigmoid.unsqueeze(1),
                                            batched_pixel_graph=batched_pixel_graph)
        # For Eval

        merge = self.deep_pool4(feature4, feature3, x_gcn=sod_gcn2_feature)  # A + F
        merge = self.deep_pool3(merge, feature2, x_gcn=sod_gcn1_feature)  # A + F
        merge = self.deep_pool2(merge, feature1)  # A + F
        merge = self.deep_pool1(merge)  # A

        # ScoreLayer
        merge = self.score(merge)
        if x_size is not None:
            merge = F.interpolate(merge, x_size, mode='bilinear', align_corners=True)
            # For Eval
            sod_gcn2_sigmoid = F.interpolate(sod_gcn2_sigmoid, x_size, mode='bilinear', align_corners=True)
            # For Eval
        return gcn2_logits, gcn2_logits_sigmoid, sod_gcn2_sigmoid, merge, torch.sigmoid(merge)

    @staticmethod
    def sod_feature(data_where, gcn_feature, batched_pixel_graph):
        # 构造特征
        _shape = torch.max(data_where, dim=0)[0] + 1
        _size = (_shape[0], gcn_feature.shape[-1], _shape[1], _shape[2])
        _gcn_feature_for_sod = gcn_feature[batched_pixel_graph.batch]

        sod_gcn_feature = torch.Tensor(size=_size).to(gcn_feature.device)
        sod_gcn_feature[data_where[:, 0], :, data_where[:, 1], data_where[:, 2]] = _gcn_feature_for_sod
        return sod_gcn_feature

    pass


class RunnerSOD(object):

    def __init__(self, data_root_path, down_ratio=4, sp_size=4, train_print_freq=100, test_print_freq=50,
                 root_ckpt_dir="./ckpt/dir", lr=None, num_workers=8, use_gpu=True, gpu_id="1",
                 has_bn=True, normalize=True, residual=False, concat=True, weight_decay=0.0,
                 is_sgd=False, pretrained_model=None):
        self.train_print_freq = train_print_freq
        self.test_print_freq = test_print_freq

        self.device = gpu_setup(use_gpu=use_gpu, gpu_id=gpu_id)
        self.root_ckpt_dir = Tools.new_dir(root_ckpt_dir)

        self.train_dataset = MyDataset(
            data_root_path=data_root_path, is_train=True, down_ratio=down_ratio, sp_size=sp_size)
        self.test_dataset = MyDataset(
            data_root_path=data_root_path, is_train=False, down_ratio=down_ratio, sp_size=sp_size)

        self.train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=True,
                                       num_workers=num_workers, collate_fn=self.train_dataset.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False,
                                      num_workers=num_workers, collate_fn=self.test_dataset.collate_fn)

        self.model = MyGCNNet(has_bn=has_bn, normalize=normalize, residual=residual, concat=concat).to(self.device)
        self.model.vgg16.load_pretrained_model(pretrained_model=pretrained_model)

        self.lr_s = lr
        if is_sgd:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr_s[0][1], momentum=0.9, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_s[0][1], weight_decay=weight_decay)

        Tools.print("Total param: {} lr_s={} Optimizer={}".format(
            self._view_model_param(self.model), self.lr_s, self.optimizer))
        self._print_network(self.model)

        self.loss_class = nn.BCELoss().to(self.device)
        pass

    def loss_bce(self, logits_sigmoid, labels):
        loss = self.loss_class(logits_sigmoid, labels)
        return loss

    def load_model(self, model_file_name):
        ckpt = torch.load(model_file_name, map_location=self.device)
        self.model.load_state_dict(ckpt, strict=False)
        Tools.print('Load Model: {}'.format(model_file_name))
        pass

    def train(self, epochs, start_epoch=0):
        for epoch in range(start_epoch, epochs):
            Tools.print()
            Tools.print("Start Epoch {}".format(epoch))

            self._lr(epoch)
            Tools.print('Epoch:{:02d},lr={:.4f}'.format(epoch, self.optimizer.param_groups[0]['lr']))

            (train_loss, train_loss1, train_loss2,
             train_mae, train_score, train_mae2, train_score2) = self._train_epoch()
            self._save_checkpoint(self.model, self.root_ckpt_dir, epoch)
            test_loss, test_loss1, test_loss2, test_mae, test_score, test_mae2, test_score2 = self.test()

            Tools.print('E:{:2d}, Train sod-mae-score={:.4f}-{:.4f} '
                        'gcn-mae-score={:.4f}-{:.4f} loss={:.4f}({:.4f}+{:.4f})'.format(
                epoch, train_mae, train_score, train_mae2, train_score2, train_loss, train_loss1, train_loss2))
            Tools.print('E:{:2d}, Test  sod-mae-score={:.4f}-{:.4f} '
                        'gcn-mae-score={:.4f}-{:.4f} loss={:.4f}({:.4f}+{:.4f})'.format(
                epoch, test_mae, test_score, test_mae2, test_score2, test_loss, test_loss1, test_loss2))
            pass
        pass

    def _train_epoch(self):
        self.model.train()

        # 统计
        th_num = 25
        epoch_loss, epoch_loss1, epoch_loss2, nb_data = 0, 0, 0, 0
        epoch_mae, epoch_prec, epoch_recall = 0.0, np.zeros(shape=(th_num,)) + 1e-6, np.zeros(shape=(th_num,)) + 1e-6
        epoch_mae2, epoch_prec2, epoch_recall2 = 0.0, np.zeros(shape=(th_num,)) + 1e-6, np.zeros(shape=(th_num,)) + 1e-6

        # Run
        iter_size = 10
        self.model.zero_grad()
        tr_num = len(self.train_loader)
        for i, (images, _, labels_sod, batched_graph,
                batched_pixel_graph, segments, _, _) in enumerate(self.train_loader):
            # Data
            images = images.float().to(self.device)
            labels = batched_graph.y.to(self.device)
            labels_sod = torch.unsqueeze(torch.Tensor(labels_sod), dim=1).to(self.device)
            batched_graph.batch = batched_graph.batch.to(self.device)
            batched_graph.edge_index = batched_graph.edge_index.to(self.device)

            batched_pixel_graph.batch = batched_pixel_graph.batch.to(self.device)
            batched_pixel_graph.edge_index = batched_pixel_graph.edge_index.to(self.device)
            batched_pixel_graph.data_where = batched_pixel_graph.data_where.to(self.device)

            gcn_logits, gcn_logits_sigmoid, _, sod_logits, sod_logits_sigmoid = self.model.forward(
                images, batched_graph, batched_pixel_graph)

            loss_fuse1 = F.binary_cross_entropy_with_logits(sod_logits, labels_sod, reduction='sum')
            loss_fuse2 = F.binary_cross_entropy_with_logits(gcn_logits, labels, reduction='sum')
            loss = loss_fuse1 / iter_size + loss_fuse2

            loss.backward()

            if (i + 1) % iter_size == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                pass

            labels_val = labels.cpu().detach().numpy()
            labels_sod_val = labels_sod.cpu().detach().numpy()
            gcn_logits_sigmoid_val = gcn_logits_sigmoid.cpu().detach().numpy()
            sod_logits_sigmoid_val = sod_logits_sigmoid.cpu().detach().numpy()

            # Stat
            nb_data += images.size(0)
            epoch_loss += loss.detach().item()
            epoch_loss1 += loss_fuse1.detach().item()
            epoch_loss2 += loss_fuse2.detach().item()

            # cal 1
            mae = self._eval_mae(sod_logits_sigmoid_val, labels_sod_val)
            prec, recall = self._eval_pr(sod_logits_sigmoid_val, labels_sod_val, th_num)
            epoch_mae += mae
            epoch_prec += prec
            epoch_recall += recall

            # cal 2
            mae2 = self._eval_mae(gcn_logits_sigmoid_val, labels_val)
            prec2, recall2 = self._eval_pr(gcn_logits_sigmoid_val, labels_val, th_num)
            epoch_mae2 += mae2
            epoch_prec2 += prec2
            epoch_recall2 += recall2

            # Print
            if i % self.train_print_freq == 0:
                Tools.print("{:4d}-{:4d} loss={:.4f}({:.4f}+{:.4f})-{:.4f}({:.4f}+{:.4f}) "
                            "sod-mse={:.4f}({:.4f}) gcn-mse={:.4f}({:.4f})".format(
                    i, tr_num, loss.detach().item(), loss_fuse1.detach().item(), loss_fuse2.detach().item(),
                    epoch_loss / (i + 1), epoch_loss1 / (i + 1), epoch_loss2 / (i + 1),
                    mae, epoch_mae / (i + 1), mae2, epoch_mae2 / nb_data))
                pass
            pass

        # 结果
        avg_loss, avg_loss1, avg_loss2 = epoch_loss / tr_num, epoch_loss1 / tr_num, epoch_loss2 / tr_num

        avg_mae, avg_prec, avg_recall = epoch_mae / tr_num, epoch_prec / tr_num, epoch_recall / tr_num
        score = (1 + 0.3) * avg_prec * avg_recall / (0.3 * avg_prec + avg_recall)
        avg_mae2, avg_prec2, avg_recall2 = epoch_mae2 / nb_data, epoch_prec2 / nb_data, epoch_recall2 / nb_data
        score2 = (1 + 0.3) * avg_prec2 * avg_recall2 / (0.3 * avg_prec2 + avg_recall2)

        return avg_loss, avg_loss1, avg_loss2, avg_mae, score.max(), avg_mae2, score2.max()

    def test(self, model_file=None, is_train_loader=False):
        if model_file:
            self.load_model(model_file_name=model_file)

        self.model.train()

        Tools.print()
        th_num = 25

        # 统计
        epoch_test_loss, epoch_test_loss1, epoch_test_loss2, nb_data = 0, 0, 0, 0
        epoch_test_mae, epoch_test_mae2 = 0.0, 0.0
        epoch_test_prec, epoch_test_recall = np.zeros(shape=(th_num,)) + 1e-6, np.zeros(shape=(th_num,)) + 1e-6
        epoch_test_prec2, epoch_test_recall2 = np.zeros(shape=(th_num,)) + 1e-6, np.zeros(shape=(th_num,)) + 1e-6

        loader = self.train_loader if is_train_loader else self.test_loader
        tr_num = len(loader)
        with torch.no_grad():
            for i, (images, _, labels_sod,
                    batched_graph, batched_pixel_graph, segments, _, _) in enumerate(loader):
                # Data
                images = images.float().to(self.device)
                labels = batched_graph.y.to(self.device)
                labels_sod = torch.unsqueeze(torch.Tensor(labels_sod), dim=1).to(self.device)
                batched_graph.batch = batched_graph.batch.to(self.device)
                batched_graph.edge_index = batched_graph.edge_index.to(self.device)

                batched_pixel_graph.batch = batched_pixel_graph.batch.to(self.device)
                batched_pixel_graph.edge_index = batched_pixel_graph.edge_index.to(self.device)
                batched_pixel_graph.data_where = batched_pixel_graph.data_where.to(self.device)

                _, gcn_logits_sigmoid, _, _, sod_logits_sigmoid = self.model.forward(
                    images, batched_graph, batched_pixel_graph)

                loss1 = self.loss_bce(gcn_logits_sigmoid, labels)
                loss2 = self.loss_bce(sod_logits_sigmoid, labels_sod)
                loss = loss1 + loss2

                labels_val = labels.cpu().detach().numpy()
                labels_sod_val = labels_sod.cpu().detach().numpy()
                gcn_logits_sigmoid_val = gcn_logits_sigmoid.cpu().detach().numpy()
                sod_logits_sigmoid_val = sod_logits_sigmoid.cpu().detach().numpy()

                # Stat
                nb_data += images.size(0)
                epoch_test_loss += loss.detach().item()
                epoch_test_loss1 += loss1.detach().item()
                epoch_test_loss2 += loss2.detach().item()

                # cal 1
                mae = self._eval_mae(sod_logits_sigmoid_val, labels_sod_val)
                prec, recall = self._eval_pr(sod_logits_sigmoid_val, labels_sod_val, th_num)
                epoch_test_mae += mae
                epoch_test_prec += prec
                epoch_test_recall += recall

                # cal 2
                mae2 = self._eval_mae(gcn_logits_sigmoid_val, labels_val)
                prec2, recall2 = self._eval_pr(gcn_logits_sigmoid_val, labels_val, th_num)
                epoch_test_mae2 += mae2
                epoch_test_prec2 += prec2
                epoch_test_recall2 += recall2

                # Print
                if i % self.test_print_freq == 0:
                    Tools.print("{:4d}-{:4d} loss={:.4f}({:.4f}+{:.4f})-{:.4f}({:.4f}+{:.4f}) "
                                "sod-mse={:.4f}({:.4f}) gcn-mse={:.4f}({:.4f})".format(
                        i, len(loader), loss.detach().item(), loss1.detach().item(), loss2.detach().item(),
                        epoch_test_loss / (i + 1), epoch_test_loss1 / (i + 1), epoch_test_loss2 / (i + 1),
                        mae, epoch_test_mae / (i + 1), mae2, epoch_test_mae2 / nb_data))
                    pass
                pass
            pass

        # 结果1
        avg_loss, avg_loss1, avg_loss2 = epoch_test_loss / tr_num, epoch_test_loss1 / tr_num, epoch_test_loss2 / tr_num

        avg_mae, avg_prec, avg_recall = epoch_test_mae / tr_num, epoch_test_prec / tr_num, epoch_test_recall / tr_num
        score = (1 + 0.3) * avg_prec * avg_recall / (0.3 * avg_prec + avg_recall)
        avg_mae2, avg_prec2, avg_recall2 = epoch_test_mae2 / nb_data, epoch_test_prec2 / nb_data, epoch_test_recall2 / nb_data
        score2 = (1 + 0.3) * avg_prec2 * avg_recall2 / (0.3 * avg_prec2 + avg_recall2)

        return avg_loss, avg_loss1, avg_loss2, avg_mae, score.max(), avg_mae2, score2.max()

    @staticmethod
    def _print_network(model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        Tools.print(model)
        Tools.print("The number of parameters: {}".format(num_params))
        pass

    @staticmethod
    def _cal_sod(pre, segment):
        result = np.asarray(segment.copy(), dtype=np.float32)
        for i in range(len(pre)):
            result[segment == i] = pre[i]
            pass
        return result

    def _lr(self, epoch):
        for lr in self.lr_s:
            if lr[0] == epoch:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr[1]
                pass
            pass
        pass

    @staticmethod
    def _save_checkpoint(model, root_ckpt_dir, epoch):
        torch.save(model.state_dict(), os.path.join(root_ckpt_dir, 'epoch_{}.pkl'.format(epoch)))
        pass

    @staticmethod
    def _eval_mae(y_pred, y):
        return np.abs(y_pred - y).mean()

    @staticmethod
    def _eval_pr(y_pred, y, th_num=100):
        prec, recall = np.zeros(shape=(th_num,)), np.zeros(shape=(th_num,))
        th_list = np.linspace(0, 1 - 1e-10, th_num)
        for i in range(th_num):
            y_temp = y_pred >= th_list[i]
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / y.sum()
            pass
        return prec, recall

    @staticmethod
    def _view_model_param(model):
        total_param = 0
        for param in model.parameters():
            total_param += np.prod(list(param.data.size()))
        return total_param

    pass


"""
ckpt:./ckpt/MFNet_SOD/0
Total param: 43975617 lr_s=[[0, 5e-05], [15, 5e-06]]
x_gcn = self.conv_gcn(x_gcn)
x_gcn = torch.sigmoid(x_gcn)
res = x_gcn * res
Train sod-mae-score=0.0094-0.9856 gcn-mae-score=0.0421-0.9192 
Test  sod-mae-score=0.0377-0.8846 gcn-mae-score=0.0742-0.7442
For Final Eval Result, Please Use: https://github.com/ArcherFMY/sal_eval_toolbox
"""


if __name__ == '__main__':
    #################################################################################
    _data_root_path = "/mnt/4T/Data/SOD/DUTS"  # Change to Your Data Path
    _pretrained_model = "./pretrained/vgg16-397923af.pth"  # https://download.pytorch.org/models/vgg16-397923af.pth
    #################################################################################

    _train_print_freq = 1000
    _test_print_freq = 1000
    _num_workers = 10
    _use_gpu = True

    _gpu_id = "0"

    _epochs = 24
    _is_sgd = False
    _weight_decay = 5e-4
    _lr = [[0, 5e-5], [15, 5e-6]]

    _has_bn = True
    _has_residual = True
    _is_normalize = True
    _concat = True

    _sp_size, _down_ratio = 4, 4

    _root_ckpt_dir = "./ckpt/MFNet_SOD/{}".format(_gpu_id)
    Tools.print("epochs:{} ckpt:{} sp size:{} down_ratio:{} workers:{} gpu:{} has_residual:{} "
                "is_normalize:{} has_bn:{} concat:{} is_sgd:{} weight_decay:{}".format(
        _epochs, _root_ckpt_dir, _sp_size, _down_ratio, _num_workers, _gpu_id,
        _has_residual, _is_normalize, _has_bn, _concat, _is_sgd, _weight_decay))

    runner = RunnerSOD(data_root_path=_data_root_path, root_ckpt_dir=_root_ckpt_dir,
                       sp_size=_sp_size, is_sgd=_is_sgd, lr=_lr,
                       residual=_has_residual, normalize=_is_normalize, down_ratio=_down_ratio,
                       has_bn=_has_bn, concat=_concat, weight_decay=_weight_decay,
                       train_print_freq=_train_print_freq, test_print_freq=_test_print_freq,
                       num_workers=_num_workers, use_gpu=_use_gpu, gpu_id=_gpu_id)
    runner.train(_epochs, start_epoch=0)
    pass
