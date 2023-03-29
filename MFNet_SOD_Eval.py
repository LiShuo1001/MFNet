import os
import torch
import numpy as np
from PIL import Image
from SODData import SODData
from alisuretool.Tools import Tools
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


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


class MyEvalDataset(Dataset):

    def __init__(self, image_name_list, label_name_list=None, down_ratio=4, down_ratio2=1, sp_size=4, min_size=256):
        super().__init__()
        self.sp_size = sp_size
        self.down_ratio_for_sp = down_ratio
        self.down_ratio_for_sod = down_ratio2
        self.min_size = min_size
        self.image_name_list = image_name_list
        self.label_name_list = label_name_list
        pass

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        # 读数据
        image = Image.open(self.image_name_list[idx]).convert("RGB")
        image_name = self.image_name_list[idx]

        # 限制最小大小
        if image.size[0] < self.min_size or image.size[1] < self.min_size:
            if image.size[0] < image.size[1]:
                image = image.resize((self.min_size, int(self.min_size / image.size[0] * image.size[1])))
            else:
                image = image.resize((int(self.min_size / image.size[1] * image.size[0]), self.min_size))
            pass

        w, h = image.size

        if self.label_name_list is not None:
            label = Image.open(self.label_name_list[idx]).convert("L").resize(image.size)
        else:
            label = Image.new("L", size=image.size)
        label_for_sod = np.asarray(label.resize((w//self.down_ratio_for_sod, h//self.down_ratio_for_sod))) / 255

        # 归一化
        _normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_data = transforms.Compose([transforms.ToTensor(), _normalize])(image).unsqueeze(dim=0)

        # 超像素
        image_small_data = np.asarray(image.resize((w//self.down_ratio_for_sp, h//self.down_ratio_for_sp)))
        label_for_sp = np.asarray(label.resize((w//self.down_ratio_for_sp, h//self.down_ratio_for_sp))) / 255

        graph, pixel_graph, segment = self.get_sp_info(image_small_data, label_for_sp)
        # 返回
        return graph, pixel_graph, img_data, label_for_sp, label_for_sod, segment, image_small_data, image_name

    def get_sp_info(self, image, label):
        graph, pixel_graph, segment = MyDataset.get_sp_info(image, label, self.sp_size)
        return graph, pixel_graph, segment

    @staticmethod
    def collate_fn(samples):
        MyDataset.collate_fn(samples=samples)
        return MyDataset.collate_fn(samples=samples)

    pass


class RunnerSODEval(object):

    def __init__(self, down_ratio=4, sp_size=4, min_size=256, use_gpu=True, gpu_id="1",
                 has_bn=True, normalize=True, residual=False, concat=True):
        self.down_ratio = down_ratio
        self.sp_size = sp_size
        self.min_size = min_size

        self.device = gpu_setup(use_gpu=use_gpu, gpu_id=gpu_id)
        self.model = MyGCNNet(has_bn=has_bn, normalize=normalize, residual=residual, concat=concat).to(self.device)

        Tools.print("Total param: {}".format(self._view_model_param(self.model)))
        self._print_network(self.model)
        pass

    def load_model(self, model_file_name):
        ckpt = torch.load(model_file_name, map_location=self.device)

        self.model.load_state_dict(ckpt, strict=False)
        Tools.print('Load Model: {}'.format(model_file_name))
        pass

    def save_result(self, model_file=None, image_name_list=None, label_name_list=None, save_path=None):
        assert image_name_list is not None
        if model_file is not None:
            self.load_model(model_file_name=model_file)
        self.model.train()

        # 统计
        Tools.print()
        th_num = 25
        nb_data = 0
        epoch_test_mae, epoch_test_mae2 = 0.0, 0.0
        epoch_test_prec, epoch_test_recall = np.zeros(shape=(th_num,)) + 1e-6, np.zeros(shape=(th_num,)) + 1e-6
        epoch_test_prec2, epoch_test_recall2 = np.zeros(shape=(th_num,)) + 1e-6, np.zeros(shape=(th_num,)) + 1e-6

        dataset = MyEvalDataset(image_name_list, label_name_list=label_name_list,
                                down_ratio=self.down_ratio, sp_size=self.sp_size, min_size=self.min_size)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)
        tr_num = len(loader)
        with torch.no_grad():
            for i, (images, _, labels_sod,
                    batched_graph, batched_pixel_graph, segments, _, image_names) in enumerate(loader):
                # Data
                images = images.float().to(self.device)
                labels = batched_graph.y.to(self.device)
                labels_sod = torch.unsqueeze(torch.Tensor(labels_sod), dim=1).to(self.device)
                batched_graph.batch = batched_graph.batch.to(self.device)
                batched_graph.edge_index = batched_graph.edge_index.to(self.device)

                batched_pixel_graph.batch = batched_pixel_graph.batch.to(self.device)
                batched_pixel_graph.edge_index = batched_pixel_graph.edge_index.to(self.device)
                batched_pixel_graph.data_where = batched_pixel_graph.data_where.to(self.device)

                _, gcn_logits_sigmoid, sod_gcn2_sigmoid, _, sod_logits_sigmoid = self.model.forward(
                    images, batched_graph, batched_pixel_graph)

                labels_val = labels.cpu().detach().numpy()
                labels_sod_val = labels_sod.cpu().detach().numpy()
                gcn_logits_sigmoid_val = gcn_logits_sigmoid.cpu().detach().numpy()
                sod_logits_sigmoid_val = sod_logits_sigmoid.cpu().detach().numpy()

                # Stat
                nb_data += images.size(0)

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

                if save_path is not None:
                    im_size = Image.open(image_names[0]).size
                    image_name = os.path.splitext(os.path.basename(image_names[0]))[0]

                    save_file_name = Tools.new_dir(os.path.join(save_path, "SP", "{}.png".format(image_name)))
                    sp_result = torch.squeeze(sod_gcn2_sigmoid).detach().cpu().numpy()
                    Image.fromarray(np.asarray(sp_result * 255, dtype=np.uint8)).resize(im_size).save(save_file_name)

                    save_file_name = Tools.new_dir(os.path.join(save_path, "SOD", "{}.png".format(image_name)))
                    sod_result = torch.squeeze(sod_logits_sigmoid).detach().cpu().numpy()
                    Image.fromarray(np.asarray(sod_result * 255, dtype=np.uint8)).resize(im_size).save(save_file_name)

                    save_file_name = Tools.new_dir(os.path.join(save_path, "SP-GT", "{}.png".format(image_name)))
                    tar_sod = self._cal_sod(labels_val.tolist(), segments[0])
                    Image.fromarray(np.asarray(tar_sod * 255, dtype=np.uint8)).resize(im_size).save(save_file_name)
                    pass

                # Print
                if i % 500 == 0:
                    Tools.print("{:4d}-{:4d} sod-mse={:.4f}({:.4f}) gcn-mse={:.4f}({:.4f})".format(
                        i, len(loader), mae, epoch_test_mae/(i+1), mae2, epoch_test_mae2/nb_data))
                    pass
                pass
            pass

        avg_mae, avg_prec, avg_recall = epoch_test_mae/tr_num, epoch_test_prec/tr_num, epoch_test_recall/tr_num
        score = (1 + 0.3) * avg_prec * avg_recall / (0.3 * avg_prec + avg_recall)
        avg_mae2, avg_prec2, avg_recall2 = epoch_test_mae2/nb_data, epoch_test_prec2/nb_data, epoch_test_recall2/nb_data
        score2 = (1 + 0.3) * avg_prec2 * avg_recall2 / (0.3 * avg_prec2 + avg_recall2)

        Tools.print('{} sod-mae-score={:.4f}-{:.4f} gcn-mae-score={:.4f}-{:.4f}'.format(
            save_path, avg_mae, score.max(), avg_mae2, score2.max()))
        pass

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
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
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
For Final Eval Result, Please Use: https://github.com/ArcherFMY/sal_eval_toolbox
"""


if __name__ == '__main__':
    model_name = "MFNet_SOD"
    from MFNet_SOD import MyGCNNet, MyDataset
    model_file = "./ckpt/MFNet_SOD/0/epoch_xx.pkl"
    _data_root_path = "/mnt/4T/Data/SOD"  # Change to Your Data Path
    result_path = "./result/MFNet_SOD/0/epoch_xx"

    _gpu_id = "0"

    _use_gpu = True
    _improved = True
    _has_bn = True
    _has_residual = True
    _is_normalize = True
    _concat = True

    _sp_size, _down_ratio = 4, 4
    runner = RunnerSODEval(sp_size=_sp_size, residual=_has_residual,
                           normalize=_is_normalize, down_ratio=_down_ratio,
                           has_bn=_has_bn, concat=_concat, use_gpu=_use_gpu, gpu_id=_gpu_id)

    sod_data = SODData(data_root_path=_data_root_path)
    # 根据需要调整测试的数据集
    for data_set in [sod_data.cssd, sod_data.ecssd, sod_data.msra_1000_asd, sod_data.msra10k,
                     sod_data.msra_b, sod_data.sed1, sod_data.sed2, sod_data.dut_dmron_5168,
                     sod_data.dut_dmron_5166, sod_data.hku_is,
                     sod_data.sod, sod_data.thur15000, sod_data.pascal1500, sod_data.pascal_s,
                     sod_data.judd, sod_data.duts_te, sod_data.duts_tr, sod_data.cub_200_2011]:
        img_name_list, lbl_name_list, dataset_name_list = data_set()
        Tools.print("Begin eval {} {}".format(dataset_name_list[0], len(img_name_list)))

        runner.save_result(model_file=model_file, image_name_list=img_name_list, label_name_list=lbl_name_list,
                           save_path="{}/{}/{}".format(result_path, model_name, dataset_name_list[0]))
        pass
    pass
