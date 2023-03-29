import os
from glob import glob


class SODData(object):

    def __init__(self, data_root_path="E:\data\SOD", is_sort=True):
        self.data_root_path = data_root_path
        self.is_sort = is_sort

        # ext
        self.jpg = "jpg"
        self.JPEG = "JPEG"
        self.png = "png"
        self.bmp = "bmp"
        pass

    def get_all_train_and_mask(self):
        _all_image_list, _all_mask_list, _all_name_list = [], [], []

        _images, _masks, _names = self.cssd()
        _all_image_list.extend(_images)
        _all_mask_list.extend(_masks)
        _all_name_list.extend(_names)

        _images, _masks, _names = self.ecssd()
        _all_image_list.extend(_images)
        _all_mask_list.extend(_masks)
        _all_name_list.extend(_names)

        _images, _masks, _names = self.msra_1000_asd()
        _all_image_list.extend(_images)
        _all_mask_list.extend(_masks)
        _all_name_list.extend(_names)

        _images, _masks, _names = self.msra10k()
        _all_image_list.extend(_images)
        _all_mask_list.extend(_masks)
        _all_name_list.extend(_names)

        _images, _masks, _names = self.msra_b()
        _all_image_list.extend(_images)
        _all_mask_list.extend(_masks)
        _all_name_list.extend(_names)

        _images, _masks, _names = self.sed2()
        _all_image_list.extend(_images)
        _all_mask_list.extend(_masks)
        _all_name_list.extend(_names)

        _images, _masks, _names = self.dut_dmron_5166()
        _all_image_list.extend(_images)
        _all_mask_list.extend(_masks)
        _all_name_list.extend(_names)

        _images, _masks, _names = self.dut_dmron_5168()
        _all_image_list.extend(_images)
        _all_mask_list.extend(_masks)
        _all_name_list.extend(_names)

        _images, _masks, _names = self.hku_is()
        _all_image_list.extend(_images)
        _all_mask_list.extend(_masks)
        _all_name_list.extend(_names)

        _images, _masks, _names = self.sod()
        _all_image_list.extend(_images)
        _all_mask_list.extend(_masks)
        _all_name_list.extend(_names)

        _images, _masks, _names = self.thur15000()
        _all_image_list.extend(_images)
        _all_mask_list.extend(_masks)
        _all_name_list.extend(_names)

        _images, _masks, _names = self.pascal1500()
        _all_image_list.extend(_images)
        _all_mask_list.extend(_masks)
        _all_name_list.extend(_names)

        _images, _masks, _names = self.pascal_s()
        _all_image_list.extend(_images)
        _all_mask_list.extend(_masks)
        _all_name_list.extend(_names)

        _images, _masks, _names = self.judd()
        _all_image_list.extend(_images)
        _all_mask_list.extend(_masks)
        _all_name_list.extend(_names)

        _images, _masks, _names = self.duts_te()
        _all_image_list.extend(_images)
        _all_mask_list.extend(_masks)
        _all_name_list.extend(_names)

        _images, _masks, _names = self.duts_tr()
        _all_image_list.extend(_images)
        _all_mask_list.extend(_masks)
        _all_name_list.extend(_names)

        _images, _masks, _names = self.cub_200_2011()
        _all_image_list.extend(_images)
        _all_mask_list.extend(_masks)
        _all_name_list.extend(_names)

        return _all_image_list, _all_mask_list, _all_name_list

    @staticmethod
    def _file_name(file_path, has_ext=False):
        file_name = os.path.basename(file_path)
        return file_name if has_ext else os.path.splitext(file_name)[0]

    def _sod_data_file_list(self, data_root_path, image_path, mask_path=None, image_ext="jpg", mask_ext="png"):
        image_list = glob(os.path.join(data_root_path, image_path, "*.{}".format(image_ext)))
        if self.is_sort:
            image_list.sort()

        if mask_path:
            image_list_new, mask_list = [], []
            for image_name in image_list:
                one = os.path.join(data_root_path, mask_path, "{}.{}".format(self._file_name(image_name), mask_ext))
                if os.path.exists(one):
                    mask_list.append(one)
                    image_list_new.append(image_name)
                pass
            return image_list_new, mask_list

        return image_list

    def _sod_data_file_list2(self, data_root_path, image_mask_path=None, image_ext="jpg", mask_ext="png"):
        image_list = glob(os.path.join(data_root_path, image_mask_path, "*.{}".format(image_ext)))
        if self.is_sort:
            image_list.sort()

        image_list_new, mask_list = [], []
        for image_name in image_list:
            one = os.path.join(data_root_path, image_mask_path, "{}.{}".format(self._file_name(image_name), mask_ext))
            if os.path.exists(one):
                mask_list.append(one)
                image_list_new.append(image_name)
            pass

        return image_list_new, mask_list

    # CSSD: 200
    def cssd(self, image_path="./CSSD/images", mask_path="./CSSD/ground_truth_mask"):
        image_list, mask_list = self._sod_data_file_list(
            self.data_root_path, image_path, mask_path, image_ext=self.jpg, mask_ext=self.png)
        dataset_name_list = ["CSSD" for _ in image_list]
        return image_list, mask_list, dataset_name_list

    # ECSSD: 1000
    def ecssd(self, image_path="./ECSSD/images", mask_path="./ECSSD/ground_truth_mask"):
        image_list, mask_list = self._sod_data_file_list(
            self.data_root_path, image_path, mask_path, image_ext=self.jpg, mask_ext=self.png)
        dataset_name_list = ["ECSSD" for _ in image_list]
        return image_list, mask_list, dataset_name_list

    # MSRA1000(ASD): 1000
    def msra_1000_asd(self, image_mask_path="./MSRA/(ASD)Image+GT"):
        image_list, mask_list = self._sod_data_file_list2(self.data_root_path, image_mask_path=image_mask_path,
                                                          image_ext=self.jpg, mask_ext=self.bmp)
        dataset_name_list = ["ASD" for _ in image_list]
        return image_list, mask_list, dataset_name_list

    # MSRA10K: 10000
    def msra10k(self, image_path="./MSRA/MSRA10K/images", mask_path="./MSRA/MSRA10K/gt_masks"):
        image_list, mask_list = self._sod_data_file_list(
            self.data_root_path, image_path, mask_path, image_ext=self.jpg, mask_ext=self.png)
        dataset_name_list = ["MSRA10K" for _ in image_list]
        return image_list, mask_list, dataset_name_list

    # MSRA-B: 5000
    def msra_b(self, image_mask_path="./MSRA/MSRA-B"):
        image_list, mask_list = self._sod_data_file_list2(self.data_root_path, image_mask_path=image_mask_path,
                                                          image_ext=self.jpg, mask_ext=self.png)
        dataset_name_list = ["MSRA-B" for _ in image_list]
        return image_list, mask_list, dataset_name_list

    # SED1: 100
    def sed1(self, image_path="./SED/SED1/Img", mask_path="./SED/SED1/GT"):
        image_list, mask_list = self._sod_data_file_list(
            self.data_root_path, image_path, mask_path, image_ext=self.png, mask_ext=self.png)
        dataset_name_list = ["SED1" for _ in image_list]
        return image_list, mask_list, dataset_name_list

    # SED2: 100
    def sed2(self, image_path="./SED/SED2/Img", mask_path="./SED/SED2/GT"):
        image_list, mask_list = self._sod_data_file_list(
            self.data_root_path, image_path, mask_path, image_ext=self.png, mask_ext=self.png)
        dataset_name_list = ["SED2" for _ in image_list]
        return image_list, mask_list, dataset_name_list

    # DUT-OMRON: 5166
    def dut_dmron_5166(self, image_path="./DUT-OMRON/DUT-OMRON/Img", mask_path="./DUT-OMRON/DUT-OMRON/GT"):
        image_list, mask_list = self._sod_data_file_list(
            self.data_root_path, image_path, mask_path, image_ext=self.jpg, mask_ext=self.png)
        dataset_name_list = ["DUT-OMRON" for _ in image_list]
        return image_list, mask_list, dataset_name_list

    # DUT-OMRON: 5168
    def dut_dmron_5168(self, image_path="./DUT-OMRON/DUT-OMRON-image",
                       mask_path="./DUT-OMRON/pixelwiseGT-new-PNG"):
        image_list, mask_list = self._sod_data_file_list(
            self.data_root_path, image_path, mask_path, image_ext=self.jpg, mask_ext=self.png)
        dataset_name_list = ["DUT-OMRON-5168" for _ in image_list]
        return image_list, mask_list, dataset_name_list

    # HKU-IS: 4447
    def hku_is(self, image_path="./HKU-IS/HKU-IS/Img", mask_path="./HKU-IS/HKU-IS/GT"):
        image_list, mask_list = self._sod_data_file_list(
            self.data_root_path, image_path, mask_path, image_ext=self.png, mask_ext=self.png)
        dataset_name_list = ["HKU-IS" for _ in image_list]
        return image_list, mask_list, dataset_name_list

    # SOD: 300
    def sod(self, image_path="./SOD/SOD/images", mask_path="./SOD/SOD/GT"):
        image_list, mask_list = self._sod_data_file_list(
            self.data_root_path, image_path, mask_path, image_ext=self.jpg, mask_ext=self.png)
        dataset_name_list = ["SOD" for _ in image_list]
        return image_list, mask_list, dataset_name_list

    # THUR15000: 6233
    def thur15000(self, image_mask_path="./THUR15000/THUR15000"):
        names = ["Butterfly", "CoffeeMug", "DogJump", "Giraffe", "plane"]
        image_list, mask_list, name_list = [], [], []
        for name in names:
            image, mask = self._sod_data_file_list2(self.data_root_path, image_ext=self.jpg, mask_ext=self.png,
                                                    image_mask_path=os.path.join(image_mask_path, name, "Src"))
            image_list.extend(image)
            mask_list.extend(mask)
            name_list.extend(["THUR15000-{}".format(name) for _ in image])
            pass
        return image_list, mask_list, name_list

    # PASCAL1500: 1500
    def pascal1500(self, image_path="./PASCAL-S/PASCAL1500/image",
                   mask_path="./PASCAL-S/PASCAL1500/GroundTruth"):
        image_list, mask_list = self._sod_data_file_list(
            self.data_root_path, image_path, mask_path, image_ext=self.jpg, mask_ext=self.png)
        dataset_name_list = ["PASCAL1500" for _ in image_list]
        return image_list, mask_list, dataset_name_list

    # PASCAL-S: 850
    def pascal_s(self, image_mask_path="./PASCAL-S/PASCAL-S/Imgs"):
        image_list, mask_list = self._sod_data_file_list2(self.data_root_path, image_mask_path=image_mask_path,
                                                          image_ext=self.jpg, mask_ext=self.png)
        dataset_name_list = ["PASCAL-S" for _ in image_list]
        return image_list, mask_list, dataset_name_list

    # Judd: 900
    def judd(self, image_path="./Judd/Judd/Img", mask_path="./Judd/Judd/GT"):
        image_list, mask_list = self._sod_data_file_list(
            self.data_root_path, image_path, mask_path, image_ext=self.jpg, mask_ext=self.png)
        dataset_name_list = ["Judd" for _ in image_list]
        return image_list, mask_list, dataset_name_list

    def duts(self):
        image_list1, mask_list1, dataset_name_list1 = self.duts_te()
        image_list2, mask_list2, dataset_name_list2 = self.duts_tr()
        return image_list1 + image_list2, image_list1 + image_list2, dataset_name_list1 + dataset_name_list2

    # DUTS-TE: 5019
    def duts_te(self, image_path="./DUTS/DUTS-TE/DUTS-TE-Image", mask_path="./DUTS/DUTS-TE/DUTS-TE-Mask"):
        image_list, mask_list = self._sod_data_file_list(
            self.data_root_path, image_path, mask_path, image_ext=self.jpg, mask_ext=self.png)
        dataset_name_list = ["DUTS-TE" for _ in image_list]
        return image_list, mask_list, dataset_name_list

    # DUTS-TR: 10553
    def duts_tr(self, image_path="./DUTS/DUTS-TR/DUTS-TR-Image", mask_path="./DUTS/DUTS-TR/DUTS-TR-Mask"):
        image_list, mask_list = self._sod_data_file_list(
            self.data_root_path, image_path, mask_path, image_ext=self.jpg, mask_ext=self.png)
        dataset_name_list = ["DUTS-TR" for _ in image_list]
        return image_list, mask_list, dataset_name_list

    # CUB-200-2011: 11788
    def cub_200_2011(self, image_path="./CUB/CUB-200-2011/CUB_200_2011/images",
                     mask_path="./CUB/CUB-200-2011/segmentations"):
        image_list, mask_list = [],  []
        class_dir = os.listdir(os.path.join(self.data_root_path, image_path))
        for class_one in class_dir:
            image, mask = self._sod_data_file_list(self.data_root_path, os.path.join(image_path, class_one),
                                                   os.path.join(mask_path, class_one),
                                                   image_ext=self.jpg, mask_ext=self.png)
            image_list.extend(image)
            mask_list.extend(mask)
        dataset_name_list = ["CUB-200-2011" for _ in image_list]
        return image_list, mask_list, dataset_name_list

    def imagenet2012_train(self, data_root_path="E:\\data\\ImageNet\\ILSVRC2015",
                           image_path="./Data/CLS-LOC/train"):
        image_list = []
        class_dir = os.listdir(os.path.join(data_root_path, image_path))
        for class_one in class_dir:
            image = self._sod_data_file_list(data_root_path, os.path.join(image_path, class_one), image_ext=self.JPEG)
            image_list.extend(image)
        dataset_name_list = ["ImageNet-Train" for _ in image_list]
        return image_list, dataset_name_list

    def imagenet2012_val(self, data_root_path="E:\\data\\ImageNet\\ILSVRC2015",
                         image_path="./Data/CLS-LOC/val_old"):
        image_list = self._sod_data_file_list(data_root_path, image_path, image_ext=self.JPEG)
        dataset_name_list = ["ImageNet-Val" for _ in image_list]
        return image_list, dataset_name_list

    def demo(self):
        sod_data = SODData()
        _image_list, _mask_list, _name_list = sod_data.cssd()
        _image_list, _mask_list, _name_list = sod_data.ecssd()
        _image_list, _mask_list, _name_list = sod_data.msra_1000_asd()
        _image_list, _mask_list, _name_list = sod_data.msra10k()
        _image_list, _mask_list, _name_list = sod_data.msra_b()
        _image_list, _mask_list, _name_list = sod_data.sed2()
        _image_list, _mask_list, _name_list = sod_data.dut_dmron_5166()
        _image_list, _mask_list, _name_list = sod_data.dut_dmron_5168()
        _image_list, _mask_list, _name_list = sod_data.hku_is()
        _image_list, _mask_list, _name_list = sod_data.sod()
        _image_list, _mask_list, _name_list = sod_data.thur15000()
        _image_list, _mask_list, _name_list = sod_data.pascal1500()
        _image_list, _mask_list, _name_list = sod_data.pascal_s()
        _image_list, _mask_list, _name_list = sod_data.judd()
        _image_list, _mask_list, _name_list = sod_data.duts_te()
        _image_list, _mask_list, _name_list = sod_data.duts_tr()
        _image_list, _mask_list, _name_list = sod_data.cub_200_2011()
        _image_list, _name_list = sod_data.imagenet2012_train()
        _image_list, _name_list = sod_data.imagenet2012_val()
        pass

    pass


if __name__ == '__main__':

    sod_data = SODData()
    count = 0
    _image_list, _mask_list, _name_list = sod_data.cssd()
    count += len(_image_list)
    _image_list, _mask_list, _name_list = sod_data.ecssd()
    count += len(_image_list)
    _image_list, _mask_list, _name_list = sod_data.msra_1000_asd()
    count += len(_image_list)
    _image_list, _mask_list, _name_list = sod_data.msra10k()
    count += len(_image_list)
    _image_list, _mask_list, _name_list = sod_data.msra_b()
    count += len(_image_list)
    _image_list, _mask_list, _name_list = sod_data.sed2()
    count += len(_image_list)
    _image_list, _mask_list, _name_list = sod_data.dut_dmron_5166()
    count += len(_image_list)
    _image_list, _mask_list, _name_list = sod_data.dut_dmron_5168()
    count += len(_image_list)
    _image_list, _mask_list, _name_list = sod_data.hku_is()
    count += len(_image_list)
    _image_list, _mask_list, _name_list = sod_data.sod()
    count += len(_image_list)
    _image_list, _mask_list, _name_list = sod_data.thur15000()
    count += len(_image_list)
    _image_list, _mask_list, _name_list = sod_data.pascal1500()
    count += len(_image_list)
    _image_list, _mask_list, _name_list = sod_data.pascal_s()
    count += len(_image_list)
    _image_list, _mask_list, _name_list = sod_data.judd()
    count += len(_image_list)
    _image_list, _mask_list, _name_list = sod_data.duts_te()
    count += len(_image_list)
    _image_list, _mask_list, _name_list = sod_data.duts_tr()
    count += len(_image_list)
    _image_list, _mask_list, _name_list = sod_data.cub_200_2011()
    count += len(_image_list)

    print(count)
    pass