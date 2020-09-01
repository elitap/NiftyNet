import os
import re
import monai
import torch
import numpy as np
import shutil

from torch.utils.data.sampler import Sampler
from monai.data import DataLoader, list_data_collate
from glob import glob

from numpy import pi
from scipy.ndimage.morphology import binary_dilation

from monai.transforms import (
    Compose,
    LoadNiftid,
    AddChanneld,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    BorderPadd,
    RandSpatialCropd,
    AsChannelFirstd,
    RandAffined,
    Spacingd,
    ToTensord,
    ToNumpyd, MapTransform, KeysCollection, Transform)

CKP_FILE_MASK = "checkpoint_epoch={}.pth"
ALLOWED_STAGES = ("coarse", "fine")

DEFAULT_KEYS = {'img': "volume", 'label': "segmentation", 'mask': "foreground"}
CACHE_DATA = "../data/miccai/full_dataset_nifti/cache"
TRAIN_DATA = "../data/miccai/full_dataset_nifti/train"
TEST_DATA = "../data/miccai/full_dataset_nifti/test"


class Dilate(Transform):

    def __init__(self, radius: int) -> None:
        self.radius = radius

    def __call__(self, img):
        """
        Args:
            img (ndarray): channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        dilated = list()
        for channel in img:
            dilated.append(binary_dilation(channel, iterations=self.radius))
        return np.stack(dilated).astype(img.dtype)


class DilateMaskd(MapTransform):

    def __init__(self, keys: KeysCollection, radius: int = 0) -> None:

        super().__init__(keys)
        self.dilater = Dilate(radius)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d.keys():
                d[key] = self.dilater(d[key])
        return d


class RandomRepeatingSampler(Sampler):

    def __init__(self, data_source, replacement=False, num_samples=None, repeat=1):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self._repeat = repeat

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

        if not isinstance(self._repeat, int) or self._repeat <= 1:
            raise ValueError("repeat should be a integer greater one "
                             "value, but got num_samples={}".format(self.repeat))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return self._repeat * len(self.data_source)
        return self._repeat * self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(np.repeat(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist(),
                                  self._repeat))
        return iter(np.repeat(torch.randperm(n).tolist(), self._repeat))

    def __len__(self):
        return self.num_samples


class DLCreator():

    def __init__(self,
                 data: str = None,
                 keys: dict = None,
                 train: bool = True,
                 eval_train: bool = False,
                 stage: str = "coarse",
                 fine_stage_mask_path: str = ""):

        if data is None:
            if train or eval_train:
                self.data_path = os.path.abspath(TRAIN_DATA)
            else:
                self.data_path = os.path.abspath(TEST_DATA)
        else:
            self.data_path = data

        self.keys = keys
        if keys is None:
            self.keys = DEFAULT_KEYS

        self.train = train
        self.stage = stage
        self.fine_stage_mask_path = os.path.abspath(fine_stage_mask_path)
        self.fine_stage_dil_radius = 13

        if stage is "coarse":
            self.pixdim = (2.2, 2.2, 2.2)
            self.spatial_size = [16, 16, 16]
            self.num_samples = 72
        else:
            self.pixdim = (1.1, 1.1, 2.2)
            self.spatial_size = [24, 24, 24]
            self.num_samples = 24
        self.num_worker = 10

    def get_data_loader(self):
        if self.stage not in ALLOWED_STAGES:
            raise ValueError("stage {} not defined".format(self.stage))
        if self.stage is "fine" and not os.path.exists(self.fine_stage_mask_path):
            raise ValueError("foreground for fine stage not set, train the coarse stage first")

        # Setup transforms, dataset
        images = sorted(glob(os.path.join(self.data_path, '*' + self.keys['img'] + '.nii.gz')))
        segs = sorted(glob(os.path.join(self.data_path, '*' + self.keys['label'] + '.nii.gz')))
        add_channel_transform = AddChanneld(keys=["img", "seg", "mask"])
        # this does basically nothing
        as_channel_first = AsChannelFirstd(keys=["mask"], channel_dim=0)
        dilate_mask = DilateMaskd(keys=["none"])

        if self.stage is "coarse":
            masks = sorted(glob(os.path.join(self.data_path, '*' + self.keys['mask'] + '.nii.gz')))
        else:
            masks = sorted(glob(os.path.join(self.fine_stage_mask_path, '*.nii.gz')))
            add_channel_transform = AddChanneld(keys=["img", "seg"])
            as_channel_first = AsChannelFirstd(keys=["mask"])
            dilate_mask = DilateMaskd(keys=["mask"], radius=self.fine_stage_dil_radius)

        files = [{"img": img, "seg": seg, "mask": mask} for img, seg, mask in zip(images, segs, masks)]


        dl = None

        if self.train:
            # define transforms for image and segmentation
            # TODO possible write a dilation transform for the fine stage
            transforms = Compose(
                [
                    LoadNiftid(keys=["img", "seg", "mask"]),
                    add_channel_transform,
                    as_channel_first,
                    Spacingd(keys=['img', 'seg', "mask"], pixdim=self.pixdim,
                             mode=('bilinear', 'nearest', 'nearest')),
                    ScaleIntensityd(keys=["img"]),
                    dilate_mask,
                    BorderPadd(keys=["img", "seg", "mask"], spatial_border=self.spatial_size[0]),
                    RandAffined(keys=["img", "seg", "mask"], mode=('bilinear', 'nearest', 'nearest'), prob=0.6,
                                rotate_range=[5.0*pi/180, 5.0*pi/180, 5.0*pi/180], scale_range=[0.08, 0.08, 0.08]),
                    ToNumpyd(keys=["img", "seg", "mask"]),
                    RandCropByPosNegLabeld(
                        keys=["img", "seg", "mask"], label_key="mask",
                        spatial_size=self.spatial_size, pos=1, neg=0,
                        num_samples=self.num_samples
                    ),
                    ToTensord(keys=["img", "seg"]),
                ]
            )

            # ds = monai.data.Dataset(data=files, transform=transforms)
            # ds = monai.data.PersistentDataset(data=files, transform=transforms, cache_dir=CACHE_DATA)
            ds = monai.data.CacheDataset(data=files, transform=transforms)


            # batch size is set by the random corp pos neg transform
            dl = DataLoader(ds,
                            # samples 15 times from the same image so in the end
                            # 15 * 72 = 1080 samples are drawn from each image
                            sampler=RandomRepeatingSampler(ds, repeat=15),
                            num_workers=self.num_worker,
                            pin_memory=torch.cuda.is_available())
        else:
            transforms = Compose(
                [
                    LoadNiftid(keys=["img", "seg", "mask"]),
                    add_channel_transform,
                    as_channel_first,
                    Spacingd(keys=['img', 'seg', "mask"], pixdim=self.pixdim,
                             mode=('bilinear', 'nearest', 'nearest')),
                    ScaleIntensityd(keys=["img"]),
                    dilate_mask,
                    ToTensord(keys=["img", "seg", "mask"]),
                ]
            )
            ds = monai.data.Dataset(data=files, transform=transforms)
            # batch size is set by the sliding window inference
            dl = DataLoader(ds,
                            num_workers=self.num_worker,
                            pin_memory=torch.cuda.is_available())

        # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
        # fuck read that comment, batch size set to 1 draws n_namples so 72
        return dl


def get_checkpoint_file(dir: str, epoch: int = -1):
    ckpts = list()
    for ckp in os.listdir(dir):
        ckp_id = re.search(r'.*epoch=(\d+)', ckp)
        if ckp_id:
            ckpts.append(int(ckp_id.group(1)))

    if len(ckpts) == 0:
        return [None, -1]

    if epoch == -1:
        return [os.path.join(dir, CKP_FILE_MASK.format(max(ckpts))), max(ckpts)]

    # get closest ckp to specified iter
    ckp = min(ckpts, key=lambda x: abs(x - epoch))
    return [os.path.join(dir, CKP_FILE_MASK.format(ckp)), ckp]

def remove_subfolders(dir: str):
    for subfolder in os.listdir(dir):
        subfolder_full = os.path.join(dir, subfolder)
        if os.path.isdir(subfolder_full) and len(os.listdir(subfolder_full)) == 1:
            for subfile in os.listdir(subfolder_full):
                subfile_full = os.path.join(subfolder_full, subfile)
                os.renames(subfile_full, os.path.join(dir, subfile))


def gen_foreground(train_data: str, test_data: str, keys: dict() = DEFAULT_KEYS):
    try:
        import SimpleITK as sitk
    except:
        raise ImportError("SimpleITK not installed to use the automatic foreground"
                          "generation SimpleITK need to be installed")

    print("Generating foreground mask ...")

    if keys is None:
        keys = DEFAULT_KEYS
    if train_data is None:
        train_data = TRAIN_DATA
    if test_data is None:
        test_data = TEST_DATA

    def _run_thresh(data):
        for img in os.listdir(data):
            if keys["img"] in img:
                sitk_img = sitk.ReadImage(os.path.join(data, img))
                otsu_itk = sitk.OtsuMultipleThresholds(sitk_img)
                new_img = img.replace(keys["img"], keys["mask"])
                sitk.WriteImage(sitk.Cast(otsu_itk, sitk.sitkUInt8),
                                os.path.join(data, new_img))

    _run_thresh(train_data)
    _run_thresh(test_data)

