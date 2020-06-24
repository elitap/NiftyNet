import os
import argparse

import shutil
import SimpleITK as sitk
import defs
import uuid
import subprocess
from multiprocessing.pool import ThreadPool
import threading

DMAP_SUBDIR = "dmaps"
TMP_SUBDIR = "tmp"


def create_empty_tmp_dir(dir):
    tmp = os.path.join(dir, TMP_SUBDIR)
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    return tmp


def create_dmap_dir(dir):
    dmap_dir = os.path.join(dir, DMAP_SUBDIR)
    if not os.path.exists(dmap_dir):
        os.makedirs(dmap_dir)


def create_dmap((file, dir, tmp_dir)):
    itk_img = sitk.ReadImage(os.path.join(dir, file), sitk.sitkUInt8)
    dmap_dir = os.path.join(dir, DMAP_SUBDIR)

    np_img = sitk.GetArrayFromImage(itk_img)

    for key, value in defs.LABELS.iteritems():
        np_img_cpy = np_img.copy()
        np_img_cpy[np_img != value] = 0
        np_img_cpy[np_img == value] = 1

        itk_img_save = sitk.GetImageFromArray(np_img_cpy)

        itk_img_save.SetOrigin(itk_img.GetOrigin())
        itk_img_save.SetDirection(itk_img.GetDirection())
        itk_img_save.SetSpacing(itk_img.GetSpacing())

        tmp_file = os.path.join(tmp_dir, str(uuid.uuid4()) + ".nii.gz")
        sitk.WriteImage(itk_img_save, tmp_file)

        filename = os.path.join(dmap_dir, file[:9] + "_" + key + ".nii.gz")
        print threading.currentThread(), filename
        subprocess.check_output(['plastimatch', 'dmap', '--input', tmp_file, '--output', filename])


def split_seg_files(dir, filter, threads):
    tmp_dir = create_empty_tmp_dir(dir)
    create_dmap_dir(dir)
    data = list()
    for f in os.listdir(dir):
        if filter in f:
            data.append([f, dir, tmp_dir])

    t = ThreadPool(threads)
    t.map(create_dmap, data)

    shutil.rmtree(os.path.join(dir, TMP_SUBDIR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--data',
                        required=True,
                        help="directory containing the segmentation files which should be split to individual organs"
                        )
    parser.add_argument('--filter',
                        required=False,
                        default='segmentation',
                        help="filter to find the ")
    parser.add_argument('--threads','-t',
                        required=False,
                        type=int,
                        default=1)

    args = parser.parse_args()
    split_seg_files(args.data, args.filter, args.threads)