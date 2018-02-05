import argparse
import SimpleITK as sitk
import numpy as np
import os
from defs import LABELS

POST_PROC_RES_DIR = "postprocess"


def filter_using_coarse_out(res_itk, filter_itk, dilation_radius):
    cast_img_filter = sitk.CastImageFilter()
    cast_img_filter.SetOutputPixelType(sitk.sitkUInt8)
    filter_itk = cast_img_filter.Execute(filter_itk)
    res_itk_new = cast_img_filter.Execute(res_itk)

    res_np = sitk.GetArrayFromImage(res_itk_new)
    filter_np = sitk.GetArrayFromImage(filter_itk)
    for key, value in LABELS.iteritems():
        filter_cp = np.copy(filter_np)
        filter_cp[filter_cp != value] = 0
        filter_cp[filter_cp == value] = 1

        filter_itk = sitk.GetImageFromArray(filter_cp)
        filter_itk = cast_img_filter.Execute(filter_itk)
        dilation = sitk.BinaryDilateImageFilter()
        dilation.SetKernelType(sitk.sitkBox)
        dilation.SetKernelRadius(dilation_radius)
        dilation.SetForegroundValue(1)
        dilation.SetBackgroundValue(0)
        filter_itk = dilation.Execute(filter_itk)

        filter_cp = sitk.GetArrayFromImage(filter_itk)

        res_np[(filter_cp == 0) & (res_np == value)] = 0

    new_res_itk = sitk.GetImageFromArray(res_np)
    new_res_itk.SetSpacing(res_itk.GetSpacing())
    new_res_itk.SetOrigin(res_itk.GetOrigin())
    new_res_itk.SetDirection(res_itk.GetDirection())

    return new_res_itk


def postproc(resdir, postprocdir, filter_dir, coarse_stage_out_name_filter, dilation_radius):
    if not os.path.exists(postprocdir):
        os.mkdir(postprocdir)
    for res_file in os.listdir(resdir):
        full_file = os.path.join(resdir, res_file)
        if os.path.isfile(full_file):
            id = res_file[0:9]

            for filter in os.listdir(filter_dir):
                full_filter = os.path.join(filter_dir, filter)
                if os.path.isfile(full_filter) and id in filter and coarse_stage_out_name_filter in filter:

                    filter_itk = sitk.ReadImage(full_filter)
                    res_itk = sitk.ReadImage(full_file)
                    np.testing.assert_almost_equal(filter_itk.GetSpacing(), res_itk.GetSpacing(), 5,
                                                   "Spacing dimension does not match")
                    np.testing.assert_almost_equal(filter_itk.GetSize(), res_itk.GetSize(), 5,
                                                   "Size dimension does not match")

                    new_res_itk = filter_using_coarse_out(filter_itk, res_itk, dilation_radius)

                    new_file = os.path.join(postprocdir, res_file[0:9] + "_out_postproc.nii.gz")
                    print new_file
                    sitk.WriteImage(new_res_itk, new_file)
                else:
                    if not coarse_stage_out_name_filter in filter:
                        print coarse_stage_out_name_filter, "not found in", filter


def old_postproc(filter_itk, res_itk, dilation_radius):
    cast_img_filter = sitk.CastImageFilter()
    cast_img_filter.SetOutputPixelType(sitk.sitkUInt8)
    filter_itk = cast_img_filter.Execute(filter_itk)

    dilation = sitk.BinaryDilateImageFilter()
    dilation.SetKernelType(sitk.sitkBox)
    dilation.SetKernelRadius(dilation_radius)
    dilation.SetForegroundValue(1)
    dilation.SetBackgroundValue(0)
    filter_itk = dilation.Execute(filter_itk)
    foreground_np = sitk.GetArrayFromImage(filter_itk)
    res_np = sitk.GetArrayFromImage(res_itk)
    res_np[foreground_np == 0] = 0

    new_res_itk = sitk.GetImageFromArray(res_np)
    new_res_itk.SetSpacing(res_itk.GetSpacing())
    new_res_itk.SetOrigin(res_itk.GetOrigin())
    new_res_itk.SetDirection(res_itk.GetDirection())

    return new_res_itk


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--resdir',
                        required=True,
                        help="result directory containing the fine stage results"
                        )
    parser.add_argument('--postprocdir',
                        required=True,
                        help="directory the results are written to"
                        )
    parser.add_argument('--coarse_stage_out',
                        required=True,
                        help="result dir containing the filter (coarse stage results)"
                        )
    parser.add_argument('--coarse_stage_out_name_filter',
                        required=False,
                        default='out',
                        help='filter for the files in the foreground dir'
                        )
    parser.add_argument('--dilation',
                        required=False,
                        type=int,
                        default=5)

    args = parser.parse_args()
    postproc(args.resdir, args.postprocdir, args.coarse_stage_out, args.coarse_stage_out_name_filter, args.dilation)