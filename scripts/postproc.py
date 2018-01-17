import argparse
import SimpleITK as sitk
import numpy as np
import os

POST_PROC_RES_DIR = "postprocess"


def postproc(resdir, postprocdir, forgrounddir, foreground_filter, dilation_radius):
    if not os.path.exists(postprocdir):
        os.mkdir(postprocdir)
    for res_file in os.listdir(resdir):
        full_file = os.path.join(resdir, res_file)
        if os.path.isfile(full_file):
            id = res_file[0:9]
            
            for foreground in os.listdir(forgrounddir):
                full_foreground = os.path.join(forgrounddir, foreground)
                if os.path.isfile(full_foreground) and id in foreground and foreground_filter in foreground:
                    
                    foreground_itk = sitk.ReadImage(full_foreground)
                    res_itk = sitk.ReadImage(full_file)
                    np.testing.assert_almost_equal(foreground_itk.GetSpacing(), res_itk.GetSpacing(), 5,
                                                   "Spacing dimension does not match")
                    np.testing.assert_almost_equal(foreground_itk.GetSize(), res_itk.GetSize(), 5,
                                                   "Size dimension does not match")


                    cast_img_filter = sitk.CastImageFilter()
                    cast_img_filter.SetOutputPixelType(sitk.sitkUInt8)
                    foreground_itk = cast_img_filter.Execute(foreground_itk)

                    dilation = sitk.BinaryDilateImageFilter()
                    dilation.SetKernelType(sitk.sitkBox)
                    dilation.SetKernelRadius(dilation_radius)
                    dilation.SetForegroundValue(1)
                    dilation.SetBackgroundValue(0)
                    foreground_itk = dilation.Execute(foreground_itk)

                    foreground_np = sitk.GetArrayFromImage(foreground_itk)
                    res_np = sitk.GetArrayFromImage(res_itk)

                    res_np[foreground_np == 0] = 0

                    new_res_itk = sitk.GetImageFromArray(res_np)
                    new_res_itk.SetSpacing(res_itk.GetSpacing())
                    new_res_itk.SetOrigin(res_itk.GetOrigin())
                    new_res_itk.SetDirection(res_itk.GetDirection())

                    new_file = os.path.join(postprocdir, res_file[0:9] + "_out_postproc.nii.gz")
                    print new_file
                    sitk.WriteImage(new_res_itk, new_file)


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
    parser.add_argument('--foreground',
                        required=True,
                        help="result dir containing the filter (coarse stage results)"
                        )
    parser.add_argument('--foreground_filter',
                        required=False,
                        default='foreground',
                        help='filter for the files in the foreground dir'
                        )
    parser.add_argument('--dilation',
                        required=False,
                        type=int,
                        default=5)

    args = parser.parse_args()
    postproc(args.resdir, args.postprocdir, args.foreground, args.foreground_filter, args.dilation)