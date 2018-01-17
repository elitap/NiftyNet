import os
import argparse

import numpy as np
import SimpleITK as sitk

def combineLabels(image_itk, dilation_radius = 3):

    image_np = sitk.GetArrayFromImage(image_itk)
    image_np[image_np != 0] = 1.0

    new_image_itk = sitk.GetImageFromArray(image_np)

    cast_img_filter = sitk.CastImageFilter()
    cast_img_filter.SetOutputPixelType(sitk.sitkUInt8)
    new_image_itk = cast_img_filter.Execute(new_image_itk)

    dilation = sitk.BinaryDilateImageFilter()
    dilation.SetKernelType(sitk.sitkBox)
    dilation.SetKernelRadius(dilation_radius)
    dilation.SetForegroundValue(1)
    dilation.SetBackgroundValue(0)
    new_image_itk = dilation.Execute(new_image_itk)

    new_image_itk.SetOrigin(image_itk.GetOrigin())
    new_image_itk.SetDirection(image_itk.GetDirection())
    new_image_itk.SetSpacing(image_itk.GetSpacing())

    castImage = sitk.CastImageFilter()
    castImage.SetOutputPixelType(sitk.sitkFloat32)
    new_image_itk = castImage.Execute(new_image_itk)
    return new_image_itk


def generateForgroundMap(dir, filter):
    for file in os.listdir(dir):
        full_file = os.path.join(dir,file)
        if os.path.isfile(full_file) and filter in file:
            print file
            foreground = combineLabels(sitk.ReadImage(full_file))
            sitk.WriteImage(foreground, os.path.join(dir,file[:10] + "foreground.nrrd"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--dir',
                        required=True,
                        help=("dataset path")
                        )
    parser.add_argument('--filter',
                        required=False,
                        default="out"
                        )

    args = parser.parse_args()
    generateForgroundMap(args.dir, args.filter)