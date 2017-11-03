import os
import argparse

import numpy as np
import SimpleITK as sitk

def combineLabels(image_itk):

    image_np = sitk.GetArrayFromImage(image_itk)
    image_np[image_np != 0] = 1.0

    new_image_itk = sitk.GetImageFromArray(image_np.astype(np.float32))
    new_image_itk.SetOrigin(image_itk.GetOrigin())
    new_image_itk.SetDirection(image_itk.GetDirection())
    new_image_itk.SetSpacing(image_itk.GetSpacing())
    #image_itk = sitk.BinaryThreshold(image_itk, 1 - 0.1, 9 + 0.1, 1, 0)

    castImage = sitk.CastImageFilter()
    castImage.SetOutputPixelType(sitk.sitkFloat32)
    image_itk = castImage.Execute(image_itk)
    return new_image_itk


def generateForgroundMap(dir, filter):
    for file in os.listdir(dir):
        full_file = os.path.join(dir,file)
        if os.path.isfile(full_file) and filter in file:
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
                        default="volume"
                        )

    args = parser.parse_args()

    generateForgroundMap(args.dir, args.filter)