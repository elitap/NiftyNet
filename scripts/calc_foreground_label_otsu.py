import os
import argparse

import numpy as np
import SimpleITK as sitk

def getForeground(image_itk):
    otsu_itk = sitk.OtsuMultipleThresholds(image_itk)

    connected_itk = sitk.ConnectedComponent(otsu_itk)
    connected_np = sitk.GetArrayFromImage(connected_itk)

    connected_itk = sitk.ConnectedComponent(otsu_itk)
    connected_np = sitk.GetArrayFromImage(connected_itk)

    uniq, count = np.unique(connected_np.flatten(), return_counts=True)
    foreground_idx = np.argsort(count)[-2]
    largest_component = uniq[foreground_idx]  # the foreground should be the second largest component
    largest_component_cnt = count[foreground_idx]
    print "uniq labels after otsu", uniq[:4], count[:4], foreground_idx, largest_component

    foreground_percentage = float(largest_component_cnt) / float(connected_np.flatten().size)
    print("[Selective foreground sampling] foreground percentage: " + str(foreground_percentage))
    assert foreground_percentage > 0.05, "less then 5 percent of the pixels were declared as foreground"
    assert foreground_percentage < 0.5, "more then 50 percent of the pixels are declared as foreground"

    # this line caused problems if the order was the other way around!! why??
    connected_np[connected_np != largest_component] = 0
    connected_np[connected_np == largest_component] = 1

    uniq, count = np.unique(connected_np.flatten(), return_counts=True)
    print "uniq count after thresholding: ", uniq[:4], count[:4]

    connected_itk = sitk.GetImageFromArray(connected_np)
    connected_itk.SetSpacing(image_itk.GetSpacing())
    connected_itk.SetOrigin(image_itk.GetOrigin())
    connected_itk.SetDirection(image_itk.GetDirection())

    # connected_itk = sitk.BinaryThreshold(connected_itk, largest_component - 0.1, largest_component + 0.1) <- this line!!!!

    castImage = sitk.CastImageFilter()
    castImage.SetOutputPixelType(sitk.sitkFloat32)
    connected_itk = castImage.Execute(connected_itk)

    #connected_np = sitk.GetArrayFromImage(connected_itk)
    return connected_itk


def gnerateForgroundMap(dir, filter):
    for file in os.listdir(dir):
        full_file = os.path.join(dir,file)
        if os.path.isfile(full_file) and filter in file:
            print file
            foreground = getForeground(sitk.ReadImage(full_file))
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
    gnerateForgroundMap(args.dir, args.filter)