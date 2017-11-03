import os
import argparse

import numpy as np
import SimpleITK as sitk

def getForeground(image_itk):
    otsu_itk = sitk.OtsuMultipleThresholds(image_itk)

    connected_itk = sitk.ConnectedComponent(otsu_itk)
    connected_np = sitk.GetArrayFromImage(connected_itk)

    uniq, count = np.unique(connected_np.flatten(), return_counts=True)
    foreground_idx = np.argsort(count)[-2]
    largest_component = uniq[foreground_idx]  # the foreground should be the second largest component
    largest_component_cnt = count[foreground_idx]

    foreground_percentage = float(largest_component_cnt) / float(connected_np.size)
    print("[Selective foreground sampling] foreground percentage: " + str(foreground_percentage))
    assert foreground_percentage > 0.05, "less then 5 percent of the pixels were declared as foreground"

    connected_itk = sitk.BinaryThreshold(connected_itk, largest_component - 0.1, largest_component + 0.1)

    castImage = sitk.CastImageFilter()
    castImage.SetOutputPixelType(sitk.sitkFloat32)
    connected_itk = castImage.Execute(connected_itk)

    #connected_np = sitk.GetArrayFromImage(connected_itk)
    return connected_itk


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
    getForeground(args.dir, args.filter)