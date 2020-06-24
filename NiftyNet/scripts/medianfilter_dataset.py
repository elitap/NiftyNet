import argparse
import SimpleITK as sitk
import numpy as np
import os

def applay_meanfilter(datasetpath, resultpath, filter):

    if not os.path.exists(resultpath):
        os.mkdir(resultpath)


    for file in os.listdir(datasetpath):
        if filter in file:
            itk_img = sitk.ReadImage(os.path.join(datasetpath, file))

            medianfilter = sitk.MedianImageFilter()
            medianfilter.SetRadius(1)

            itk_res_img = medianfilter.Execute(itk_img)
            sitk.WriteImage(itk_res_img, os.path.join(resultpath, file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--dataset',
                        required=True,
                        help="dataset path"
                        )
    parser.add_argument('--resultpath',
                        required=True,
                        help="dataset path"
                        )
    parser.add_argument('--filter',
                        required=False,
                        default='volume',
                        help="identifier for the image to filter"
                        )

    args = parser.parse_args()
    applay_meanfilter(args.dataset, args.resultpath, args.filter)
