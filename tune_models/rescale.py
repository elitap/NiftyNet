#calluclates the dice scores of a resultdir containing labeldresult files (all ograns at once), with its ground truth
import argparse
import SimpleITK as sitk
import numpy as np
import os
from original_sizes import ORIG_SIZE


VALID_FILES = [".mha", ".nrrd", ".mhd", ".nii", ".gz"]


def resample(infile, outfile, spacingScale, interpolationtype, origsize):
    itk_img = sitk.ReadImage(infile)


    if origsize:
        id = os.path.split(infile)[1][:9]
        newSize = ORIG_SIZE[id][0]
        newSpacing = ORIG_SIZE[id][1]
    else:
        newSpacing = np.array(itk_img.GetSpacing()) * spacingScale
        newSpacing = np.floor(newSpacing * 10)/10
        imagescale = np.array(itk_img.GetSpacing()) / newSpacing

        newSize = np.array(itk_img.GetSize()) * imagescale

        newSpacing[2] = newSpacing[2] / spacingScale
        newSize[2] = newSize[2] * spacingScale

        newSize = newSize.round().astype(int).tolist()
        newSpacing = newSpacing.tolist()


    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolationtype)

    # resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetSize(newSize)
    resampler.SetOutputSpacing(newSpacing)
    resampler.SetOutputOrigin(itk_img.GetOrigin())
    resampled_img = resampler.Execute(itk_img)

    resampled_img.SetOrigin([0, 0, 0])

    unique = np.unique(sitk.GetArrayFromImage(resampled_img))

    print infile, spacingScale, resampled_img.GetSpacing(), resampled_img.GetSize(), itk_img.GetSize(), unique, len(unique)

    sitk.WriteImage(resampled_img, outfile)


def resampleAll(inpath, outpath, scale, volfilter, origsize):
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    for file in os.listdir(inpath):
        if os.path.splitext(file)[1] in VALID_FILES:
            infile = os.path.join(inpath, file)
            outfile = os.path.join(outpath, file)
            interplationType = sitk.sitkBSpline if volfilter in file else sitk.sitkNearestNeighbor
            resample(infile, outfile, scale, interplationType, origsize)
        else:
            print file, "not a valid file"

def getOriginalMeasurements(path, filter):
    sizeInfo = {}
    for file in os.listdir(path):
        if os.path.splitext(file)[1] in VALID_FILES and filter in file:
            itk_img = sitk.ReadImage(os.path.join(path, file))
            sizeInfo[file[:9]] = (itk_img.GetSize(), itk_img.GetSpacing())

    print sizeInfo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--path',
                        required=True
                        )
    parser.add_argument('--result',
                        required=True,
                        default=''
                        )
    parser.add_argument('--origsize',
                        action='store_true')
    parser.add_argument('--scale',
                        required=False,
                        type=float,
                        default=0.5)
    parser.add_argument('--volumefilter',
                        required=False,
                        default='volume')

    args = parser.parse_args()
    resampleAll(args.path, args.result, args.scale, args.volumefilter, args.origsize)
    #getOriginalMeasurements("/home/elias/Dokumente/head_neck_seg/NiftyNet/data/combined_challenge/Onsite","foreground")



