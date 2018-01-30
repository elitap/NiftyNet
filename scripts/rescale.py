#calluclates the dice scores of a resultdir containing labeldresult files (all ograns at once), with its ground truth
import argparse
import SimpleITK as sitk
import numpy as np
import os
from defs import ORIG_SIZE
from defs import UP_SCALE_SUBDIR


VALID_FILES = [".mha", ".nrrd", ".mhd", ".nii", ".gz"]

RESULTDIR = "output/%s"

def resample(infile, outfile, spacingScale, interpolationtype, origsize):
    itk_img = sitk.ReadImage(infile)

    if origsize:
        id = os.path.split(infile)[1][:9]
        newSize = ORIG_SIZE[id][0]
        newSpacing = ORIG_SIZE[id][1]
    else:
        newSpacing = np.array(itk_img.GetSpacing()) * spacingScale
        #newSpacing = np.floor(newSpacing * 10)/10
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

    print infile, spacingScale, resampled_img.GetSpacing(), resampled_img.GetSize(), itk_img.GetSize(), unique, len(unique), "interpolation type: ", interpolationtype

    sitk.WriteImage(resampled_img, outfile)


def resampleFolder(inpath, outpath, scale=2, volfilter="volume", origsize=True):
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


def resampleAllModelsToOrigsize(model_dir, single_checkpoint):
    for model in os.listdir(model_dir):
        full_result_dir = os.path.join(model_dir, model)
        if ("" in model) and os.path.isdir(full_result_dir):
            checkpoints = []
            if single_checkpoint is None:
                checkpoints = range(32000, 50001, 2000)
            else:
                checkpoints.append(single_checkpoint)
            for checkpoint in checkpoints:
                checkpoint_dir = os.path.join(full_result_dir, RESULTDIR) % str(checkpoint)
                if os.path.exists(checkpoint_dir):
                    resampleFolder(checkpoint_dir, os.path.join(checkpoint_dir, UP_SCALE_SUBDIR))
                else:
                    print "Checkpoint not found: ", checkpoint_dir

            if os.path.exists(checkpoint_dir):
                resampleFolder(checkpoint_dir, os.path.join(checkpoint_dir, UP_SCALE_SUBDIR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--datasetpath',
                        required=True,
                        help="path to directory with images to rescale or the model directory if the results of "
                             "more models are to rescale"
                        )
    parser.add_argument('--checkpoint',
                        required=False,
                        type=str,
                        default=None
                        )
    parser.add_argument('--result',
                        required=False,
                        default='',
                        help="path to directory to save the rescaled images"
                        )
    parser.add_argument('--scale',
                        required=False,
                        type=float,
                        default=-1)
    parser.add_argument('--volumefilter',
                        required=False,
                        default='volume',
                        help="filter for volumes to change the interpolation correctly")



    args = parser.parse_args()
    if args.scale == -1:
        resampleAllModelsToOrigsize(args.path, args.checkpoint)
    else:
        resampleFolder(args.path, args.result, args.scale, args.volumefilter, False)




