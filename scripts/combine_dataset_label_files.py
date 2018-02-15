#This script, combines single labels into one file containing all labels

import argparse
import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from defs import LABELS


SPACING = np.array([1.1, 1.1, 3])

VOLUME_FILENAME = "img.nrrd"
VOLUME_POSTFIX = "_volume.nrrd"
LABEL_POSTFIX = "_segmentation.nrrd"

LABEL_DIR = "structures"

GROUND_TRUTH_DIR = "gt"

def resample(spacing, image, isLabel = True):

    scale = image.GetSpacing() / spacing
    newSize = image.GetSize() * scale
    newSize = newSize.round().astype(int).tolist()

    resampler = sitk.ResampleImageFilter()
    if isLabel:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
       resampler.SetInterpolator(sitk.sitkBSpline)

    resampler.SetSize(newSize)
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampledImg = resampler.Execute(image)

    resampledImg.SetOrigin([0, 0, 0])

    #if not isLabel:
    #    print "Resampling done! Before: ", image.GetSize(), image.GetSpacing(), \
    #            " after: ", resampledImg.GetSize(), resampledImg.GetSpacing()

    return resampledImg


def combineLabels(subjdir, outdir, rescale):

    subjectId = os.path.split(subjdir)[1]
    with open(os.path.join(outdir, subjectId+"_summary.txt"), 'w') as fileptr:

        volume = sitk.ReadImage(os.path.join(subjdir, VOLUME_FILENAME))
        if rescale:
            volume = resample(SPACING, volume, False)

        sitk.WriteImage(volume, os.path.join(outdir, subjectId+VOLUME_POSTFIX))

        itklabel = sitk.Image(volume.GetSize(), sitk.sitkUInt8)
        label = sitk.GetArrayFromImage(itklabel)

        label_cnt = 0

        for key, value in LABELS.iteritems():
            labelpath = os.path.join(subjdir, LABEL_DIR, key+".nrrd")

            if os.path.exists(labelpath):
                label_cnt += 1
                #print os.path.join(subjdir, LABEL_DIR, key)
                itkImg = sitk.ReadImage(labelpath)

                if rescale:
                    itkImg = resample(SPACING, itkImg)

                #sitk.WriteImage(itkImg, os.path.join(outdir, subjectId + "_" + key))

                npImg = sitk.GetArrayFromImage(itkImg)

                label[npImg == 1] = value

                fileptr.write(key + "\n")
                #showImg(sitk.GetImageFromArray(label), 0, 9)
                #print key, "added to the label of volume ", id

        itklabel = sitk.GetImageFromArray(label)
        itklabel.SetOrigin(volume.GetOrigin())
        itklabel.SetSpacing(volume.GetSpacing())
        sitk.WriteImage(itklabel, os.path.join(outdir, subjectId + LABEL_POSTFIX))

        print label_cnt, "labels found for ", subjectId


def combineLabelsAll(homedir, outdir, rescale):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for subjectDir in os.listdir(homedir):
        subjectPath = os.path.join(homedir, subjectDir)
        if os.path.isdir(subjectPath):
            combineLabels(subjectPath, outdir, rescale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--datasetpath',
                        required=True,
                        help="dataset path, containing a directory for each subject with a CT image and a subdirectory"
                             "with all the ground truth segmentations for the head and neck organs in separate files."
                        )
    parser.add_argument('--outdir', '-o',
                        required=True,
                        help="output directory where the results are saved to"
                        )
    parser.add_argument('--rescale',
                        action='store_true',
                        help="bring the combined segmentation and the CT volume to a unique spacing"
                        )

    args = parser.parse_args()
    combineLabelsAll(args.datasetpath, args.outdir, args.rescale)




