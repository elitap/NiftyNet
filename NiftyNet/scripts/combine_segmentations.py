import os
import argparse

import numpy as np
import SimpleITK as sitk
import defs

RES_DIR = "../fine_stage/paper/{}/output/50000_{}/orig_size/"

def getSegmentationsFromKey(key, confLst, dataset):
    segmentations = []
    for conf in confLst:
        fstConfDir = RES_DIR.format(confLst[0], dataset)
        for file in os.listdir(fstConfDir):
            if key in file:
                path = os.path.join(fstConfDir, file)
                segmentations.append(sitk.ReadImage(path, sitk.sitkUInt8))
                break

    assert len(confLst) == len(segmentations), "not all segfiles found for " + key
    return segmentations


def majorityVote(segmentations):
    labelForUndecidedPixels = 0
    return sitk.LabelVoting(segmentations, labelForUndecidedPixels)


def staple(segmentations):
    threshold = 0.8

    result_np = sitk.GetArrayFromImage(segmentations[0]).copy()
    result_np[result_np != 0] = 0

    for key, value in defs.LABELS.iteritems():

        singleOrganSeg = []
        for segmentation in segmentations:
            segmentation_np = sitk.GetArrayFromImage(segmentation).copy()
            segmentation_np[segmentation_np != value] = 0
            segmentation_np[segmentation_np == value] = 1
            singleOrganSeg.append(npToOrigITK(segmentation_np, segmentation))


        reference_segmentation_STAPLE_probabilities = sitk.STAPLE(singleOrganSeg, value)
        reference_segmentation_STAPLE = reference_segmentation_STAPLE_probabilities > threshold
        STAPLE_np = sitk.GetArrayFromImage(reference_segmentation_STAPLE)
        result_np[STAPLE_np == 1] = value

    return npToOrigITK(result_np, segmentations[0])


def combineSegmentations(configFile, resultpath, dataset):

    if not os.path.exists(resultpath):
        os.makedirs(resultpath)

    confLst = []
    with open(configFile, "r") as fptr:
        for config in fptr:
            if len(config) == 0 or config[0] == '#':
                continue
            confLst.append(config.rstrip().replace('.ini', ''))

    fstConfDir = RES_DIR.format(confLst[0], dataset)
    for file in os.listdir(fstConfDir):
        print file
        key = file[:9]
        segmentations = getSegmentationsFromKey(key, confLst, dataset)

        result = staple(segmentations)
        #result = majorityVote(segmentations)

        outfile = os.path.join(resultpath, key+"_out.nii.gz")
        sitk.WriteImage(result, outfile)


def npToOrigITK(np_image, orig_itk_img):
    new_itk = sitk.GetImageFromArray(np_image)
    new_itk.SetOrigin(orig_itk_img.GetOrigin())
    new_itk.SetDirection(orig_itk_img.GetDirection())
    new_itk.SetSpacing(orig_itk_img.GetSpacing())

    cast_img_filter = sitk.CastImageFilter()
    cast_img_filter.SetOutputPixelType(sitk.sitkUInt8)
    new_itk = cast_img_filter.Execute(new_itk)

    return new_itk


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--configfile',
                        required=True,
                        help="file containing the models to be consider for the majority voting"
                        )
    parser.add_argument('--resdir',
                        required=True,
                        help="directory where the results are written to"
                        )
    parser.add_argument('--dataset',
                        choices=['train', 'test'],
                        default='test'
                        )

    args = parser.parse_args()
    combineSegmentations(args.configfile, args.resdir, args.dataset)