import SimpleITK as sitk
import numpy as np
import os
import argparse
import subprocess
import uuid
from defs import LABELS
from defs import UP_SCALE_SUBDIR

RESULT_SUB_DIR = "output/%s/" + UP_SCALE_SUBDIR
GT_FILTER = "segmentation"
ID_IDX = 9

MEASURES = ['dice', 'haus_dist']

def getGroundTruth(model, result_file, gt_path):
    id = result_file[:ID_IDX]
    for gt_file in os.listdir(gt_path):
        if GT_FILTER in gt_file and id in gt_file:
            return os.path.join(gt_path, gt_file)
    print "ground truth not found: ", model, result_file
    return None

def writeHeader(use_plastimatch):
    if use_plastimatch:
        header = "Model,Checkpoint,File,Organ,dice,95haus_dist,avghaus_dist"
    else:
        header = "Model,Checkpoint,File,Organ" + ",".join(MEASURES)
    print header
    return header


def getPlastimatchResults(gt_itk, gt_organ_np, result_itk, result_organ_np):
    gt2save_itk = sitk.GetImageFromArray(gt_organ_np)
    gt2save_itk.SetSpacing(gt_itk.GetSpacing())
    gt2save_itk.SetOrigin(gt_itk.GetOrigin())
    gt2save_itk.SetDirection(gt_itk.GetDirection())

    tmp_gt = str(uuid.uuid4()) + ".nrrd"
    sitk.WriteImage(gt2save_itk, tmp_gt)

    res2save_itk = sitk.GetImageFromArray(result_organ_np)
    res2save_itk.SetSpacing(result_itk.GetSpacing())
    res2save_itk.SetOrigin(result_itk.GetOrigin())
    res2save_itk.SetDirection(result_itk.GetDirection())

    tmp_res = str(uuid.uuid4()) + ".nrrd"
    sitk.WriteImage(res2save_itk, tmp_res)

    plasti_res = subprocess.check_output(['plastimatch', 'dice', '--all', tmp_gt, tmp_res])

    os.remove(tmp_gt)
    os.remove(tmp_res)

    dice = ''
    hd95 = ''
    hdavg = ''
    for line in plasti_res.split('\n'):
        if 'DICE:' in line:
            dice = line.rstrip().split(' ')[-1]
        if "Avg average Hausdorff distance (boundary)" in line:
            hdavg = line.rstrip().split(' ')[-1]
        if "Percent (0.95) Hausdorff distance (boundary)" in line:
            hd95 = line.rstrip().split(' ')[-1]

    try:
        float(dice)
        # 724 diagonal distance in pix of a 512*512 img
        if float(hdavg) > 724.0:
            hdavg = "724"
        if float(hd95) > 724.0:
            hd95 = "724"
    except ValueError:
        print "Not a float"
        return

    res_string = dice + "," + hd95 + "," + hdavg
    return res_string


def getMeasurements(result_file, gt_file, model, checkpoint, use_plastimatch):
    print "eval", result_file, gt_file, model, checkpoint
    gt_itk = sitk.ReadImage(gt_file)
    gt_np = sitk.GetArrayFromImage(gt_itk)
    result_itk = sitk.ReadImage(result_file)
    result_np = sitk.GetArrayFromImage(result_itk)

    print gt_itk.GetSpacing(), result_itk.GetSpacing()
    np.testing.assert_almost_equal(gt_itk.GetSpacing(), result_itk.GetSpacing(), 5, "Spacing dimension does not match")

    result_strings = []
    header = None

    for key, value in LABELS.iteritems():
        gt_organ_np = np.zeros_like(gt_np, dtype=np.uint8) #maybeproblem with datatype
        result_organ_np = np.zeros_like(result_np, dtype=np.uint8) #maybeproblem with datatype

        gt_organ_np[gt_np == value] = 1
        result_organ_np[result_np == value] = 1

        if header is None:
            header = writeHeader(use_plastimatch)

        res_string = ''
        if use_plastimatch:
            res_string = getPlastimatchResults(gt_itk, gt_organ_np, result_itk, result_organ_np)
        else:
            #uses NiftyNet result measurements
            from niftynet.evaluation.pairwise_measures import PairwiseMeasures
            measures = PairwiseMeasures(result_organ_np, gt_organ_np, measures=MEASURES, pixdim=gt_itk.GetSpacing())
            res_string = measures.to_string()

        organ_result_string = ("%s,%s,%s,%s," + res_string) % (model, checkpoint, os.path.split(result_file)[1], key)
        result_strings.append(organ_result_string)
        print organ_result_string
    return header, result_strings


def evaluate(gt_base_path, result_base_path, result_file, single_checkpoint, use_plastimatch):
    header = None

    with open(result_file, 'w') as fileptr:
        for result_dir in os.listdir(result_base_path):

            result_sub_dir = RESULT_SUB_DIR
            if (not "half" in result_dir) and (not "quarter" in result_dir):
                result_sub_dir = os.path.split(result_sub_dir)[0]

            full_result_dir = os.path.join(result_base_path, result_dir)

            checkpoints = []
            if single_checkpoint is None:
                checkpoints = range(32000, 50001, 2000)
            else:
                checkpoints.append(single_checkpoint)

            for checkpoint in checkpoints:
                checkpoint_dir = os.path.join(full_result_dir, result_sub_dir) % str(checkpoint)
                if os.path.isdir(checkpoint_dir):
                    for result_file in os.listdir(checkpoint_dir):
                        full_result_file = os.path.join(checkpoint_dir, result_file)
                        full_gt_file = getGroundTruth(result_dir, result_file, gt_base_path)
                        if full_gt_file is not None:
                            one_file_header, one_file_results = getMeasurements(full_result_file, full_gt_file, result_dir, checkpoint, use_plastimatch)
                            if header is None:
                                header = one_file_header
                                fileptr.write(header + "\n")
                                fileptr.flush()
                            for result in one_file_results:
                                fileptr.write(result + "\n")
                                fileptr.flush()
                else:
                    print "Checkpoint not found: ", checkpoint_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--modeldir',
                        required=True
                        )
    parser.add_argument('--gtdir',
                        required=True
                        )
    parser.add_argument('--resultfile',
                        required=True
                        )
    parser.add_argument('--checkpoint',
                        required=False,
                        type=str,
                        default=None
                        )
    parser.add_argument('--useplastimatch',
                        action='store_true')

    args = parser.parse_args()
    evaluate(args.gtdir, args.modeldir, args.resultfile, args.checkpoint, args.useplastimatch)