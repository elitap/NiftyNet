from niftynet.evaluation.pairwise_measures import PairwiseMeasures

import SimpleITK as sitk
import numpy as np
import os
import argparse

#level2 is already full sized
RESULT_DIR = "output/%d/Test"
#full sized result path for level1
#RESULT_DIR = "output/%d/Test"
GT_FILTER = "segmentation"
ID_IDX = 9

LABELS = {
    "BrainStem.nrrd": 1,
    "Chiasm.nrrd": 2,
    "OpticNerve_L.nrrd": 3,
    "OpticNerve_R.nrrd": 4,
    "Parotid_L.nrrd": 5,
    "Parotid_R.nrrd": 6,
    "Mandible.nrrd": 7
}

#MEASURES = ['ref volume', 'seg volume', 'ref bg volume', 'seg bg volume', 'fp', 'fn', 'tp', 'tn', 'n_intersection', 'n_union', 'sensitivity', 'specificity', 'accuracy', 'fpr', 'dice', 'haus_dist']
MEASURES = ['dice', 'sensitivity', 'specificity']

def getGroundTruth(model, result_file, gt_path):
    #if "half" in model:
    #    gt_postfix = "_2er"
    #if "quarter" in model:
    #   gt_postfix = "_4er"
    id = result_file[:ID_IDX]
    for gt_file in os.listdir(gt_path):
        if GT_FILTER in gt_file and id in gt_file:
            return os.path.join(gt_path, gt_file)
    print "ground truth not found: ", model, result_file
    return None

def getMeasurements(result_file, gt_file, model, checkpoint):
    print "eval", result_file, gt_file, model, checkpoint
    gt_np = sitk.GetArrayFromImage(sitk.ReadImage(gt_file))
    result_np = sitk.GetArrayFromImage(sitk.ReadImage(result_file))

    result_strings = []
    header = None

    for key, value in LABELS.iteritems():
        gt_organ_np = np.zeros_like(gt_np, dtype=np.uint8) #maybeproblem with datatype
        result_organ_np = np.zeros_like(result_np, dtype=np.uint8) #maybeproblem with datatype

        gt_organ_np[gt_np == value] = 1
        result_organ_np[result_np == value] = 1
        measures = PairwiseMeasures(result_organ_np, gt_organ_np, measures=MEASURES)
        if header is None:
            header = "Model,Checkpoint,File,Organ" + measures.header_str()
            print header
        organ_result_string = ("%s,%d,%s,%s," + measures.to_string()) % (model, checkpoint, os.path.split(result_file)[1], key[:-5])
        result_strings.append(organ_result_string)
        print organ_result_string
    return header, result_strings

def evaluate(gt_base_path, result_base_path, result_file):
    results = []
    header = None
    with open(result_file, 'w') as fileptr:
        for result_dir in os.listdir(result_base_path):
            full_result_dir = os.path.join(result_base_path, result_dir)
            #if ("quarter_e-4_48-8_dice_50k_1024s" in result_dir) and os.path.isdir(full_result_dir):
            if ("1024s" in result_dir or "4096s" in result_dir) and os.path.isdir(full_result_dir):
                checkpoints = range(0, 250000, 2000)
                checkpoints.append(49999)
                for checkpoint in checkpoints:
                    checkpoint_dir = os.path.join(full_result_dir, RESULT_DIR) % checkpoint
                    if os.path.isdir(checkpoint_dir):
                        for result_file in os.listdir(checkpoint_dir):
                            full_result_file = os.path.join(checkpoint_dir, result_file)
                            full_gt_file = getGroundTruth(result_dir, result_file, gt_base_path)
                            if full_gt_file is not None:
                                one_file_header, one_file_results = getMeasurements(full_result_file, full_gt_file, result_dir, checkpoint)
                                if header is None:
                                    header = one_file_header
                                    fileptr.write(header + "\n")
                                    fileptr.flush()
                                for result in one_file_results:
                                    fileptr.write(result + "\n")
                                    fileptr.flush()
                                results.append(one_file_results)

                    else:
                        print "Checkpoint not found: ", checkpoint_dir

    return header, results

def writeCSV(header, results):
    with open("results.csv",'w') as fileptr:
        fileptr.write(header + "\n")
        fileptr.flush()
        for result in results:
            fileptr.write(result + "\n")
            fileptr.flush()

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

    args = parser.parse_args()
    header, results = evaluate(args.gtdir, args.modeldir, args.resultfile)
    #writeCSV(header, results)