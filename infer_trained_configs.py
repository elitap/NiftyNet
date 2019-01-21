import subprocess
import argparse
import os
import sys
import time

COARSE_STAGE = "coarse_stage"
FINE_STAGE = "fine_stage"

DATASET_SPLIT_FILE = "./data/HaN_MICCAI2015_Dataset/test_data_split.csv"
MODEL_BASE = "./%s/%s/models/model.ckpt-%d.index"
CONFIG_BASE = "./config/%s_configs"
LOGFILE = "./%s/log/inferencelog_gpu%d.txt"

INFERENCE_CMD = "python net_segment.py inference -c %s --save_seg_dir %s --inference_iter %d --cuda_devices %d --dataset_split_file %s"
SAVE_DIR_BASE = "output"


def execute(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Poll process for new output until finished
    while True:
        nextline = process.stdout.readline()
        if nextline == '' and process.poll() is not None:
            break
        sys.stdout.write(nextline)
        sys.stdout.flush()

    output = process.communicate()[0]
    exitCode = process.returncode
    return exitCode


def get_last_checkpoint(config, stage):
    checkpoints = range(0, 250001, 2000)

    last_checkpoint_path = ''
    last_checkpoint = 0
    for checkpoint in checkpoints:
        checkpoint_path = MODEL_BASE % (stage, os.path.splitext(config)[0], checkpoint)
        if os.path.exists(checkpoint_path):
            last_checkpoint_path = checkpoint_path
            last_checkpoint = checkpoint

    return last_checkpoint_path, last_checkpoint


def run_inference(configs, gpu, dataset_splitfile, single_checkpoint, seg_postfix, stage):
    log = (LOGFILE) % (stage, gpu)

    logsplit = os.path.split(log)
    if not os.path.exists(logsplit[0]):
        os.makedirs(logsplit[0])

    with open(log, 'w') as logptr:
        with open(configs, "r") as fptr:
            for config in fptr:
                if len(config) == 0 or config[0] == '#':
                    continue

                config = config.strip()
                full_config = os.path.join((CONFIG_BASE % (stage)), config)

                checkpoints = []
                if single_checkpoint == -1:
                    #checkpoint_path, checkpoint = get_last_checkpoint(config, stage)
                    checkpoints = range(32000, 50001, 2000)
                else:
                    checkpoints.append(single_checkpoint)
                for checkpoint in checkpoints:
                    checkpoint_path = MODEL_BASE % (stage, os.path.splitext(config)[0], checkpoint)

                    if os.path.exists(checkpoint_path):
                        cmd = INFERENCE_CMD % (full_config, os.path.join(SAVE_DIR_BASE, str(checkpoint)+seg_postfix), checkpoint, gpu, os.path.abspath(dataset_splitfile))
                        logptr.write("execute " + cmd + " \n")
                        logptr.flush()
                        start = time.time()
                        execute(cmd)
                        logptr.write("execution time %0.3f \n" % ((time.time()-start)*1000.0))
                    else:
                        print "checkpoint not found", checkpoint_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--config',
                        required=True
                        )
    parser.add_argument('--gpu',
                        required=True,
                        type=int
                        )
    parser.add_argument('--splitfile',
                        required=False,
                        type=str,
                        default=DATASET_SPLIT_FILE
                        )
    parser.add_argument('--checkpoint',
                        required=False,
                        type=int,
                        default=-1
                        )
    parser.add_argument('--save_segdir_postfix',
                        required=False,
                        type=str,
                        default=""
                        )
    parser.add_argument('--stage',
                        required=False,
                        choices=['coarse', 'fine'],
                        default='coarse'
                        )

    args = parser.parse_args()

    stage = COARSE_STAGE
    if args.stage == 'fine':
        stage = FINE_STAGE

    run_inference(args.config, args.gpu, args.splitfile, args.checkpoint, args.save_segdir_postfix, stage)
