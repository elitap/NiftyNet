import subprocess
import argparse
import os
import sys

COARSE_STAGE = "coarse_stage"
FINE_STAGE = "fine_stage"

DATASET_SPLIT_FILE = "./data/HaN_MICCAI2015_Dataset/test_data_split.csv"
CONFIGS = "./config/%s_configs"
LOGFILE = "./%s/log/tune_log_gpu%d.txt"

TRAIN_CMD = "python net_segment.py train -c %s --cuda_devices %d --dataset_split_file %s"


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


def run_train(configs, gpu, stage):

    log = ((LOGFILE) % (stage, gpu))
    logsplit = os.path.split(log)
    if not os.path.exists(logsplit[0]):
        os.makedirs(logsplit[0])

    with open(log, 'a') as logptr:
        with open(configs, "r") as fptr:
            for config in fptr:
                config = config.strip()
                if len(config) == 0 or config[0] == '#':
                    continue

                staged_config = ((CONFIGS) % (stage))
                full_configfile = os.path.join(staged_config, config)
                cmd = TRAIN_CMD % (full_configfile, gpu, os.path.abspath(DATASET_SPLIT_FILE))

                logptr.write("execute: " + cmd + "\n")
                logptr.flush()
                execute(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--config',
                        required=True
                        )
    parser.add_argument('--gpu',
                        required=True,
                        type=int
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

    run_train(args.config, args.gpu, stage)

