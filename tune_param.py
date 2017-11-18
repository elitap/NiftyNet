import subprocess
import argparse
import os
import sys
import numpy as np

STAGE = ""
CONFIGS = "./config/tune_configs"
LOGFILE = "./tune_models"+STAGE+"/log/tune_log_gpu%d.txt"

TRAIN_CMD = "python net_segment.py train -c %s --cuda_devices %d"


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


def run_train(configs, gpu, filter_gpu0=[], filter_gpu1=[]):

    log = (LOGFILE) % gpu
    with open(log, 'a') as logptr:
        with open(configs, "r") as fptr:
            for config in fptr:
                config = config.strip()
                if len(config) == 0 or config[0] == '#':
                    continue
                if (gpu == 0 and np.all([myfilter in config for myfilter in filter_gpu0])) or (gpu == 1 and np.all([myfilter in config for myfilter in filter_gpu1])):

                    full_configfile = os.path.join(CONFIGS, config)
                    cmd = TRAIN_CMD % (full_configfile, gpu)

                    logptr.write("execute: " + cmd + "\n")
                    logptr.flush()
                    execute(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--configs',
                        required=True
                        )
    parser.add_argument('--gpu',
                        required=True,
                        type=int
                        )

    args = parser.parse_args()
    run_train(args.configs, args.gpu, filter_gpu0=[], filter_gpu1=[])

