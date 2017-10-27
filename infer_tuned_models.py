import subprocess
import argparse
import os
import sys
import time

CONFIG_BASE = "./config/tune_configs"
LOGFILE = "tune_models/log/inferencelog_gpu%d.txt"

INFERENCE_CMD = "python net_segment.py inference -c %s --border %s --save_seg_dir %s --inference_iter %d"
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

def run_inference(configs, gpu):

    log = (LOGFILE) % gpu
    if not os.path.exists(log):
        open(log, 'x')
    with open(log, "a") as logptr:
        with open(configs, "r") as fptr:
            for config in fptr:
                config = config.strip()
                if (gpu == 0 and "dice" in config) or (gpu == 1 and "gdsc" in config):
                    border = "\"(8, 8, 8)\"" if "48-8" in config else "\"(16, 16, 16)\""
                    checkpoints = range(12000, 50000, 2000)
                    checkpoints.append(49999)
                    full_config = os.path.join(CONFIG_BASE, config)
                    for checkpoint in checkpoints:
                        cmd = INFERENCE_CMD % (full_config, border, os.path.join(SAVE_DIR_BASE,str(checkpoint)), checkpoint)
                        logptr.write("execute " + cmd + " \n")
                        logptr.flush()
                        start = time.time()
                        execute(cmd)
                        logptr.write("execution time %0.3f \n" % ((time.time()-start)*1000.0))

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
    run_inference(args.configs, args.gpu)
