import subprocess
import argparse
import os
import sys
import time

STAGE = ""

MODEL_BASE = "./tune_models"+STAGE+"/%s/models/model.ckpt-%d.index"
CONFIG_BASE = "./config/tune_configs"+STAGE
LOGFILE = "./tune_models"+STAGE+"/log/inferencelog_gpu%d.txt"

INFERENCE_CMD = "python net_segment.py inference -c %s --border %s --save_seg_dir %s --inference_iter %d --cuda_devices %d"
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
    with open(log, 'w') as logptr:
        with open(configs, "r") as fptr:
            for config in fptr:
                config = config.strip()
                if (gpu == 0 and "gdsc" in config) or (gpu == 1 and "dice" in config):
                    border = "\"(8, 8, 8)\"" if ("48-8" in config or "24-24" in config) else "\"(16, 16, 16)\""
                    checkpoints = range(50000, 250001, 2000)
                    #checkpoints = range(12000, 50000, 2000)
                    checkpoints.append(249999)
                    full_config = os.path.join(CONFIG_BASE, config)
                    for checkpoint in checkpoints:
                        checkpoint_path = MODEL_BASE % (os.path.splitext(config)[0], checkpoint)
                        if os.path.exists(checkpoint_path):
                            cmd = INFERENCE_CMD % (full_config, border, os.path.join(SAVE_DIR_BASE,str(checkpoint)), checkpoint, gpu)
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
