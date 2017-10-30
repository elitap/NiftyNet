import argparse
import os
import time

CHECKPOINT_FILE = "models/model.ckpt-%d.index"
OUTPUT_DIR = "output/%d"

def checker(path, check_filter):
    for name in os.listdir(path):
        full_name = os.path.join(path,name)
        if ("dice" in name or "gdsc" in name) and os.path.isdir(full_name):
            checkpoints = range(12000, 50000, 2000)
            checkpoints.append(49999)
            for checkpoint in checkpoints:
                checkpoint_path = os.path.join(full_name, check_filter) % checkpoint
                if not os.path.exists(checkpoint_path):
                    print "Not found: ", checkpoint_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--modeldir',
                        required=False,
                        default="."
                        )

    args = parser.parse_args()
    checker(args.modeldir, OUTPUT_DIR)
