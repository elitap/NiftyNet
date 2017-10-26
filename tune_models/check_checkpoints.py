import argparse
import os
import time

CHECKPOINT_FILE = "models/model.ckpt-%d.index"

def checkpoint_checker(path):

    for name in os.listdir(path):
        full_name = os.path.join(path,name)
        if ("dice" in name or "gdsc" in name) and os.path.isdir(full_name):
            checkpoints = range(12000, 50000, 2000)
            checkpoints.append(49999)
            for checkpoint in checkpoints:
                checkpoint_path = os.path.join(full_name, CHECKPOINT_FILE) % checkpoint
                if not os.path.exists(checkpoint_path):
                    print "checkpoint not found: ", checkpoint_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--modeldir',
                        required=False,
                        default="."
                        )

    args = parser.parse_args()
    checkpoint_checker(args.modeldir)
