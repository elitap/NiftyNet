import subprocess
import argparse
import os

INFERENCE_NAME = "inference_niftynet_log"
TRAIN_NAME = "train_niftynet_log"

def collect_info(modeldir, configs, resultfile, log):

    filename = ""
    if log == 'Train':
        filename = TRAIN_NAME
    else:
        filename = INFERENCE_NAME

    resultfile += log + ".csv"

    with open(resultfile, "w") as resptr:

        resptr.write("model,size,runtime\n")

        with open(configs, "r") as fptr:
            for config in fptr:
                config = config.strip()
                if config[0] == "#":
                    continue

                full_filename = os.path.join(modeldir, config, filename)
                print full_filename

                line = subprocess.check_output(['tail', '-1', full_filename]).strip()
                runtime = line.split(' ')[-1][:-2]
                try:
                    float(runtime)
                except ValueError:
                    print "Not a float"
                    continue

                size = "full"
                if "half" in config:
                    size = "half"
                if "quarter" in config:
                    size = "quarter"

                resptr.write(config+','+size+','+runtime+"\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--modelBaseDir',
                        required=True,
                        help="base directory containing the models specified in the config file"
                        )
    parser.add_argument('--config',
                        required=True,
                        help="config file, listing all models the runtime should be collected for"
                        )
    parser.add_argument('--resultfile',
                        required=False,
                        default="../results/runtime_",
                        help="csv file the results are saved in"
                        )
    parser.add_argument('--log',
                        choices=['Train', 'Infer'],
                        default='Train'
                        )

    args = parser.parse_args()
    collect_info(args.dir, args.config, args.result, args.log)