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

        resptr.write("model,size,runtime,imgs\n")

        with open(configs, "r") as fptr:
            for config in fptr:
                config = config.strip()
                if config[0] == "#":
                    continue

                full_filename = os.path.join(modeldir, config, filename)
                print full_filename


                # line = subprocess.check_output(['tail', '-1', full_filename]).strip()
                # before iterating over
                img_cnt = 0
                runtime = 0
                with open(full_filename, "r") as infptr:
                    for line in infptr:
                        if "SegmentationApplication stopped" in line:
                            runtime_str = line.strip().split(' ')[-1][:-2]
                            try:
                                runtime += float(runtime_str)
                            except ValueError:
                                print "Not a float"
                                break

                        if "grid sampling window sizes: {" in line:
                            img_cnt += 1





                size = "full"
                if "half" in config:
                    size = "half"
                if "quarter" in config:
                    size = "quarter"

                resptr.write(config+','+size+','+str(runtime)+','+str(img_cnt)+"\n")

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
    collect_info(args.modelBaseDir, args.config, args.resultfile, args.log)