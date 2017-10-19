import subprocess
import argparse
import os
import sys

ENDING = [".ini"]
CONFIGS = "/home/elias/Dokumente/head_neck_seg/NiftyNet/config/tune_configs"

DONE = ["full_e-3_96-1_dice_50k_1024s.ini", "quarter_e-5_48-8_gdsc_50k_1024s.ini"]

MYPC = ["full_e-3_96-1_dice_50k_1024s.ini",
        "full_e-3_96-1_gdsc_50k_1024s.ini",
        "full_e-4_96-1_dice_50k_1024s.ini",
        "full_e-4_96-1_gdsc_50k_1024s.ini",
        "full_e-5_96-1_dice_50k_1024s.ini",
        "full_e-5_96-1_gdsc_50k_1024s.ini",
        "half_e-3_48-8_dice_50k_1024s.ini",
        "half_e-3_48-8_gdsc_50k_1024s.ini",
        "half_e-3_96-1_dice_50k_1024s.ini",
        "half_e-3_96-1_gdsc_50k_1024s.ini",
        "half_e-4_48-8_dice_50k_1024s.ini",
        "half_e-4_48-8_gdsc_50k_1024s.ini"]

OTHERPC = ["half_e-4_96-1_dice_50k_1024s.ini",
            "half_e-4_96-1_gdsc_50k_1024s.ini",
            "half_e-5_48-8_dice_50k_1024s.ini",
            "half_e-5_48-8_gdsc_50k_1024s.ini",
            "half_e-5_96-1_dice_50k_1024s.ini",
            "half_e-5_96-1_gdsc_50k_1024s.ini",
            "quarter_e-3_48-8_dice_50k_1024s.ini",
            "quarter_e-3_48-8_gdsc_50k_1024s.ini",
            "quarter_e-4_48-8_dice_50k_1024s.ini",
            "quarter_e-4_48-8_gdsc_50k_1024s.ini",
            "quarter_e-5_48-8_dice_50k_1024s.ini",
            "quarter_e-5_48-8_gdsc_50k_1024s.ini"]



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


def run_tests(gpu, pc):
    with open("tune_log_gpu"+str(gpu)+".txt", "a") as fptr:

        for file in os.listdir(CONFIGS):
            if os.path.splitext(file)[1] in ENDING:
                fullfile = os.path.join(CONFIGS, file)

                cmd = "python net_segment.py train -c " + fullfile + " --cuda_devices " + str(gpu)
                if file not in DONE and (pc == 0 and file in MYPC) and (pc == 1 and file in OTHERPC):
                    if gpu == 0 and "dice" in file and file not in DONE:
                        exec_niftynet(cmd, fptr)
                    if gpu == 1 and "gdsc" in file:
                        exec_niftynet(cmd, fptr)

def exec_niftynet(cmd, fptr):
    fptr.write("execute: " + cmd + "\n")
    fptr.flush()
    if execute(cmd):
        fptr.write("succeded \n")
    else:
        fptr.write("failed \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--gpu',
                        required=True,
                        type=int
                        )
    parser.add_argument('--pc',
                        required=True,
                        type=int
                        )

    args = parser.parse_args()
    run_tests(args.gpu, args.pc)