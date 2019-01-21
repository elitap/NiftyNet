import argparse
import os
import shutil
import subprocess
import sys
import SimpleITK as sitk
import numpy as np
<<<<<<< HEAD
import uuid
=======
>>>>>>> my_v3

from scripts import calc_foreground_label_otsu
from scripts import create_maskfrom_labels
from scripts import postprocess



COARSE_DIR = "coarse"
FINE_DIR = "fine"
NN_SPACING = np.array([1.1, 1.1, 3.0])

NN_INPUT_FILE = "_volume"
NN_INPUT_FOREGROUND = "_foreground"

HALF_SIZE_KEYWORD = "half"
QUARTER_SIZE_KEYWORD = "quarter"
OUTPUT_KEYWORD = "_niftynet_out"


INFERENCE_CMD = "python net_segment.py inference -c %s --save_seg_dir %s" \
                " --cuda_devices %d --inference_iter %d --validation_every_n 0"

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


def resample(spacing, size, origin, image, volume=False):

    resampler = sitk.ResampleImageFilter()
    interpolation = sitk.sitkNearestNeighbor
    if volume:
        interpolation = sitk.sitkBSpline

    resampler.SetInterpolator(interpolation)
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(size)
    resampler.SetOutputOrigin(image.GetOrigin())

    resampledImg = resampler.Execute(image)
    resampledImg.SetOrigin(origin)

    return resampledImg


def get_size_from_spacing(old_spacing, new_spacing, old_size):
    scale = old_spacing / new_spacing
    new_size = old_size * scale
    return new_size.round().astype(int).tolist()


def split_filename(image_file):
    split_file = os.path.splitext(image_file)
    if split_file[1] == ".gz":
        split_file = list(os.path.splitext(split_file[0]))
        split_file[1] += ".gz"
    return split_file


def rename_image_file(image_file, postfix):
    split_file = split_filename(image_file)
    return split_file[0]+postfix+split_file[1]


def create_work_dirs(input_dir):
    working_dir = os.path.join(input_dir, COARSE_DIR)
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.mkdir(working_dir)

    working_dir = os.path.join(input_dir, FINE_DIR)
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.mkdir(working_dir)


def preprocess(input_dir, filename_contains, coarse_config):
    create_work_dirs(input_dir)
    fine_data_dir = os.path.join(input_dir, FINE_DIR)
    coarse_data_dir = os.path.join(input_dir, COARSE_DIR)

    img_size_map_orig = dict()
    img_size_map_nnspace = dict()
    for input_file in os.listdir(input_dir):
        if filename_contains in input_file:
            image_path = os.path.join(input_dir, input_file)
            if os.path.isdir(image_path): continue
            itk_image = sitk.ReadImage(image_path)

            #replace reserved keywords
            input_file = input_file.replace(NN_INPUT_FILE, '').replace(NN_INPUT_FOREGROUND, '')

            #store orig spacing and size
            img_size_map_orig[split_filename(input_file)[0]] = np.concatenate(
                (itk_image.GetSpacing(), itk_image.GetSize(), itk_image.GetOrigin()))

            print "resample input"
            #resample to NN spacing
            resampled_image = resample(NN_SPACING,
                                       get_size_from_spacing(itk_image.GetSpacing(), NN_SPACING, itk_image.GetSize()),
                                       (0.0, 0.0, 0.0),
                                       itk_image,
                                       True)

            #save image for the fine stage
            input_file2save = rename_image_file(input_file, NN_INPUT_FILE)
            sitk.WriteImage(resampled_image, os.path.join(fine_data_dir, input_file2save))

            # store dataset spacing and size
            img_size_map_nnspace[split_filename(input_file)[0]] = np.concatenate(
                (resampled_image.GetSpacing(), resampled_image.GetSize()))

            #reduce size for the coarse stage
            if HALF_SIZE_KEYWORD in coarse_config:
                new_spacing = np.array([NN_SPACING[0]*2, NN_SPACING[1]*2, NN_SPACING[2]])
                resampled_image = resample(new_spacing,
                                           get_size_from_spacing(itk_image.GetSpacing(), new_spacing, itk_image.GetSize()),
                                           (0.0, 0.0, 0.0),
                                           itk_image,
                                           True)
            if QUARTER_SIZE_KEYWORD in coarse_config:
                new_spacing = np.array([NN_SPACING[0] * 4, NN_SPACING[1] * 4, NN_SPACING[2]])
                resampled_image = resample(new_spacing,
                                           get_size_from_spacing(itk_image.GetSpacing(), new_spacing, itk_image.GetSize()),
                                           (0.0, 0.0, 0.0),
                                           itk_image,
                                           True)

            sitk.WriteImage(resampled_image, os.path.join(coarse_data_dir, input_file2save))

            #get coarse stage foreground
            print "foreground extraction: "
            foreground_image = calc_foreground_label_otsu.getForeground(resampled_image)
            foreground_file2save = rename_image_file(input_file, NN_INPUT_FOREGROUND)
            sitk.WriteImage(foreground_image, os.path.join(coarse_data_dir, foreground_file2save))

    return img_size_map_orig, img_size_map_nnspace, coarse_data_dir, fine_data_dir


def replace_path_to_search(config_file, new_dir, data_path):
    new_dir = os.path.join(new_dir, "config")
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    new_config_file = os.path.join(new_dir, os.path.split(config_file)[1])
    with open(config_file, 'r') as cfile_ptr:
        with open(new_config_file, 'w') as new_cfile_ptr:
            for line in cfile_ptr:
                if line.startswith("path_to_search"):
                    new_cfile_ptr.write("path_to_search = "+data_path+"\n")
                    continue

                new_cfile_ptr.write(line)

    return new_config_file


def exec_inference(config, output_dir, gpu, checkpoint):
    cmd = INFERENCE_CMD % (config, output_dir, gpu, checkpoint)
    print cmd
    execute(cmd)


def resample_coarse_output(data_dir, output_dir, orig_size_map, nn_size_map, dilation):
    print "Resample coarse output"

    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        if os.path.isdir(file_path) or OUTPUT_KEYWORD not in file: continue

        itk_resultimage = sitk.ReadImage(file_path)

        splitted_filename = split_filename(file)
        mapkey = splitted_filename[0].replace(OUTPUT_KEYWORD, "")
        itk_resultimage_orig_size = resample(orig_size_map[mapkey][:3].tolist(),
                                             orig_size_map[mapkey][3:6].astype(int).tolist(),
                                             orig_size_map[mapkey][6:].astype(int).tolist(),
                                             itk_resultimage)

        coarse_file2save = rename_image_file(mapkey + splitted_filename[1], "_coarse")
        sitk.WriteImage(itk_resultimage_orig_size, os.path.join(output_dir, coarse_file2save))

        # resample to NN spacing
        labelmap_fine_stage = resample(nn_size_map[mapkey][:3].tolist(),
                                       nn_size_map[mapkey][3:].astype(int).tolist(),
                                       (0.0, 0.0, 0.0),
                                       itk_resultimage)

        foreground_fine_stage = create_maskfrom_labels.combineLabels(labelmap_fine_stage, dilation)
        foreground_file2save = rename_image_file(mapkey + splitted_filename[1], NN_INPUT_FOREGROUND)
        sitk.WriteImage(foreground_fine_stage, os.path.join(data_dir, foreground_file2save))


def postprocess_using_coarse_out(output_dir, itk_fine_resultimage, image_id, dilation):
    print "Postprocess"

    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isdir(file_path) or not (image_id in file and "_coarse" in file): continue

        itk_coarse_resultimage = sitk.ReadImage(file_path)
        itk_filtered_resultimage = postprocess.filter_using_coarse_out(itk_coarse_resultimage,
                                                                       itk_fine_resultimage, dilation)

        filtered_file2save = rename_image_file(file.replace("_coarse", ''), "_fine_postproc")
        sitk.WriteImage(itk_filtered_resultimage, os.path.join(output_dir, filtered_file2save))


def resample_fine_output(output_dir, orig_size_map, postprocess, dilation):
    print "Resample fine output"

    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isdir(file_path) or OUTPUT_KEYWORD not in file: continue

        itk_resultimage = sitk.ReadImage(file_path)

        splitted_filename = split_filename(file)
        mapkey = splitted_filename[0].replace(OUTPUT_KEYWORD, "")
        itk_resultimage_orig_size = resample(orig_size_map[mapkey][:3].tolist(),
                                             orig_size_map[mapkey][3:6].astype(int).tolist(),
                                             orig_size_map[mapkey][6:].astype(int).tolist(),
                                             itk_resultimage)

        fine_file2save = rename_image_file(mapkey + splitted_filename[1], "_fine")
        sitk.WriteImage(itk_resultimage_orig_size, os.path.join(output_dir, fine_file2save))

        os.remove(file_path)

        if postprocess:
            postprocess_using_coarse_out(output_dir, itk_resultimage_orig_size, mapkey, dilation)


def run_pipeline(coarse_config, fine_config, input_dir, output_dir, filename_contains, gpu, coarse_cp, fine_cp,
                 dilation, postprocess):
    orig_size_map, nn_size_map, coarse_data_dir, fine_data_dir = \
        preprocess(input_dir, filename_contains, os.path.split(coarse_config)[1])

    #replace path to search in config file
    coarse_config = replace_path_to_search(coarse_config, coarse_data_dir, coarse_data_dir)
    fine_config = replace_path_to_search(fine_config, fine_data_dir, fine_data_dir)
    #coarse stage
    exec_inference(coarse_config, output_dir=fine_data_dir, gpu=gpu, checkpoint=coarse_cp)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    resample_coarse_output(fine_data_dir, output_dir, orig_size_map, nn_size_map, dilation)
    #fine stage
    exec_inference(fine_config, output_dir=output_dir, gpu=gpu, checkpoint=fine_cp)
    resample_fine_output(output_dir, orig_size_map, postprocess, dilation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--coarse_config',
                        required=True,
                        help="path to the coarse stage config"
                        )
    parser.add_argument('--fine_config',
                        required=True,
                        help="path to the fine stage config"
                        )
    parser.add_argument('--input_dir',
                        required=True,
                        help="path to the images to segment"
                        )
    parser.add_argument('--output_dir',
                        required=True,
                        help="path the segmentation of the images is stored to"
                        )
    parser.add_argument('--filename_contains',
                        required=False,
                        default="",
                        help="keyword filtering the input directory"
                        )
    parser.add_argument('--gpu',
                        required=False,
                        type=int,
                        default=0,
                        help="specifies the gpu to be used"
                        )
    parser.add_argument('--coarse_checkpoint',
                        required=False,
                        type=int,
                        default=0,
                        help="if specified the given checkpoint is used otherwise the last checkpoint is used"
                        )
    parser.add_argument('--fine_checkpoint',
                        required=False,
                        type=int,
                        default=0,
                        help="if specified the given checkpoint is used otherwise the last checkpoint is used"
                        )
    parser.add_argument('--dilation',
                        required=False,
                        type=int,
<<<<<<< HEAD
                        default=13,
                        help="dilation radius used to create the foreground mask of the fine stage"
                        )

    args = parser.parse_args()
    run_pipeline(args.coarse_config, args.fine_config, args.input_dir, args.output_dir, args.filename_contains, args.gpu,
                 args.coarse_checkpoint, args.fine_checkpoint, args.dilation, False)
=======
                        default=11,
                        help="dilation radius used to create the foreground mask of the fine stage"
                        )
    parser.add_argument('--postprocess',
                        action='store_true')

    args = parser.parse_args()
    run_pipeline(args.coarse_config, args.fine_config, args.input_dir, args.output_dir, args.filename_contains, args.gpu,
                 args.coarse_checkpoint, args.fine_checkpoint, args.dilation, args.postprocess)
>>>>>>> my_v3
