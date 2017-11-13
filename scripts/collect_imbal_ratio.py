# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import argparse

import numpy as np
import pandas
import tensorflow as tf
from numpy.core.operand_flag_tests import inplace_add

from niftynet.engine.sampler_weighted import WeightedSampler
from niftynet.engine.sampler_weighted import weighted_spatial_coordinates
from niftynet.io.image_reader import ImageReader
from tests.test_util import ParserNamespace
from defs import LABELS


IMBALENCE_RATIO_MODL = "./tune_models/imbal_ratio_model"

DATA_PATH = "./data/combined_challenge/%s%s"
PATH_POSTFIX = {"full": "", "half": "_2er", "quarter": "_4er"}
SPACING = {"full": (1.1, 1.1, 3.0), "half": (2.2, 2.2, 3.0), "quarter": (4.4, 4.4, 3.0)}
WINDOW_SIZE = {"full": [(96, 96, 72)], "half": [(96, 96, 72), (48, 48, 48)], "quarter": [(48, 48, 48)]}
AXCODES = ('A', 'R', 'S')
SAMPLE_SIZE = 1024

HEADER = "Dataset,Datasize,Windowsize,File,Sample," + ','.join([organ for organ in LABELS.keys()])
DF_ROW = {name: '' for name in HEADER.split(',')}


def get_3d_reader(data_path, spacing, window_size):

    SEG_DATA = {
        'segmentation': ParserNamespace(
            csv_file=os.path.join(IMBALENCE_RATIO_MODL, 'segmentation.csv'),
            path_to_search=data_path,
            filename_contains=('segmentation',),
            filename_not_contains=(),
            interp_order=0,
            pixdim=spacing,
            axcodes=AXCODES,
            spatial_window_size=window_size
        ),
        'sampler': ParserNamespace(
            csv_file=os.path.join(IMBALENCE_RATIO_MODL, 'sampler.csv'),
            path_to_search=data_path,
            filename_contains=('foreground',),
            filename_not_contains=(),
            interp_order=0,
            pixdim=spacing,
            axcodes=AXCODES,
            spatial_window_size=window_size
        )
    }

    SEG_TASK = ParserNamespace(sampler=('sampler',),
                               label=('segmentation',))

    reader = ImageReader(['label', 'sampler'])
    reader.initialise_reader(SEG_DATA, SEG_TASK)
    return reader


def get_imbal_ratio(result_file, data_path, spacing, window_sizes):

    if not os.path.exists(result_file):
        with open(result_file, 'w') as fileptr:
            fileptr.write(HEADER + '\n')

    df = pandas.read_csv(result_file)

    for window_size in window_sizes:
        reader = get_3d_reader(data_path, spacing, window_size)

        DF_ROW['Windowsize'] = window_size[0]

        while True:
            image_id, data, _ = reader(idx=None, shuffle=False)
            if not data:
                break

            DF_ROW['File'] = os.path.split(reader._file_list['segmentation'][image_id])[1][:9]

            image_shapes = {name: data[name].shape for name in data.keys()}
            static_window_shapes = {name: window_size for name in data.keys()}

            # find random coordinates based on window and image shapes
            coordinates = weighted_spatial_coordinates(
                image_id,
                data,
                image_shapes,
                static_window_shapes,
                SAMPLE_SIZE)

            label_coords = coordinates['label']
            ratio_sum = 0
            for window_id in range(0, SAMPLE_SIZE):
                x_start, y_start, z_start, x_end, y_end, z_end = label_coords[window_id, 1:]
                image_window = data['label'][x_start:x_end, y_start:y_end, z_start:z_end, ...]
                unique, cnt = np.unique(image_window.flatten(), return_counts=True)

                DF_ROW['Sample'] = window_id
                background_idx = np.where(unique == 0)
                background_cnt = cnt[background_idx][0] if np.size(background_idx) > 0 else 0.0
                for organ, id in LABELS.iteritems():
                    organ_idx = np.where(unique == id)
                    organ_cnt = cnt[organ_idx][0] if np.size(organ_idx) > 0 else 0.0
                    ratio = organ_cnt/background_cnt if background_cnt != 0.0 else 1.0
                    DF_ROW[organ] = ratio
                    ratio_sum += ratio

                df = df.append(DF_ROW, ignore_index=True)
            print(DF_ROW['File'], window_size, spacing, "average window ratio", ratio_sum/SAMPLE_SIZE)


    df.to_csv(result_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--resfile',
                        required=True
                        )
    parser.add_argument('--size',
                        choices=['full', 'half', 'quarter'],
                        default='full'
                        )
    parser.add_argument('--dataset',
                        choices=['Train', 'Test', 'Onsite'],
                        default='Train'
                        )

    args = parser.parse_args()
    data_path = DATA_PATH % (args.dataset, PATH_POSTFIX[args.size])

    spacing = SPACING[args.size]
    window_sizes = WINDOW_SIZE[args.size]

    DF_ROW['Dataset'] = args.dataset
    DF_ROW['Datasize'] = args.size

    split_res_file = os.path.splitext(args.resfile)
    resultfile = split_res_file[0] + "_" + args.size + ".csv"

    get_imbal_ratio(resultfile, data_path, spacing, window_sizes)
