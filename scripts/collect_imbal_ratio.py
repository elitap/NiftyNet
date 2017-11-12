# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf

from niftynet.engine.sampler_weighted import WeightedSampler
from niftynet.engine.sampler_weighted import weighted_spatial_coordinates
from niftynet.io.image_reader import ImageReader
from tests.test_util import ParserNamespace


IMBALENCE_RATIO_MODL = "./tune_models/imbal_ratio_model"

DATA_PATH = "./data/combined_challenge/Train_4er"
SPACING = (4.4, 4.4, 3.0)
WINDOW_SIZE = (48, 48, 48)
AXCODES = ('A', 'R', 'S')
SAMPLE_SIZE = 1024


SEG_DATA = {
    'segmentation': ParserNamespace(
        csv_file=os.path.join(IMBALENCE_RATIO_MODL, 'segmentation.csv'),
        path_to_search=DATA_PATH,
        filename_contains=('segmentation',),
        filename_not_contains=(),
        interp_order=0,
        pixdim=SPACING,
        axcodes=AXCODES,
        spatial_window_size=WINDOW_SIZE
    ),
    'sampler': ParserNamespace(
        csv_file=os.path.join(IMBALENCE_RATIO_MODL, 'sampler.csv'),
        path_to_search=DATA_PATH,
        filename_contains=('foreground',),
        filename_not_contains=(),
        interp_order=0,
        pixdim=SPACING,
        axcodes=AXCODES,
        spatial_window_size=WINDOW_SIZE
    )
}
#TODO here find out what is needed
SEG_TASK = ParserNamespace(sampler=('sampler',),
                           label=('segmentation',))


def get_3d_reader():
    reader = ImageReader(['label', 'sampler'])
    reader.initialise_reader(SEG_DATA, SEG_TASK)
    return reader


def bowoasnuit():
    #TODO find out about what the samplter iterates: bs, queue, thresds

    #maybe reader has the mapping from subject id to image
    sampler = WeightedSampler(reader=get_3d_reader(),
                              data_param=SEG_DATA,
                              batch_size=SAMPLE_SIZE,
                              windows_per_image=SAMPLE_SIZE,
                              queue_length=1)

    with tf.Session() as sess:
        coordinator = tf.train.Coordinator()
        sampler.run_threads(sess, coordinator, num_threads=1)
        #TODO get link between subject id and image
        #TODO calc imbalence ratio here!!
        out = sess.run(sampler.pop_batch_op())

    sampler.close_all()



class RandomCoordinatesTest(tf.test.TestCase):
    def test_coodinates(self):
        coords = weighted_spatial_coordinates(
            subject_id=1,
            data={'sampler': np.random.rand(41, 42, 42, 1, 1)},
            img_sizes={'image': (42, 42, 42, 1, 2),
                       'label': (42, 42, 42, 1, 1)},
            win_sizes={'image': (23, 23, 40),
                       'label': (40, 32, 33)},
            n_samples=10)
        self.assertEquals(np.all(coords['image'][:0] == 1), True)
        self.assertEquals(coords['image'].shape, (10, 7))
        self.assertEquals(coords['label'].shape, (10, 7))
        self.assertAllClose(
            (coords['image'][:, 4] + coords['image'][:, 1]),
            (coords['label'][:, 4] + coords['label'][:, 1]), atol=1.0)
        self.assertAllClose(
            (coords['image'][:, 5] + coords['image'][:, 2]),
            (coords['label'][:, 5] + coords['label'][:, 2]), atol=1.0)
        self.assertAllClose(
            (coords['image'][:, 6] + coords['image'][:, 3]),
            (coords['label'][:, 6] + coords['label'][:, 3]), atol=1.0)

    def test_25D_coodinates(self):
        coords = weighted_spatial_coordinates(
            subject_id=1,
            data={'sampler': np.random.rand(42, 42, 42, 1, 1)},
            img_sizes={'image': (42, 42, 42, 1, 1),
                       'label': (42, 42, 42, 1, 1)},
            win_sizes={'image': (23, 23, 1),
                       'label': (40, 32, 1)},
            n_samples=10)
        self.assertEquals(np.all(coords['image'][:0] == 1), True)
        self.assertEquals(coords['image'].shape, (10, 7))
        self.assertEquals(coords['label'].shape, (10, 7))
        self.assertAllClose(
            (coords['image'][:, 4] + coords['image'][:, 1]),
            (coords['label'][:, 4] + coords['label'][:, 1]), atol=1.0)
        self.assertAllClose(
            (coords['image'][:, 5] + coords['image'][:, 2]),
            (coords['label'][:, 5] + coords['label'][:, 2]), atol=1.0)
        self.assertAllClose(
            (coords['image'][:, 6] + coords['image'][:, 3]),
            (coords['label'][:, 6] + coords['label'][:, 3]), atol=1.0)

    def test_2D_coodinates(self):
        coords = weighted_spatial_coordinates(
            subject_id=1,
            data={'sampler': np.random.rand(42, 42, 42, 1, 1)},
            img_sizes={'image': (42, 42, 1, 1, 1),
                       'label': (42, 42, 1, 1, 1)},
            win_sizes={'image': (23, 23, 1),
                       'label': (40, 32, 1)},
            n_samples=10)
        self.assertEquals(np.all(coords['image'][:0] == 1), True)
        self.assertEquals(coords['image'].shape, (10, 7))
        self.assertEquals(coords['label'].shape, (10, 7))
        self.assertAllClose(
            (coords['image'][:, 4] + coords['image'][:, 1]),
            (coords['label'][:, 4] + coords['label'][:, 1]), atol=1.0)
        self.assertAllClose(
            (coords['image'][:, 5] + coords['image'][:, 2]),
            (coords['label'][:, 5] + coords['label'][:, 2]), atol=1.0)
        self.assertAllClose(
            (coords['image'][:, 6] + coords['image'][:, 3]),
            (coords['label'][:, 6] + coords['label'][:, 3]), atol=1.0)

    def test_ill_coodinates(self):
        with self.assertRaisesRegexp(IndexError, ""):
            coords = weighted_spatial_coordinates(
                subject_id=1,
                data={'sampler': np.random.rand(42, 42, 42)},
                img_sizes={'image': (42, 42, 1, 1, 1),
                           'label': (42, 42, 1, 1, 1)},
                win_sizes={'image': (23, 23),
                           'label': (40, 32)},
                n_samples=10)

        with self.assertRaisesRegexp(TypeError, ""):
            coords = weighted_spatial_coordinates(
                subject_id=1,
                data={'sampler': np.random.rand(42, 42, 42, 1, 1)},
                img_sizes={'image': (42, 42, 1, 1, 1),
                           'label': (42, 42, 1, 1, 1)},
                win_sizes={'image': (23, 23, 1),
                           'label': (40, 32, 1)},
                n_samples='test')

        with self.assertRaisesRegexp(AssertionError, ""):
            coords = weighted_spatial_coordinates(
                subject_id=1,
                data={'sampler': np.random.rand(42, 42, 42, 1, 1)},
                img_sizes={'label': (42, 1, 1, 1)},
                win_sizes={'image': (23, 23, 1)},
                n_samples=0)

        with self.assertRaisesRegexp(RuntimeError, ""):
            coords = weighted_spatial_coordinates(
                subject_id=1,
                data={},
                img_sizes={'image': (42, 42, 1, 1, 1),
                           'label': (42, 42, 1, 1, 1)},
                win_sizes={'image': (23, 23, 1),
                           'label': (40, 32, 1)},
                n_samples=10)


if __name__ == "__main__":
    bowoasnuit()
