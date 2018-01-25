# -*- coding: utf-8 -*-
"""
This module manages a table of subject ids and
their associated image file names.
A subset of the table can be retrieved by partitioning the set of images into
subsets of ``Train``, ``Validation``, ``Inference``.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random

import pandas
import tensorflow as tf  # to use the system level logging

from niftynet.utilities.decorators import singleton
from niftynet.utilities.filename_matching import KeywordsMatching
from niftynet.utilities.niftynet_global_config import NiftyNetGlobalConfig
from niftynet.utilities.util_common import look_up_operations
from niftynet.utilities.util_csv import match_and_write_filenames_to_csv
from niftynet.utilities.util_csv import write_csv

COLUMN_UNIQ_ID = 'subject_id'
COLUMN_PHASE = 'phase'
TRAIN = 'Training'
VALID = 'Validation'
INFER = 'Inference'
ALL = 'All'
SUPPORTED_PHASES = set([TRAIN, VALID, INFER, ALL])


@singleton
class ImageSetsPartitioner(object):
    """
    This class maintains a pandas.dataframe of filenames for all input sections.
    The list of filenames are obtained by searching the specified folders
    or loading from an existing csv file.

    Users can query a subset of the dataframe by train/valid/infer partition
    label and input section names.
    """

    # dataframe (table) of file names in a shape of subject x modality
    _file_list = None
    # dataframes of subject_id:phase_id
    _partition_ids = None

    data_param = None
    ratios = None
    new_partition = False

    # for saving the splitting index
    data_split_file = ""
    # default parent folder location for searching the image files
    default_image_file_location = \
        NiftyNetGlobalConfig().get_niftynet_home_folder()

    def initialise(self,
                   data_param,
                   new_partition=False,
                   data_split_file=None,
                   ratios=None):
        """
        Set the data partitioner parameters

        :param data_param: corresponding to all config sections
        :param new_partition: bool value indicating whether to generate new
            partition ids and overwrite csv file
            (this class will write partition file iff new_partition)
        :param data_split_file: location of the partition id file
        :param ratios: a tuple/list with two elements:
            ``(fraction of the validation set, fraction of the inference set)``
            initialise to None will disable data partitioning
            and get_file_list always returns all subjects.
        """
        self.data_param = data_param
        if data_split_file is None:
            self.data_split_file = os.path.join('.', 'dataset_split.csv')
        else:
            self.data_split_file = data_split_file
        self.ratios = ratios

        self._file_list = None
        self._partition_ids = None

        self.load_data_sections_by_subject()
        self.new_partition = new_partition
        self.randomly_split_dataset(overwrite=new_partition)
        tf.logging.info(self)
        return self

    def number_of_subjects(self, phase=ALL):
        """
        query number of images according to phase.

        :param phase:
        :return:
        """
        if self._file_list is None:
            return 0
        phase = look_up_operations(phase, SUPPORTED_PHASES)
        if phase == ALL:
            return self._file_list[COLUMN_UNIQ_ID].count()
        if self._partition_ids is None:
            return 0
        selector = self._partition_ids[COLUMN_PHASE] == phase
        return self._partition_ids[selector].count()[COLUMN_UNIQ_ID]

    def get_file_list(self, phase=ALL, *section_names):
        """
        get file names as a dataframe, by partitioning phase and section names
        set phase to ALL to load all subsets.

        :param phase: the label of the subset generated by self._partition_ids
                    should be one of the SUPPORTED_PHASES
        :param section_names: one or multiple input section names
        :return: a pandas.dataframe of file names
        """
        if self._file_list is None:
            tf.logging.warning('Empty file list, please initialise'
                               'ImageSetsPartitioner first.')
            return []
        try:
            look_up_operations(phase, SUPPORTED_PHASES)
        except ValueError:
            tf.logging.fatal('Unknown phase argument.')
            raise
        for name in section_names:
            try:
                look_up_operations(name, set(self._file_list))
            except ValueError:
                tf.logging.fatal(
                    'Requesting files under input section [%s],\n'
                    'however the section does not exist in the config.', name)
                raise
        if phase == ALL:
            self._file_list = self._file_list.sort_index()
            if section_names:
                section_names = [COLUMN_UNIQ_ID] + list(section_names)
                return self._file_list[section_names]
            return self._file_list
        if self._partition_ids is None or self._partition_ids.empty:
            tf.logging.fatal('No partition ids available.')
            if self.new_partition:
                tf.logging.fatal('Unable to create new partitions,'
                                 'splitting ratios: %s, writing file %s',
                                 self.ratios, self.data_split_file)
            elif os.path.isfile(self.data_split_file):
                tf.logging.fatal(
                    'Unable to load %s, initialise the'
                    'ImageSetsPartitioner with `new_partition=True`'
                    'to overwrite the file.',
                    self.data_split_file)
            raise ValueError

        selector = self._partition_ids[COLUMN_PHASE] == phase
        selected = self._partition_ids[selector][[COLUMN_UNIQ_ID]]
        if selected.empty:
            tf.logging.warning(
                'Empty subset for phase [%s], returning None as file list. '
                'Please adjust splitting fractions.', phase)
            return None
        subset = pandas.merge(self._file_list, selected, on=COLUMN_UNIQ_ID)
        if subset.empty:
            tf.logging.warning(
                'No subject id matched in between file names and '
                'partition files.\nPlease check the partition files %s,\nor '
                'removing it to generate a new file automatically.',
                self.data_split_file)
        if section_names:
            section_names = [COLUMN_UNIQ_ID] + list(section_names)
            return subset[list(section_names)]
        return subset

    def load_data_sections_by_subject(self):
        """
        Go through all input data sections, converting each section
        to a list of file names.

        These lists are merged on ``COLUMN_UNIQ_ID``.

        This function sets ``self._file_list``.
        """
        if not self.data_param:
            tf.logging.fatal(
                'Nothing to load, please check input sections in the config.')
            raise ValueError
        self._file_list = None
        for section_name in self.data_param:
            modality_file_list = self.grep_files_by_data_section(section_name)
            if self._file_list is None:
                # adding all rows of the first modality
                self._file_list = modality_file_list
                continue
            n_rows = self._file_list[COLUMN_UNIQ_ID].count()
            self._file_list = pandas.merge(self._file_list,
                                           modality_file_list,
                                           how='outer',
                                           on=COLUMN_UNIQ_ID)
            if self._file_list[COLUMN_UNIQ_ID].count() < n_rows:
                tf.logging.warning('rows not matched in section [%s]',
                                   section_name)

        if self._file_list is None or self._file_list.size == 0:
            tf.logging.fatal(
                "Empty filename lists, please check the csv "
                "files (removing csv_file keyword if it is in the config file "
                "to automatically search folders and generate new csv "
                "files again).\n\n"
                "Please note in the matched file names, each subject id are "
                "created by removing all keywords listed `filename_contains` "
                "in the config.\n"
                "E.g., `filename_contains=foo, bar` will match file "
                "foo_subject42_bar.nii.gz, and the subject id is "
                "_subject42_.\n\n")
            raise IOError

    def grep_files_by_data_section(self, modality_name):
        """
        list all files by a given input data section::
            if the ``csv_file`` property of the section corresponds to a file,
                read the list from the file;
            otherwise
                write the list to ``csv_file``.

        :return: a table with two columns,
                 the column names are ``(COLUMN_UNIQ_ID, modality_name)``.
        """
        if modality_name not in self.data_param:
            tf.logging.fatal('unknown section name [%s], '
                             'current input section names: %s.',
                             modality_name, list(self.data_param))
            raise ValueError

        # input data section must have a ``csv_file`` section for loading
        # or writing filename lists
        try:
            csv_file = self.data_param[modality_name].csv_file
#            if not os.path.isfile(csv_file):
                # writing to the same folder as data_split_file
#                csv_file = os.path.join(os.path.dirname(self.data_split_file),
#                                        '{}.csv'.format(modality_name))

        except (AttributeError, TypeError):
            tf.logging.fatal('Missing `csv_file` field in the config file, '
                             'unknown configuration format.')
            raise

        if hasattr(self.data_param[modality_name], 'path_to_search') and \
                self.data_param[modality_name].path_to_search:
            tf.logging.info('[%s] search file folders, writing csv file %s',
                            modality_name, csv_file)
            section_properties = self.data_param[modality_name].__dict__.items()
            # grep files by section properties and write csv
            try:
                matcher = KeywordsMatching.from_tuple(
                    section_properties,
                    self.default_image_file_location)
                match_and_write_filenames_to_csv([matcher], csv_file)
            except (IOError, ValueError) as reading_error:
                tf.logging.warning('Ignoring input section: [%s], '
                                   'due to the following error:',
                                   modality_name)
                tf.logging.warning(repr(reading_error))
                return pandas.DataFrame(
                    columns=[COLUMN_UNIQ_ID, modality_name])
        else:
            tf.logging.info(
                '[%s] using existing csv file %s, skipped filenames search',
                modality_name, csv_file)

        if not os.path.isfile(csv_file):
            tf.logging.fatal(
                '[%s] csv file %s not found.', modality_name, csv_file)
            raise IOError
        try:
            csv_list = pandas.read_csv(
                csv_file,
                header=None,
                dtype=(str, str),
                names=[COLUMN_UNIQ_ID, modality_name],
                skipinitialspace=True)
        except Exception as csv_error:
            tf.logging.fatal(repr(csv_error))
            raise
        return csv_list

    # pylint: disable=broad-except
    def randomly_split_dataset(self, overwrite=False):
        """
        Label each subject as one of the ``TRAIN``, ``VALID``, ``INFER``,
        use ``self.ratios`` to compute the size of each set.

        The results will be written to ``self.data_split_file`` if overwrite
        otherwise it tries to read partition labels from it.

        This function sets ``self._partition_ids``.
        """
        if overwrite:
            try:
                valid_fraction, infer_fraction = self.ratios
                valid_fraction = max(min(1.0, float(valid_fraction)), 0.0)
                infer_fraction = max(min(1.0, float(infer_fraction)), 0.0)
            except (TypeError, ValueError):
                tf.logging.fatal(
                    'Unknown format of faction values %s', self.ratios)
                raise

            if (valid_fraction + infer_fraction) <= 0:
                tf.logging.warning(
                    'To split dataset into training/validation, '
                    'please make sure '
                    '"exclude_fraction_for_validation" parameter is set to '
                    'a float in between 0 and 1. Current value: %s.',
                    valid_fraction)
                # raise ValueError

            n_total = self.number_of_subjects()
            n_valid = int(math.ceil(n_total * valid_fraction))
            n_infer = int(math.ceil(n_total * infer_fraction))
            n_train = int(n_total - n_infer - n_valid)
            phases = [TRAIN] * n_train + \
                     [VALID] * n_valid + \
                     [INFER] * n_infer
            if len(phases) > n_total:
                phases = phases[:n_total]
            random.shuffle(phases)
            write_csv(self.data_split_file,
                      zip(self._file_list[COLUMN_UNIQ_ID], phases))
        elif os.path.isfile(self.data_split_file):
            tf.logging.warning(
                'Loading from existing partitioning file %s, '
                'ignoring partitioning ratios.', self.data_split_file)

        if os.path.isfile(self.data_split_file):
            try:
                self._partition_ids = pandas.read_csv(
                    self.data_split_file,
                    header=None,
                    dtype=(str, str),
                    names=[COLUMN_UNIQ_ID, COLUMN_PHASE],
                    skipinitialspace=True)
                assert not self._partition_ids.empty, \
                    "partition file is empty."
            except Exception as csv_error:
                tf.logging.warning(
                    "Unable to load the existing partition file %s, %s",
                    self.data_split_file, repr(csv_error))
                self._partition_ids = None

            try:
                is_valid_phase = \
                    self._partition_ids[COLUMN_PHASE].isin(SUPPORTED_PHASES)
                assert is_valid_phase.all(), \
                    "Partition file contains unknown phase id."
            except (TypeError, AssertionError):
                tf.logging.warning(
                    'Please make sure the values of the second column '
                    'of data splitting file %s, in the set of phases: %s.\n'
                    'Remove %s to generate random data partition file.',
                    self.data_split_file,
                    SUPPORTED_PHASES,
                    self.data_split_file)
                raise ValueError

    def __str__(self):
        return self.to_string()

    def to_string(self):
        """
        Print summary of the partitioner.
        """
        n_subjects = self.number_of_subjects()
        summary_str = '\n\nNumber of subjects {}, '.format(n_subjects)
        if self._file_list is not None:
            summary_str += 'input section names: {}\n'.format(
                list(self._file_list))
        if self._partition_ids is not None and n_subjects > 0:
            n_train = self.number_of_subjects(TRAIN)
            n_valid = self.number_of_subjects(VALID)
            n_infer = self.number_of_subjects(INFER)
            summary_str += \
                'Dataset partitioning:\n' \
                '-- {} {} cases ({:.2f}%),\n' \
                '-- {} {} cases ({:.2f}%),\n' \
                '-- {} {} cases ({:.2f}%).\n'.format(
                    TRAIN, n_train, float(n_train) / float(n_subjects) * 100.0,
                    VALID, n_valid, float(n_valid) / float(n_subjects) * 100.0,
                    INFER, n_infer, float(n_infer) / float(n_subjects) * 100.0)
        else:
            summary_str += '-- using all subjects ' \
                           '(without data partitioning).\n'
        return summary_str

    def has_phase(self, phase):
        """

        :return: True if the `phase` subset of images is not empty.
        """
        if self._partition_ids is None or self._partition_ids.empty:
            return False
        return (self._partition_ids[COLUMN_PHASE] == phase).any()

    @property
    def has_training(self):
        """

        :return: True if the TRAIN subset of images is not empty.
        """
        return self.has_phase(TRAIN)

    @property
    def has_inference(self):
        """

        :return: True if the INFER subset of images is not empty.
        """
        return self.has_phase(INFER)

    @property
    def has_validation(self):
        """

        :return: True if the VALID subset of images is not empty.
        """
        return self.has_phase(VALID)

    @property
    def validation_files(self):
        """

        :return: the list of validation filenames.
        """
        if self.has_validation:
            return self.get_file_list(VALID)
        return self.all_files

    @property
    def train_files(self):
        """

        :return: the list of training filenames.
        """
        if self.has_training:
            return self.get_file_list(TRAIN)
        return self.all_files

    @property
    def inference_files(self):
        """

        :return: the list of inference filenames
            (defaulting to list of all filenames if no partition definition)
        """
        if self.has_inference:
            return self.get_file_list(INFER)
        return self.all_files

    @property
    def all_files(self):
        """

        :return: list of all filenames
        """
        return self.get_file_list()

    def reset(self):
        """
        reset all fields of this singleton class.
        """
        self._file_list = None
        self._partition_ids = None
        self.data_param = None
        self.ratios = None
        self.new_partition = False

        self.data_split_file = ""
        self.default_image_file_location = \
            NiftyNetGlobalConfig().get_niftynet_home_folder()
