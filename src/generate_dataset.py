#!/usr/bin/env python3
import os.path
import argparse

from HC4CA.classes import *


#
#  handle input arguments
#


def file(filename):
    """
    Check if the file specified by filename exists and if
    it is a file (not directory).
    :param filename: path to file
    :return: same filename
    """
    if os.path.isfile(filename):
        return filename
    else:
        raise FileNotFoundError(filename)


def handle_args(*args):
    #
    # Use argparse to handle parsing the command line arguments.
    #   https://docs.python.org/3/library/argparse.html
    #

    parser = argparse.ArgumentParser(description='Prepare dataset data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset-path', metavar='dir/', type=str, default=None,
                        help='path to dataset')
    parser.add_argument('--metadata-path', metavar='dir/', type=str, default=None,
                        help='path to metadata directory')

    parser.add_argument('--subset', metavar='str1 str2 str3 ...', nargs="+",
                        default=['train', ],
                        help='Data subset name')
    parser.add_argument('--sensors', metavar='str1 str2 str3 ...', nargs="+",
                        default=['acceleration', 'pir', 'video'],
                        help='Sensor\'s names to read')
    parser.add_argument('--freq', metavar='freq-str', type=str, default=None,
                        help="Sampling frequency of dataset,eg:'1S'")

    parser.add_argument('--from-pickle', metavar='.pkl', type=file, default=None,
                        help='Dataframe pickled file')

    parser.add_argument('--dump-dataset', action='store_true',
                        help='Dump dataset object as pickle')
    parser.add_argument('--dump-only', metavar='str1 str2 str3 ...', nargs="+",
                        default='all',
                        help='Dump only specified sensors from'
                             ' [acceleration, pir, rssi, video_hallway, '
                             'video_kitchen, video_living_room]')
    parser.add_argument('--dump-datasubset', action='store_true',
                        help='Dump datasubset object as pickle')

    parser.add_argument('--datatest', action='store_true',
                        help='Use datatest dataset only for quick testing')

    parser.add_argument('--name', metavar='name string', type=str,
                        default='dataset',
                        help='Prefix for output files')

    return parser.parse_args(args)


def main(*args):
    parsed_args = handle_args(*args)
    # print(parsed_args)

    # Check whether we read from files or pickle
    metadata_path = parsed_args.metadata_path

    # read dataset from files
    if parsed_args.dataset_path is not None:
        dataset_path = parsed_args.dataset_path

        # take metadata from default path
        if parsed_args.metadata_path is None:
            metadata_path = dataset_path + '/metadata'

        # datatest has limited samples for faster testing
        if parsed_args.datatest:
            dataset_path = parsed_args.dataset_path + '/datatest'

        # Read entire dataset from files
        sphere = Dataset(data_path=dataset_path,
                         subsets=parsed_args.subset,
                         sensors=parsed_args.sensors,
                         meta_path=metadata_path)
    # read dataset from pickle
    else:
        if parsed_args.from_pickle is None:
            print("pickle file not given")
            raise FileNotFoundError(parsed_args.from_pickle)

        sphere = Dataset.load_from_pickle(parsed_args.from_pickle)

    # resampling of dataset
    if parsed_args.freq is not None:
        sphere.resample(freq=parsed_args.freq)

    if parsed_args.dump_dataset:
        filename = sphere.dump(file_prefix=parsed_args.name)
        print(f'\tdataset output file: {filename}')

    if parsed_args.dump_datasubset:
        # pass dump_only as the list of selected sensors
        # TODO: datasubset name as an argument
        filename = sphere.data['train'].to_pickle(
            file_prefix=parsed_args.name, sensors=parsed_args.dump_only)
        print(f'\tsubset output file: {filename}')


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:])
