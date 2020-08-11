import argparse
from hr3d_train import train
from hr3d_eval import infer

from util.data import gen_foreground, DEFAULT_KEYS


def run_pipe(model: str,
             train_data: str,
             test_data: str,
             keys: dict,
             epochs: int = 60,
             cuda_device: int = 0):
    train(model, data=train_data, keys=keys, stage="coarse", epochs=epochs,
          device=cuda_device)
    infer(model, data=train_data, keys=keys, stage="coarse", eval_train=True,
          device=cuda_device)
    train(model, data=train_data, keys=keys, stage="fine", epochs=epochs,
          device=cuda_device)
    infer(model, data=test_data, keys=keys, stage="coarse", device=cuda_device)
    infer(model, data=test_data, keys=keys, stage="fine", device=cuda_device)


def prog_args():
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--model', '-m',
                        required=False,
                        type=str,
                        default="../models/hr3d_sample",
                        help="model directory stores all the checkpoints, logs,"
                             "and results"
                        )
    parser.add_argument('--train_data',
                        required=False,
                        type=str,
                        default=None,
                        help="path to the directory containing the training data"
                             "if not provided the default path from utils.data.TRAIN_DATA"
                             "is used"
                        )
    parser.add_argument('--test_data',
                        required=False,
                        type=str,
                        default=None,
                        help="path to the directory containing the test data"
                             "if not provided the default path from utils.data.TEST_DATA"
                             "is used"
                        )
    parser.add_argument('--create_foreground',
                        action='store_true',
                        help="if provided the foreground mask for masking the training"
                             "and inference of the coarse stage is generated (requires"
                             "simple itk to be installed)"
                        )
    parser.add_argument('--keys',
                        required=False,
                        type=str,
                        nargs=3,
                        default=None,
                        help="keys to identify the volume, ground truth label (containing"
                             "the segmented organs enumerated in one file) and mask (being"
                             "a binary file masking the search space for the training and"
                             "inference), if not provided the default keys from utils.data.DEFAULT_KEYS"
                             "are used"
                        )
    parser.add_argument('--epochs',
                        required=False,
                        type=int,
                        default=60,
                        help="number of epochs to train"
                        )
    parser.add_argument('--cuda',
                        required=False,
                        type=int,
                        default=0,
                        help="cuda device to train on"
                        )
    return parser


if __name__ == "__main__":
    args = prog_args().parse_args()

    keys = None
    if args.keys is not None:
        keys = dict()
        for cnt, key in enumerate(DEFAULT_KEYS.keys()):
            keys[key] = args.keys[cnt]

    if args.create_foreground:
        gen_foreground(args.train_data, args.test_data, keys)

    run_pipe(args.model, args.train_data, args.test_data, keys, args.epochs, args.cuda)



