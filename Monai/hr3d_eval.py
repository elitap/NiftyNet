import monai
import torch
import sys
import os
import logging
import numpy as np
from logging import basicConfig, StreamHandler, FileHandler

from ignite.engine import Engine
from monai.handlers import StatsHandler, SegmentationSaver, CheckpointLoader, MeanDice

from util.infer import sliding_window_inference
#from monai.inferers import sliding_window_inference
from monai.networks.utils import predict_segmentation

from util.data import get_checkpoint_file, DLCreator, remove_subfolders


def infer(modeldir: str,
          data: str = None,
          keys: dict = None,
          epoch: int = -1,
          stage: str = "coarse",
          eval_train: bool = False,
          device: int = 0):

    monai.config.print_config()
    # logging
    basicConfig(handlers=[StreamHandler(sys.stdout),
                         FileHandler(os.path.join(modeldir, stage + "_test_log.txt"))],
                level=logging.INFO)

    coarse_out = os.path.join(modeldir, "output", "coarse")
    if eval_train:
        coarse_out = os.path.join(coarse_out, "train")

    dl_creator = DLCreator(data=data, keys=keys, stage=stage,
                           train=False, eval_train=eval_train,
                           fine_stage_mask_path=coarse_out)
    loader = dl_creator.get_data_loader()

    device = torch.device("cuda:{}".format(device))
    net = monai.networks.nets.HighResNet(out_channels=8).to(device)

    def _sliding_window_processor(engine, batch):
        slw_device = torch.device('cpu')
        net.eval()
        with torch.no_grad():
            # inference should still be done on main device
            val_images = batch['img'].to(device)

            # after inference all callcs are done on cpu as gpu mem is to small
            val_labels, val_mask = batch['seg'].to(slw_device), \
                                   batch['mask'].to(slw_device)
            seg_probs = sliding_window_inference(val_images,
                                                 dl_creator.spatial_size,
                                                 dl_creator.num_samples,
                                                 net,
                                                 mask=val_mask,
                                                 overlap=0.25,
                                                 device=slw_device
                                                 )
            return seg_probs, val_labels

    evaluator = Engine(_sliding_window_processor)

    def activated_output_transform(output):
        y_pred, y = output
        y_pred = torch.softmax(y_pred, dim=1)
        return y_pred, y

    # add evaluation metric to the evaluator engine
    MeanDice(output_transform=activated_output_transform, to_onehot_y=True,
             mutually_exclusive=True, device=torch.device('cpu')).attach(evaluator, "Mean_Dice")

    # StatsHandler prints loss at every iteration and print metrics at every epoch,
    # we don't need to print loss for evaluator, so just print metrics, user can also customize print functions
    val_stats_handler = StatsHandler(
        name='evaluator',
        output_transform=lambda x: None  # no need to print loss value, so disable per iteration output
    )
    val_stats_handler.attach(evaluator)

    # convert the necessary metadata from batch data
    output_dir = os.path.join(modeldir, "output", stage)
    if eval_train:
        output_dir = os.path.join(output_dir, "train")

    SegmentationSaver(output_dir=output_dir,
                      output_ext='.nii.gz',
                      output_postfix=stage,
                      dtype=np.uint8,
                      batch_transform=lambda batch: batch["seg_meta_dict"],
                      # batch_transform=lambda batch: {'filename_or_obj': batch['img.filename_or_obj'],
                      #                                'affine': batch['img.affine'],
                      #                                'spatial_shape' : batch['img.spatial_shape'],
                      #                                'original_affine': batch['img.original_affine'],
                      #                                },
                      output_transform=lambda output: predict_segmentation(output[0], mutually_exclusive=True)
                      ).attach(
        evaluator)

    ckpt_file, _ = get_checkpoint_file(os.path.join(modeldir, "ckpts", stage), epoch)
    assert ckpt_file is not None, "no checkpoint found in model dir, train before evaluate"

    CheckpointLoader(load_path=ckpt_file,
                     load_dict={'net': net}).attach(evaluator)

    state = evaluator.run(loader)
    print(state)

    remove_subfolders(output_dir)


if __name__ == "__main__":
    infer("../models/test_pipe", stage="coarse", eval_train=True)
