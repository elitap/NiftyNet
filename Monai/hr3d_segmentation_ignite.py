import os
import sys
import tempfile
import shutil
from glob import glob
import logging
import nibabel as nib
import numpy as np
import torch
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, _prepare_batch, Engine
from ignite.handlers import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

import monai
from monai.transforms import (
    Compose,
    LoadNiftid,
    AddChanneld,
    ScaleIntensityd,
    RandRotated,
    RandZoomd,
    RandCropByPosNegLabeld,
    RandSpatialCropd,
    Spacingd,
    ToTensord,
)
from monai.handlers import (
    StatsHandler,
    TensorBoardStatsHandler,
    TensorBoardImageHandler,
    MeanDice,
    stopping_fn_from_metric, SegmentationSaver, CheckpointLoader,
)
from monai.data import create_test_image_3d, list_data_collate, sliding_window_inference
from monai.networks.utils import predict_segmentation

monai.config.print_config()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def getDataLoader(train=True, valitation=True):

    train_data_path = os.path.abspath("../../data/miccai/full_dataset_nifti/train")
    test_data_path = os.path.abspath("../../data/miccai/full_dataset_nifti/test")

    def load_filepaths(train_data_path):
        volume_key = "volume"
        label_key = "segmentation"
        mask_key = "foreground"

        # Setup transforms, dataset
        images = sorted(glob(os.path.join(train_data_path, '*' + volume_key + '.nii.gz')))
        segs = sorted(glob(os.path.join(train_data_path, '*' + label_key + '.nii.gz')))
        masks = sorted(glob(os.path.join(train_data_path, '*' + mask_key + '.nii.gz')))
        file_dict = [{"img": img, "seg": seg, "mask": mask} for img, seg, mask in zip(images, segs, masks)]
        return file_dict

    train_files = load_filepaths(train_data_path)
    test_files = load_filepaths(test_data_path)

    ret = [None, None]

    if train:
        # define transforms for image and segmentation
        train_transforms = Compose(
            [
                LoadNiftid(keys=["img", "seg", "mask"]),
                AddChanneld(keys=["img", "seg", "mask"]),
                Spacingd(keys=['img', 'seg', "mask"], pixdim=(2.2, 2.2, 2.2),
                         interp_order=(3, 0, 0)),
                ScaleIntensityd(keys=["img"]),
                # differnt interpolation orders just work with the current master!!
                # RandRotated(keys=['image', 'label', "mask"], degrees=5,  reshape=False,
                #            interp_order=(3, 0, 0)), #this just rotates around the z plane its not real 3d
                # RandZoomd(keys=['image', 'label', "mask"], min_zoom=0.92, max_zoom=1.08,
                #          interp_order=(3, 0, 0)),
                RandCropByPosNegLabeld(
                    keys=["img", "seg", "mask"], label_key="mask", size=[16, 16, 16], pos=1, neg=0, num_samples=72
                ),
                ToTensord(keys=["img", "seg"]),
            ]
        )

        # check dataset define dataset, data loader
        # check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
        # check_loader = DataLoader(
        #    check_ds, batch_size=1, num_workers=12, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available()
        # )
        # for i in range(25):
        #     #check_data = monai.utils.misc.first(check_loader)
        #     check_data = next(iter(check_loader))
        #     print(check_data["img"].shape, check_data["seg"].shape, torch.unique(check_data["seg"]),
        #           torch.unique(check_data["mask"]))

        # create a training data loader
        train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
        # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
        # fuck read that comment, batch size set to 1 draws n_namples so 72
        train_ldr = DataLoader(
            train_ds,
            batch_size=1,
            shuffle=True,
            num_workers=12,
            collate_fn=list_data_collate,
            pin_memory=torch.cuda.is_available(),
        )
        ret[0] = train_ldr

    if valitation:

        val_transforms = Compose(
            [
                LoadNiftid(keys=["img", "seg"]),
                AddChanneld(keys=["img", "seg"]),
                Spacingd(keys=['img', 'seg'], pixdim=(2.2, 2.2, 2.2),
                         interp_order=(3, 0, 0)),
                ScaleIntensityd(keys=["img"]),
                ToTensord(keys=["img", "seg"]),
            ]
        )

        # create a validation data loader
        val_ds = monai.data.Dataset(data=test_files, transform=val_transforms)
        val_ldr = DataLoader(
            val_ds,
            batch_size=72,
            num_workers=12,
            collate_fn=list_data_collate,
            pin_memory=torch.cuda.is_available()
        )
        ret[1] = val_ldr

    return ret


def train():
    [train_loader, val_loader] = getDataLoader(valitation=False)
    # Create Model, Loss, Optimizer
    # Create UNet, DiceLoss and Adam optimizer.
    net = monai.networks.nets.HighResNet(
        out_channels=8
    )
    loss = monai.losses.DiceLoss(to_onehot_y=True, do_softmax=True, squared_pred=True)
    lr = 1e-3
    opt = torch.optim.Adam(net.parameters(), lr)
    device = torch.device("cuda:0")

    # Ignite trainer expects batch=(img, seg) and returns output=loss at every iteration,
    # user can add output_transform to return other values, like: y_pred, y, etc.
    def prepare_batch(batch, device=None, non_blocking=False):
        return _prepare_batch((batch["img"], batch["seg"]), device, non_blocking)

    trainer = create_supervised_trainer(net, opt, loss, device, False, prepare_batch=prepare_batch)
    # adding checkpoint handler to save models (network params and optimizer stats) during training
    checkpoint_handler = ModelCheckpoint("./runs/", "net", require_empty=False, n_saved=5)
    trainer.add_event_handler(
        event_name=Events.EPOCH_COMPLETED(every=200), handler=checkpoint_handler, to_save={"net": net, "opt": opt}
    )
    # StatsHandler prints loss at every iteration and print metrics at every epoch,
    # we don't set metrics for trainer here, so just print loss, user can also customize print functions
    # and can use output_transform to convert engine.state.output if it's not loss value
    train_stats_handler = StatsHandler(name="trainer")
    train_stats_handler.attach(trainer)
    # TensorBoardStatsHandler plots loss at every iteration and plots metrics at every epoch, same as StatsHandler
    train_tensorboard_stats_handler = TensorBoardStatsHandler()
    train_tensorboard_stats_handler.attach(trainer)
    if not val_loader is None:
        validation_every_n_iters = 5
        # set parameters for validation
        metric_name = "Mean_Dice"

        def activated_output_transform(output):
            y_pred, y = output
            y_pred = torch.softmax(y_pred, dim=1)
            return y_pred, y

        # add evaluation metric to the evaluator engine
        val_metrics = {metric_name: MeanDice(output_transform=activated_output_transform, to_onehot_y=True)}

        # Ignite evaluator expects batch=(img, seg) and returns output=(y_pred, y) at every iteration,
        # user can add output_transform to return other values
        evaluator = create_supervised_evaluator(net, val_metrics, device, True, prepare_batch=prepare_batch)

        @trainer.on(Events.ITERATION_COMPLETED(every=validation_every_n_iters))
        def run_validation(engine):
            evaluator.run(val_loader, epoch_length=1)

        # add early stopping handler to evaluator
        # early_stopper = EarlyStopping(patience=200, score_function=stopping_fn_from_metric(metric_name), trainer=trainer)
        # evaluator.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=early_stopper)

        # add stats event handler to print validation stats via evaluator
        val_stats_handler = StatsHandler(
            name="evaluator",
            output_transform=lambda x: None,  # no need to print loss value, so disable per iteration output
            global_epoch_transform=lambda x: trainer.state.epoch,
        )  # fetch global epoch number from trainer
        val_stats_handler.attach(evaluator)

        # add handler to record metrics to TensorBoard at every validation epoch
        val_tensorboard_stats_handler = TensorBoardStatsHandler(
            output_transform=lambda x: None,  # no need to plot loss value, so disable per iteration output
            global_epoch_transform=lambda x: trainer.state.iteration,
        )  # fetch global iteration number from trainer
        val_tensorboard_stats_handler.attach(evaluator)

        # add handler to draw the first image and the corresponding label and model output in the last batch
        # here we draw the 3D output as GIF format along the depth axis, every 2 validation iterations.
        # val_tensorboard_image_handler = TensorBoardImageHandler(
        #     batch_transform=lambda batch: (batch["img"], batch["seg"]),
        #     output_transform=lambda output: predict_segmentation(output[0]),
        #     global_iter_transform=lambda x: trainer.state.epoch,
        # )
        # evaluator.add_event_handler(event_name=Events.ITERATION_COMPLETED(every=2), handler=val_tensorboard_image_handler)
    train_epochs = 3000
    state = trainer.run(train_loader, train_epochs)
    print(state)


def evaluate():
    [_, val_loader] = getDataLoader(train=False, valitation=True)
    # Create Model
    device = torch.device("cuda:0")
    net = monai.networks.nets.HighResNet(
        out_channels=8
    ).to(device)

    def _sliding_window_processor(engine, batch):
        net.eval()
        with torch.no_grad():
            val_images, val_labels = batch['img'].to(device), batch['seg'].to(device)
            seg_probs = sliding_window_inference(val_images, (16, 16, 16), 1, net)
            return seg_probs, val_labels

    evaluator = Engine(_sliding_window_processor)

    def activated_output_transform(output):
        y_pred, y = output
        y_pred = torch.softmax(y_pred, dim=1)
        return y_pred, y

    # add evaluation metric to the evaluator engine
    MeanDice(output_transform=activated_output_transform, to_onehot_y=True).attach(evaluator, "Mean_Dice")

    # StatsHandler prints loss at every iteration and print metrics at every epoch,
    # we don't need to print loss for evaluator, so just print metrics, user can also customize print functions
    val_stats_handler = StatsHandler(
        name='evaluator',
        output_transform=lambda x: None  # no need to print loss value, so disable per iteration output
    )
    val_stats_handler.attach(evaluator)

    # TODO batches in segementation saver have to full resolution!!!!
    # convert the necessary metadata from batch data
    SegmentationSaver(output_dir='./runs/output/', output_ext='.nii.gz', output_postfix='seg', name='evaluator',
                      batch_transform=lambda batch: {'filename_or_obj': batch['img.filename_or_obj'],
                                                     'affine': batch['img.affine']},
                      output_transform=lambda output: predict_segmentation(output[0], mutually_exclusive=True)).attach(evaluator)
    # the model was trained by "unet_training_dict" example
    CheckpointLoader(load_path='./runs/net_checkpoint_75000.pth', load_dict={'net': net}).attach(evaluator)

    state = evaluator.run(val_loader)


if __name__ == "__main__":

    train = True

    if train:
        train()
    else:
        evaluate()
