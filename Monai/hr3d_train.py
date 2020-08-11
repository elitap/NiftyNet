import monai
import torch
import logging
from logging import basicConfig, StreamHandler, FileHandler
import sys
import os
from ignite.engine import _prepare_batch
from monai.engines import create_supervised_trainer, create_supervised_evaluator
from monai.handlers import StatsHandler, TensorBoardStatsHandler, MeanDice, CheckpointLoader, CheckpointSaver
from monai.handlers.checkpoint_saver import Events

from loss.mydice import MyDiceLoss
from util.data import DLCreator, get_checkpoint_file

WITH_VALIDATION = False
VAL_EVERY_N = 5


def train(modeldir: str,
          data: str = None,
          keys: dict = None,
          epoch: int = -1,
          epochs: int = 120,
          stage: str = "coarse",
          device: int = 0):

    os.makedirs(modeldir, exist_ok=True)
    monai.config.print_config()

    # logging
    basicConfig(handlers=[StreamHandler(sys.stdout),
                         FileHandler(os.path.join(modeldir, stage+"_train_log.txt"))],
                level=logging.INFO)

    coarse_out = os.path.join(modeldir, "output", "coarse", "train")
    if not stage is "coarse":
        assert os.path.exists(coarse_out), "eval results of the training " \
                                           "data from the coarse stage is not available under\n" + coarse_out
    train_loader = DLCreator(data=data, keys=keys, stage=stage,
                             fine_stage_mask_path=coarse_out).get_data_loader()

    # Create Model, Loss, Optimizer
    net = monai.networks.nets.HighResNet(out_channels=8)
    loss = MyDiceLoss(to_onehot_y=True, softmax=True, squared_pred=True)
    lr = 1e-3
    opt = torch.optim.Adam(net.parameters(), lr)
    device = torch.device("cuda:{}".format(device))

    # Ignite trainer expects batch=(img, seg) and returns output=loss at every iteration,
    # user can add output_transform to return other values, like: y_pred, y, etc.
    def prepare_batch(batch, device=None, non_blocking=False):
        return _prepare_batch((batch["img"], batch["seg"]), device, non_blocking)

    trainer = create_supervised_trainer(net, opt, loss, device, False, prepare_batch=prepare_batch)

    # adding checkpoint handler to save models (network params and optimizer stats) during training
    save_dict = {"net": net, "opt": opt}
    checkpoint_save_handler = CheckpointSaver(save_dir=os.path.join(modeldir, "ckpts", stage),
                                              save_dict=save_dict,
                                              epoch_level=True,
                                              save_final=True,
                                              save_interval=10)
    checkpoint_save_handler.attach(trainer)

    # StatsHandler prints loss at every iteration and print metrics at every epoch,
    # we don't set metrics for trainer here, so just print loss, user can also customize print functions
    # and can use output_transform to convert engine.state.output if it's not loss value
    train_stats_handler = StatsHandler(name="trainer")
    train_stats_handler.attach(trainer)
    # TensorBoardStatsHandler plots loss at every iteration and plots metrics at every epoch, same as StatsHandler
    train_tensorboard_stats_handler = TensorBoardStatsHandler(log_dir=os.path.join(modeldir, "log", stage))
    train_tensorboard_stats_handler.attach(trainer)

    ckpt_file, epoch = get_checkpoint_file(os.path.join(modeldir, "ckpts", stage), epoch)
    if ckpt_file is not None:
        checkpoint_load_handler = CheckpointLoader(load_path=ckpt_file,
                                                   load_dict=save_dict)
        checkpoint_load_handler.attach(trainer)

        @trainer.on(Events.STARTED)
        def resume_training(engine):
            engine.state.iteration = (epoch) * len(engine.state.dataloader)
            engine.state.epoch = (epoch)


    if WITH_VALIDATION:
        val_loader = DLCreator(train=False, stage=stage).get_data_loader()

        # set parameters for validation
        metric_name = "Mean_Dice"

        def activated_output_transform(output):
            y_pred, y = output
            y_pred = torch.softmax(y_pred, dim=1)
            return y_pred, y

        # add evaluation metric to the evaluator engine
        val_metrics = {metric_name: MeanDice(output_transform=activated_output_transform, to_onehot_y=True,
                                             mutually_exclusive=True)}

        # Ignite evaluator expects batch=(img, seg) and returns output=(y_pred, y) at every iteration,
        # user can add output_transform to return other values
        evaluator = create_supervised_evaluator(net, val_metrics, device, True, prepare_batch=prepare_batch)

        @trainer.on(Events.EPOCH_COMPLETED(every=VAL_EVERY_N))
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
    state = trainer.run(train_loader, epochs)
    print(state)


if __name__ == "__main__":
    train("../models/mydice", stage="fine")
