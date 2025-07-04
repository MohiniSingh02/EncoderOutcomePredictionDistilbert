import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lightning.pytorch.cli import LightningCLI

from model.dataset import MIMICClassificationDataModule
from model.lightning_model import ClassificationModel

torch.set_float32_matmul_precision('high')


class DataAwareModelCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments('data.num_labels', 'model.num_classes', apply_on='instantiate')


if __name__ == '__main__':
    cli = DataAwareModelCLI(ClassificationModel,
                            MIMICClassificationDataModule,
                            save_config_kwargs=dict(overwrite=True)
                            )
    if cli.subcommand == 'fit':
        cli.trainer.test(ckpt_path='best', datamodule=cli.datamodule)
