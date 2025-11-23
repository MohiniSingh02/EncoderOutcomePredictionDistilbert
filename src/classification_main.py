import torch
from lightning.pytorch.cli import LightningCLI

from src.model.dataset import MIMICClassificationDataModule
from src.model.lightning_model import ClassificationModel

torch.set_float32_matmul_precision('high')


class DataAwareModelCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):

        # keep multitask flags in sync
        parser.link_arguments("data.multitask", "model.multitask", apply_on="instantiate")

        # Single-task setup
        parser.link_arguments('data.num_labels', 'model.num_classes', apply_on='instantiate')

        # Multitask setup
        parser.link_arguments('data.num_labels_icd9', 'model.num_classes_icd9', apply_on='instantiate')
        parser.link_arguments('data.num_labels_icd10', 'model.num_classes_icd10', apply_on='instantiate')


if __name__ == '__main__':
    cli = DataAwareModelCLI(ClassificationModel,
                            MIMICClassificationDataModule,
                            save_config_kwargs=dict(overwrite=True)
                            )
    if cli.subcommand == 'fit':
        cli.trainer.test(ckpt_path=cli.trainer.checkpoint_callback.best_model_path,
                         datamodule=cli.datamodule
                         )
