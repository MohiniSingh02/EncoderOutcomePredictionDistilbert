from lightning.pytorch.cli import LightningCLI

from lightning_model import ClassificationModel
from dataset import MIMICClassificationDataModule


class DataAwareModelCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments('data.num_labels', 'model.num_classes', apply_on='instantiate')


if __name__ == '__main__':
    cli = DataAwareModelCLI(ClassificationModel,
                            MIMICClassificationDataModule,
                            save_config_kwargs=dict(overwrite=True),
                            # trainer_defaults=dict(logger=WandbLogger(project='EncoderOutcomePrediction', anonymous=True))
                            )
