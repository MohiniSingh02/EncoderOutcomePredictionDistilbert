import wandb
import yaml

if __name__ == '__main__':
    versions = ['icd9', 'icd10']
    splits = ['hosp', 'icu']

    for version in versions:
        for split in splits:
            with open(f'sweep-config/{version}_{split}.yaml', 'r') as ymlfile:
                sweep_config = yaml.safe_load(ymlfile)

            print(f'Sweep for {version}_{split}: {wandb.sweep(sweep=sweep_config)}')
