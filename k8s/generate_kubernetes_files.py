import os

from jinja2 import Environment, FileSystemLoader

if __name__ == '__main__':
    # Load the Jinja2 template from the file system
    file_loader = FileSystemLoader('.')
    env = Environment(loader=file_loader)

    versions = ['icd9', 'icd10']
    splits = ['hosp', 'icu']
    modes = ['baseline', 'multitask']
    template_names = ['sweep-agent', 'sweep-config', 'single_training']

    # Parameters for single-training
    params = dict(
        replicas=2,
        image_tag='multitask',
        image_registry='registry.datexis.com/tsteffek/encoderoutcomeprediction',
        hpo_count=10,
        model='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        # model='UFNLP/gatortronS',
        truncate_again=True,
        model_pvc='large-eod-model-pvc',
        data_pvc='eod-pvc',
    )

    specific_params = {
        'icd9': {
            'hosp': {'sweep_id': '5xx1yyfx'},
            'icu': {'sweep_id': 'hn48h2n5'}
        },
        'icd10': {
            'hosp': {'sweep_id': 'ffh2r6w6'},
            'icu': {'sweep_id': 's9aqw642'}
        },
    }

    # Multitask parameters
    multitask_params = {
        'hosp': {'sweep_id': 'g6kw3wzg'},
        'icu': {'sweep_id':  'pe11u4kh'}
    }


    def get_param_style(multitask: bool):
        """Returns dict defining how parameters should be named in sweeps."""
        if multitask:
            # Flat names for multitask for readability
            return dict(
                lr="lr",
                warmup_steps="warmup_steps",
                weight_decay="weight_decay",
                decay_steps="decay_steps",
                batch_size="batch_size",
            )
        else:
            # Hierarchical names for baseline/single-task sweeps
            return dict(
                lr="model.lr",
                warmup_steps="model.warmup_steps",
                weight_decay="model.weight_decay",
                decay_steps="model.decay_steps",
                batch_size="data.batch_size",
            )


    # generating single training (baseline) YAMls
    for template_name in template_names:
        print(f'Creating {template_name}')
        for version in versions:
            for split in splits:
                file_name = f'/{version}_{split}.yaml'
                print(f'            └ {file_name}...', end='')

                template = env.get_template(f'templates/{template_name}.yaml.j2')

                naming = get_param_style(multitask=False)

                # Rendering the template with parameters
                output = template.render(**params, **specific_params[version][split], icd_version=version,
                                         mimic_split=split, multitask=False, **naming,)

                os.makedirs(template_name, exist_ok=True)
                with open(template_name + file_name, 'w+') as f:
                    f.write(output)
                print('Done!')

    # Generate multitask YAMLs
    for template_name in template_names:
        print(f'Creating multitask {template_name}')
        for split in splits:
            if template_name == "sweep-agent":
                file_name = f'/sweep-agent_multitask_{split}.yaml'
            elif template_name == "sweep-config":
                file_name = f'/sweep-config_multitask_{split}.yaml'
            else:
                file_name = f'/multitask_{split}.yaml'

            print(f'            └ {file_name}...', end='')

            template = env.get_template(f'templates/{template_name}.yaml.j2')

            split_params = {
                'icd9_data_dir': f"/data-pvc/data/mimic-iv/icd9/{split}",
                'icd10_data_dir': f"/data-pvc/data/mimic-iv/icd10/{split}",
                'multitask': True,
                **multitask_params[split],
            }

            naming = get_param_style(multitask=True)

            # Rendering the template with multitask params
            output = template.render(
                **params,
                **split_params,
                icd_version="multitask",
                mimic_split=split,
                **naming,
            )

            os.makedirs(template_name, exist_ok=True)
            with open(template_name + file_name, 'w+') as f:
                f.write(output)
            print('Done!')
