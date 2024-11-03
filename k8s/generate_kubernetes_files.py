import os

from jinja2 import Environment, FileSystemLoader

if __name__ == '__main__':
    # Load the Jinja2 template from the file system
    file_loader = FileSystemLoader('.')
    env = Environment(loader=file_loader)

    versions = ['icd9', 'icd10']
    splits = ['hosp', 'icu']

    # Parameters
    params = dict(
        replicas=2,
        image_tag='4a4e199',
        hpo_count=10,
        # model='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        model='UFNLP/gatortronS',
        truncate_again=True,
        model_pvc='large-eod-model-pvc',
        data_pvc='eod-pvc',
        icd9_hosp_sweep_id='5xx1yyfx'
    )

    specific_params = {
        'icd9': {
            'hosp': {'sweep_id': '5xx1yyfx'},
            'icu': {'sweep_id': 'hn48h2n5'}
        },
        'icd10': {
            'hosp': {'sweep_id': 'ffh2r6w6'},
            'icu': {'sweep_id': 's9aqw642'}
        }
    }

    template_names = ['sweep-agent', 'sweep-config', 'single_training']
    for template_name in template_names:
        print(f'Creating {template_name}')
        for version in versions:
            for split in splits:
                file_name = f'/{version}_{split}.yaml'
                print(f'            └ {file_name}...', end='')

                template = env.get_template(f'templates/{template_name}.yaml.j2')

                # Render the template with parameters
                output = template.render(**params, **specific_params[version][split], icd_version=version, mimic_split=split)

                if not os.path.exists(template_name):
                    os.makedirs(template_name)
                with open(template_name + file_name, 'w+') as f:
                    f.write(output)

                print('Done!')
