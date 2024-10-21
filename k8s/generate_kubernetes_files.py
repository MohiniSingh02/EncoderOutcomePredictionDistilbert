import os

from jinja2 import Environment, FileSystemLoader

if __name__ == '__main__':
    # Load the Jinja2 template from the file system
    file_loader = FileSystemLoader('.')
    env = Environment(loader=file_loader)
    template = env.get_template('templates/deployment.yaml.j2')

    versions = ['icd9', 'icd10']
    splits = ['hosp', 'icu']

    # Parameters
    params = dict(
        replicas=5,
        image_tag='8fd95ed',
        hpo_count=5,
        model='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        truncate_again=True,
        model_pvc='large-eod-model-pvc',
        data_pvc='eod-pvc',
    )

    template_names = ['deployment', 'sweep-config', 'single_training']
    for template_name in template_names:
        print(f'Creating {template_name}')
        for version in versions:
            for split in splits:
                file_name = f'/{version}_{split}.yaml'
                print(f'            └ {file_name}...', end='')

                template = env.get_template(f'templates/{template_name}.yaml.j2')

                # Render the template with parameters
                output = template.render(**params, icd_version=version, mimic_split=split)

                if not os.path.exists(template_name):
                    os.makedirs(template_name)
                with open(template_name + file_name, 'w+') as f:
                    f.write(output)

                print('Done!')
