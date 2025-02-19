import subprocess
import yaml

def train_model(split_number):
    yaml_file_path = f"data/VOCdevkit-Peras/VOC2007/2025-02-03_5-Fold_Cross/split_{split_number}/split_{split_number}_dataset.yaml"

    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)

    data['split'] = split_number


    with open(yaml_file_path, 'w') as file:
        yaml.safe_dump(data, file)

    command = f"python train.py --data {yaml_file_path}"
    subprocess.run(command, shell=True)

for i in range(1, 6):
    train_model(i)
