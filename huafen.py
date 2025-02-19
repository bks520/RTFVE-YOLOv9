import datetime
import shutil
from pathlib import Path
from collections import Counter
import os
import yaml
import pandas as pd
from sklearn.model_selection import KFold

# 定义数据集路径
dataset_path = Path(r'E:\WPS Cloud\OpenCV\yolov9-main-down\data\VOCdevkit-apples\VOC2007')  # 替换成你的数据集路径

# 定义 images 和 labels 文件夹路径
images_dir = dataset_path / 'images'
labels_dir = dataset_path / 'labels'

# 检查 images 和 labels 文件夹是否存在
if not images_dir.exists() or not labels_dir.exists():
    raise FileNotFoundError(f"未找到 'images' 或 'labels' 文件夹，请检查路径：{images_dir}, {labels_dir}")

# 获取所有标签文件的列表
labels = sorted(labels_dir.rglob("*.txt"))  # 所有标签文件在 'labels' 目录中

# 检查是否找到标签文件
if not labels:
    raise FileNotFoundError(f"未找到任何标签文件，请检查路径：{labels_dir}")

# 打印找到的标签文件数量
print(f"找到的标签文件数量: {len(labels)}")

# 从 YAML 文件加载类名
yaml_file = dataset_path / 'classes.yaml'
if not yaml_file.exists():
    raise FileNotFoundError(f"YAML 文件未找到：{yaml_file}")

with open(yaml_file, 'r', encoding="utf8") as y:
    classes = yaml.safe_load(y)['names']
cls_idx = sorted(classes.keys())

# 创建 DataFrame 来存储每张图像的标签计数
indx = [l.stem for l in labels]  # 使用基本文件名作为 ID（无扩展名）
labels_df = pd.DataFrame([], columns=cls_idx, index=indx)

# 计算每张图像的标签计数
for label in labels:
    lbl_counter = Counter()
    with open(label, 'r') as lf:
        lines = lf.readlines()

    for l in lines:
        # YOLO 标签使用每行的第一个位置的整数作为类别
        lbl_counter[int(l.split(' ')[0])] += 1

    labels_df.loc[label.stem] = lbl_counter

# 检查标签数据是否为空
if labels_df.empty:
    raise ValueError("标签数据为空，请检查标签文件是否正确加载。")

# 用 0.0 替换 NaN 值
labels_df = labels_df.fillna(0.0)

# 使用 K-Fold 交叉验证拆分数据集
ksplit = min(10, len(labels_df))  # 最大分割数不超过样本数
kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)  # 设置 random_state 以获得可重复的结果
kfolds = list(kf.split(labels_df))
folds = [f'split_{n}' for n in range(1, ksplit + 1)]
folds_df = pd.DataFrame(index=indx, columns=folds)

# 为每个折叠分配图像到训练集或验证集
for idx, (train, val) in enumerate(kfolds, start=1):
    folds_df[f'split_{idx}'].loc[labels_df.iloc[train].index] = 'train'
    folds_df[f'split_{idx}'].loc[labels_df.iloc[val].index] = 'val'

# 创建目录以保存分割后的数据集
save_path = Path(dataset_path / f'{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross')
save_path.mkdir(parents=True, exist_ok=True)

# 获取图像文件列表
images = sorted(images_dir.rglob("*.jpg"))  # 更改文件扩展名以匹配你的数据
if not images:
    raise FileNotFoundError(f"未找到任何图像文件，请检查路径：{images_dir}")

# 检查索引和文件名是否一致
image_stems = [image.stem for image in images]
print(f"图像文件名: {image_stems}")
print(f"folds_df.index: {folds_df.index}")

# 创建保存的 YAML 文件路径列表
ds_yamls = []

# 循环遍历每个折叠并复制图像和标签
for split in folds_df.columns:
    # 为每个折叠创建目录
    split_dir = save_path / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
    (split_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
    (split_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
    (split_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)

    # 创建数据集的 YAML 文件
    dataset_yaml = split_dir / f'{split}_dataset.yaml'
    ds_yamls.append(dataset_yaml.as_posix())

    with open(dataset_yaml, 'w') as ds_y:
        yaml.safe_dump({
            'path': str(split_dir),
            'train': 'train',
            'val': 'val',
            'names': classes
        }, ds_y)
print("生成的 YAML 文件列表:", ds_yamls)

# 将文件路径保存到一个 txt 文件中
with open(dataset_path / 'file_paths.txt', 'w') as f:
    for path in ds_yamls:
        f.write(path + '\n')

# 为每个折叠复制图像和标签到相应的目录
for image, label in zip(images, labels):
    if image.stem not in folds_df.index:
        print(f"跳过未匹配文件: {image.stem}")
        continue

    for split, k_split in folds_df.loc[image.stem].items():
        # 目标目录
        img_to_path = save_path / split / k_split / 'images'
        lbl_to_path = save_path / split / k_split / 'labels'

        # 将图像和标签文件复制到新目录中
        shutil.copy(image, img_to_path / image.name)
        shutil.copy(label, lbl_to_path / label.name)
