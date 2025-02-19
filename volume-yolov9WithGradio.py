import detect
import gradio as gr
from PIL import Image
import os
import math
from pathlib import Path
import re
import cv2

def find_latest_file_by_extension(folder_path, extension):
    """查找给定文件夹中具有特定扩展名的最新文件"""
    latest_file = None
    latest_time = 0
    # 遍历文件夹中所有文件
    for file in Path(folder_path).glob(f'*{extension}'):
        # 获取文件的修改时间
        file_time = os.path.getmtime(file)
        # 如果文件是更近的，则更新最新文件变量
        if file_time > latest_time:
            latest_file = file
            latest_time = file_time
    return latest_file

def find_latest_folder(base_path, prefix="youzidemo"):
    """查找最新的符合前缀的文件夹"""
    base_path = Path(base_path)
    all_folders = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    # 使用正则表达式匹配确保文件夹名符合'youzi'后跟数字的模式
    filtered_folders = [d for d in all_folders if re.match(rf'{prefix}\d+', d.name)]
    if not filtered_folders:
        return None
    latest_folder = max(filtered_folders, key=os.path.getmtime)
    return latest_folder

def read_labels_from_latest_run(base_path=r".\runs\detect", prefix="youzidemo"):
    """从最新的运行中读取图片地址和标签文件内容"""
    latest_folder = find_latest_folder(base_path, prefix)
    if not latest_folder:
        print("没有找到符合条件的最新文件夹")
        return None, None
    
    result_image_path = find_latest_file_by_extension(latest_folder, '.jpg')
    label_file_path = find_latest_file_by_extension(latest_folder/ "labels", '.txt') 
    
    return result_image_path, label_file_path

def draw_boxes(img_path, labels_path):
    """
    从标签文件读取边界框信息，并在原图上绘制边界框和序号标记。
    
    :param img_path: 原始图像的路径。
    :param labels_path: 标签文件的路径。
    """
    # 读取图像
    image = cv2.imread(img_path)
    # 确保图像加载成功
    if image is None:
        print(f"无法加载图像: {img_path}")
        return

    # 获取图像尺寸
    img_height, img_width = image.shape[:2]
    
    pixel_width = []
    target_x_center = []

    # 读取并解析标签文件
    with open(labels_path, 'r') as file:
        lines = file.readlines()
        for index, line in enumerate(lines, start=1):  # 从1开始计数
            # 解析类别和边界框信息
            parts = line.strip().split()
            _, x_center, y_center, width, height = map(float, parts)
            
            # 将边界框坐标转换为图像尺寸
            x_center, y_center, width, height = x_center * img_width, y_center * img_height, width * img_width, height * img_height
            
            # 计算边界框的左上角和右下角坐标
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)
            
            # 在图像上绘制边界框和序号标记
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, str(index), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            pixel_width.append(img_width)
            target_x_center.append(x_center)

    return image,pixel_width,target_x_center

def distanceStereo(distanceBetweenPictures,angleView,widthPixel,targetPixelInLeft,targetPixelInRight):
    
    B = float(distanceBetweenPictures)  # 相机基线距离的单位值
    x0 = widthPixel  # 第一幅图像中物体的单位尺寸
    theta_0 = float(angleView)  # 
    x1 = targetPixelInLeft  # 第二幅图像中物体的像素坐标
    x2 = targetPixelInRight   # 第一幅图像中物体的像素坐标

    D = (B * x0) / (2 * math.tan(theta_0 / 2) * (x1 - x2))

    return D

def volumeEstimation(object_pixel, S, D,f):

    # f = 50  # 焦距，单位：mm
    # D = 2000  # 相机到物体的距离，单位：mm
    # S = 0.005  # 每个像素的大小，单位：mm
    # p = 400  # 物体在图像中的宽度，单位：像素

    # W = (p * S * D) / f
    # print(f"物体的实际宽度是: {W}mm")

    # S = img_width  # 图像的总宽度，单位为像素
   # 物体在图像中的宽度，单位为像素
    result_texts = []  # 用于存储每个物体的体积计算结果

    for index, p in enumerate(object_pixel, start=1):  # 从1开始计数
        # 计算物体的实际宽度
        W = (p * S * D) / f

        radius = W / 2
        volume = (4/3) * math.pi * (radius ** 3)
        # volume = volume / 1000000
        result_texts.append(f"物体 {index} 的体积是: {volume:.2f} 立方分米")

    return "\n".join(result_texts)  # 返回所有结果的组合文本

def objectOetection(input_image):

    detect.run(source = input_image, device = "cpu", weights="youzi.pt", name="youzidemo", save_txt=True)

# python detect.py --source ./data/images/youzi609.jpg --img 640 --device cpu --weights youzi.pt --name youzi --save-txt

def process_image(input_image_left, input_image_right,distanceBetweenPictures,angleView,cameraFocus,sensorPixels):
    """
    处理上传的图像并返回处理结果。
    """
    # 保存上传的图像到临时文件，以便`objectOetection`可以访问
    input_image_path_left = "temp_input_left.jpg"
    input_image_path_right = "temp_input_right.jpg"

    #左侧
    input_image_left.save(input_image_path_left)
    input_image_right.save(input_image_path_right)

    # 执行目标检测
    objectOetection(input_image_path_left)
    _, labels_path = read_labels_from_latest_run()
    result_image_left, pixel_width, target_x_left = draw_boxes(input_image_path_left, labels_path)
    imgLeft = Image.open(input_image_path_left)
    img_width_left, _ = imgLeft.size


    #右侧
    objectOetection(input_image_path_right)
    _, labels_path = read_labels_from_latest_run()
    result_image, pixel_width, target_x_right = draw_boxes(input_image_path_right, labels_path)
    imgRight = Image.open(input_image_path_right)
    img_width_right, _ = imgRight.size

    target_x_num = len(target_x_left) if len(target_x_left) > len(target_x_Right) else len(target_x_Right)

    for i in range(target_x_num):
        D = distanceStereo(distanceBetweenPictures,angleView, img_width_left,target_x_left[i],target_x_right[i])
        output_text = volumeEstimation(pixel_width, float(sensorPixels), D, float(cameraFocus))

    cv2.imwrite("processed_image.jpg", result_image_left)

    # 确保返回的是文件路径或PIL.Image对象
    output_image = Image.open("processed_image.jpg")

    return output_image, output_text

with gr.Blocks() as app:

    gr.Markdown("## 基于YOLOV9的柚子体积估计")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 输入图像")
            input_image_left = gr.Image(label="左侧图像",type='pil')
            input_image_right = gr.Image(label="右侧图像",type='pil')

        with gr.Column():
            gr.Markdown("### 输出图像")
            output_image = gr.Image(label="处理后的图像")

    distanceBetweenPictures = gr.Textbox(label="两个相机间的基线距离")
    angleView = gr.Textbox(label="视场角度")
    cameraFocus = gr.Textbox(label="相机焦距")
    sensorPixels = gr.Textbox(label="传感器像素大小")
    output_text = gr.Textbox(label="体积计算结果")
    
    # 按钮触发函数执行
    process_button = gr.Button("执行代码")
    process_button.click(
        process_image, 
        inputs=[input_image_left,input_image_right,distanceBetweenPictures,angleView,cameraFocus,sensorPixels],
        outputs=[output_image, output_text]
    )

app.launch(share=True)