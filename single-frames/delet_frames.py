import cv2
import numpy as np
import os
import pickle

def crop_bbox_to_square(img, bbox, output_size=(256, 256)):
    """
    根据 bbox 构造正方形裁剪区域，并将裁剪结果调整为指定尺寸
    :param img: 输入图像
    :param bbox: 边界框坐标，格式为 [x_min, y_min, x_max, y_max]
    :param output_size: 输出图像尺寸，默认为 (256, 256)
    :return: 裁剪并缩放后的图像
    """
    x_min, y_min, x_max, y_max = bbox
    img_h, img_w = img.shape[:2]

    bbox_w = x_max - x_min
    bbox_h = y_max - y_min

    # 使用 bbox 的最大边长构造正方形区域
    side = max(bbox_w, bbox_h)

    # 以 bbox 中心为正方形中心
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    new_x_min = max(0, int(center_x - side / 2))
    new_y_min = max(0, int(center_y - side / 2))
    new_x_max = min(img_w, int(center_x + side / 2))
    new_y_max = min(img_h, int(center_y + side / 2))

    # 如果裁剪区域太小，则直接缩放整个图像
    if (new_x_max - new_x_min < 10) or (new_y_max - new_y_min < 10):
        return cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)

    cropped_img = img[new_y_min:new_y_max, new_x_min:new_x_max]
    resized_img = cv2.resize(cropped_img, output_size, interpolation=cv2.INTER_LINEAR)

    return resized_img

def process_pkl_file(pkl_file, output_base="output"):
    """
    加载 pkl 文件，遍历其中每个条目，根据 img_path 和 bbox 裁剪图像，
    并在 output 文件夹下按照 img_path 的目录结构保存结果图像
    :param pkl_file: pkl 文件路径
    :param output_base: 输出文件夹基础路径
    """
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    for item in data:
        rel_img_path = item.get("img_path")
        bbox = item.get("bbox")
        if rel_img_path is None or bbox is None:
            continue

        full_img_path = os.path.join("datasets", "videos", rel_img_path)
        img = cv2.imread(full_img_path)
        if img is None:
            print(f"无法加载图像: {full_img_path}")
            continue

        # 支持两种 bbox 格式： [x_min, y_min, x_max, y_max] 或 [[x_min, y_min], [x_max, y_max]]
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()
        if isinstance(bbox, (list, tuple)):
            if len(bbox) == 4 and all(isinstance(i, (int, float)) for i in bbox):
                pass  # 格式正确
            elif len(bbox) == 2 and all(isinstance(b, (list, tuple)) for b in bbox):
                (x1, y1), (x2, y2) = bbox
                bbox = [x1, y1, x2, y2]
            else:
                print(f"无效的 bbox 格式: {bbox}")
                continue
        else:
            print(f"无效的 bbox 类型: {bbox}")
            continue

        cropped_img = crop_bbox_to_square(img, bbox, output_size=(256, 256))
        save_path = os.path.join(output_base, rel_img_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cropped_img)
        print(f"处理并保存图像: {save_path}")

if __name__ == '__main__':
    # 分别处理 train.pkl 和 val.pkl 文件
    process_pkl_file("datasets/train.pkl")
    process_pkl_file("datasets/val.pkl")

# import os
# import cv2
# import pickle
# import numpy as np

# def draw_bbox_on_image(image_path, bbox):
#     """
#     读取图像，并在图像上根据 bbox 坐标绘制矩形框
#     :param image_path: 完整图像路径
#     :param bbox: 边界框坐标，可以是 numpy 数组、列表或元组，
#                  支持格式 [x1, y1, x2, y2] 或 [[x1, y1], [x2, y2]]
#     :return: 绘制好边界框的图像
#     """
#     # 读取图像
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"无法加载图像: {image_path}")
#         return None

#     # 如果 bbox 是 numpy 数组，则先转换为列表
#     if isinstance(bbox, np.ndarray):
#         bbox = bbox.tolist()

#     try:
#         if isinstance(bbox, (list, tuple)):
#             if len(bbox) == 4 and all(isinstance(i, (int, float)) for i in bbox):
#                 # 格式为 [x1, y1, x2, y2]
#                 x1, y1, x2, y2 = bbox
#             elif len(bbox) == 2 and all(isinstance(b, (list, tuple)) for b in bbox):
#                 # 格式为 [[x1, y1], [x2, y2]]
#                 x1, y1 = bbox[0]
#                 x2, y2 = bbox[1]
#             else:
#                 raise ValueError("Bbox 格式不正确")
#             # 将坐标转换为整数
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#         else:
#             raise TypeError("Bbox 不是列表或元组")
#     except Exception as e:
#         print(f"处理 Bbox 时出错: {bbox}，错误信息: {e}")
#         return image

#     # 绘制矩形框，颜色为绿色，线宽为 2
#     image_with_bbox = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     return image_with_bbox

# def process_pkl_file(pkl_file, output_base="output"):
#     """
#     加载 pkl 文件，遍历其中每个条目，根据 img_path 和 Bbox 画框，
#     并在 output 文件夹下按照 img_path 的目录结构保存结果图像
#     :param pkl_file: pkl 文件路径
#     :param output_base: 输出文件夹基础路径
#     """
#     with open(pkl_file, 'rb') as f:
#         data = pickle.load(f)
    
#     for item in data:
#         # 获取相对路径和 Bbox 坐标
#         rel_img_path = item.get("img_path")
#         bbox = item.get("bbox")
#         if rel_img_path is None or bbox is None:
#             continue

#         # 拼接完整图像路径（前面加上 "datasets/videos" 文件夹）
#         full_img_path = os.path.join("datasets", "videos", rel_img_path)
#         image_with_bbox = draw_bbox_on_image(full_img_path, bbox)
#         if image_with_bbox is not None:
#             # 构造输出路径：保持与原 img_path 相同的目录结构
#             save_path = os.path.join(output_base, rel_img_path)
#             os.makedirs(os.path.dirname(save_path), exist_ok=True)
#             cv2.imwrite(save_path, image_with_bbox)
#             print(f"处理并保存图像：{save_path}")

# if __name__ == '__main__':
#     # 分别处理 train.pkl 和 val.pkl 文件
#     process_pkl_file("datasets/train.pkl")
#     process_pkl_file("datasets/val.pkl")






# import pickle
# import numpy as np

# # 加载 pkl 文件
# with open('datasets/train.pkl', 'rb') as f:
#     diving_data = pickle.load(f)
# print(diving_data[0:2])
# # 提取所有的 'Scores' 列表
# scores = [entry['Scores'] for entry in diving_data]

# # 将 scores 转换为 NumPy 数组，方便计算最大值和最小值
# scores_array = np.array(scores)

# # 计算最大值和最小值
# max_score = np.max(scores_array)
# min_score = np.min(scores_array)

# print(f"最大值: {max_score}")
# print(f"最小值: {min_score}")





# import os
# import re

# def rename_files_in_folder(folder_path):
#     # 遍历文件夹中的所有文件
#     for filename in os.listdir(folder_path):
#         # 构建完整的文件路径
#         file_path = os.path.join(folder_path, filename)
        
#         # 如果是文件夹，则递归调用
#         if os.path.isdir(file_path):
#             rename_files_in_folder(file_path)
        
#         # 检查文件是否符合 'frame_XXXX.jpg' 格式
#         match = re.match(r'frame_(\d+)\.jpg', filename)
#         if match:
#             # 提取文件名中的数字部分
#             new_name = match.group(1) + '.jpg'
            
#             # 获取新文件路径
#             new_file = os.path.join(folder_path, new_name)
            
#             # 重命名文件
#             os.rename(file_path, new_file)
#             print(f'Renamed: {filename} -> {new_name}')

# # 指定根目录路径
# root_folder = 'datasets/videos/3.2/407C/72.00'  # 替换为你实际的文件夹路径

# # 调用函数来处理根目录及其所有子文件夹
# rename_files_in_folder(root_folder)
