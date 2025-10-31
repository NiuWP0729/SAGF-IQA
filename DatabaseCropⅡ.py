import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

def spectral_residual_saliency(image):
    """使用频谱残差法计算显著性图"""
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算傅里叶变换
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    # 计算对数幅度谱
    magnitude_spectrum = np.log(np.abs(fshift))
    # 计算谱残差
    spectral_residual = magnitude_spectrum - cv2.boxFilter(magnitude_spectrum, -1, (3, 3))
    # 逆傅里叶变换
    saliency_map = np.abs(np.fft.ifft2(np.exp(spectral_residual + 1j * np.angle(fshift))))
    # 归一化
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
    return saliency_map

def get_center_point(saliency_map, threshold=0.5):
    """计算显著性图的中心点"""
    # 二值化
    binary_map = (saliency_map > threshold).astype(np.uint8)

    # 计算轮廓
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None  # 没有找到轮廓

    # 找到最大轮廓
    largest_contour = max(contours, key=lambda x: cv2.contourArea(x))

    # 计算轮廓的中心点
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    else:
        return None

def crop_image(image, center, size=224):
    """以中心点裁剪图像"""
    h, w = image.shape[:2]

    if center is None:
        # 如果没有显著性中心点，使用图像中心点
        center = (w // 2, h // 2)

    # 计算裁剪区域
    x1 = max(0, center[0] - size // 2)
    y1 = max(0, center[1] - size // 2)
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)

    # 调整裁剪区域确保为224x224
    if x2 - x1 < size:
        diff = size - (x2 - x1)
        x1 = max(0, x1 - diff // 2)
        x2 = x1 + size
    if y2 - y1 < size:
        diff = size - (y2 - y1)
        y1 = max(0, y1 - diff // 2)
        y2 = y1 + size

    # 裁剪图像
    cropped = image[y1:y2, x1:x2]

    # 如果尺寸不足，填充边界
    if cropped.shape[0] < size or cropped.shape[1] < size:
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        h_crop, w_crop = cropped.shape[:2]
        padded[:h_crop, :w_crop] = cropped
        return padded
    else:
        return cropped

def process_image(image_path, output_dir):
    """处理单张图像"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    print(f"成功读取图像: {image_path}")  # 添加调试信息

    # 调整图像大小以加速处理（可选）
    h, w = image.shape[:2]
    if max(h, w) > 800:
        scale = 800 / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    # 计算显著性（使用频谱残差法）
    saliency_map = spectral_residual_saliency(image)

    # 计算显著性图的中心点
    center = get_center_point(saliency_map)

    # 裁剪图像
    cropped = crop_image(image, center)

    # 保存裁剪后的图像
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, cropped)

    return output_path

def process_dataset(input_dir, output_dir):
    """处理整个数据集"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图像路径
    image_paths = glob(os.path.join(input_dir, '*.JPG')) + \
                  glob(os.path.join(input_dir, '*.jpeg')) + \
                  glob(os.path.join(input_dir, '*.png')) + \
                  glob(os.path.join(input_dir, '*.bmp'))

    # 处理每张图像
    for image_path in tqdm(image_paths, desc="处理图像"):
        process_image(image_path, output_dir)

    print(f"完成! 裁剪后的图像已保存到: {output_dir}")

# 使用示例
if __name__ == "__main__":
    input_dir = "E:/IQA/koniq10k_1024x768/1024x768"  # 替换为你的数据集路径
    output_dir = "E:/IQA/koniq10k_1024x768/Cropped_Images2"  # 替换为输出路径
    process_dataset(input_dir, output_dir)