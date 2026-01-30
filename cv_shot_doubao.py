import cv2
import numpy as np
import pyautogui
import os
import time
import platform
from mac_app_utils import is_mac_app, get_resource_file_path, get_default_imgs_path

def capture_screen_and_save(save_path=None, optimize_for_speed=True, max_png=1280):
    """
    截屏并保存，同时计算Retina缩放因子和图片压缩比例
    返回: (success, resize_scale, retina_scales)
    """
    if save_path is None:
        default_imgs_path = get_default_imgs_path()
        save_path = os.path.join(default_imgs_path, "screen.png")
    elif not os.path.isabs(save_path) and is_mac_app():
        save_path = get_resource_file_path(save_path)
        
    output_dir = os.path.dirname(save_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # 1. 获取屏幕逻辑尺寸 (即鼠标操作的坐标系)
        screen_width, screen_height = pyautogui.size()
        
        # 2. 获取物理截图 (Retina屏上像素通常是逻辑尺寸的2倍)
        screenshot = pyautogui.screenshot()
        screenshot_np = np.array(screenshot)
        screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
        
        # 3. 计算物理分辨率到逻辑分辨率的倍率 (Retina Factor)
        physical_h, physical_w, _ = screenshot_bgr.shape
        retina_scale_x = physical_w / screen_width
        retina_scale_y = physical_h / screen_height
        retina_scales = (retina_scale_x, retina_scale_y)

        # 4. 图片压缩处理 (为了适应 Vision 模型输入限制)
        resize_scale = 1.0
        if optimize_for_speed:
            height, width, _ = screenshot_bgr.shape
            max_edge = max(height, width)
            if max_edge > max_png:
                resize_scale = max_png / max_edge
                screenshot_bgr = cv2.resize(screenshot_bgr, None, fx=resize_scale, fy=resize_scale)
            
        # 5. 保存处理后的图片
        # 使用低压缩级别(1)以提高保存速度
        save_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 1] if optimize_for_speed else []
        success = cv2.imwrite(save_path, screenshot_bgr, save_params)
        
        return success, resize_scale, retina_scales

    except Exception as e:
        print(f"截屏错误: {e}")
        return False, 1.0, (1.0, 1.0)

def draw_grid_on_image(image_path, output_path=None):
    """
    (新增) 在图片上绘制 10x10 网格，辅助 7B 模型定位
    """
    if output_path is None:
        output_path = image_path
        
    try:
        img = cv2.imread(image_path)
        if img is None: return False
        
        h, w, _ = img.shape
        overlay = img.copy()
        
        # 网格设置: 绿色细线
        grid_color = (0, 255, 0) 
        step_x = w / 10
        step_y = h / 10
        
        # 绘制网格线
        for i in range(1, 10):
            x = int(i * step_x)
            cv2.line(overlay, (x, 0), (x, h), grid_color, 1)
            y = int(i * step_y)
            cv2.line(overlay, (0, y), (w, y), grid_color, 1)
            
        # 融合图片 (0.3 透明度，避免遮挡文字)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        
        cv2.imwrite(output_path, img)
        return True
    except Exception as e:
        print(f"绘制网格失败: {e}")
        return False

def mark_coordinate_on_image(coordinates, input_path=None, output_path=None, point_radius=10, point_color=(0, 0, 255), thickness=-1):
    """
    在图片上标记指定坐标点
    """
    default_imgs_path = get_default_imgs_path()
    
    if input_path is None:
        input_path = os.path.join(default_imgs_path, "screen.png")
    elif not os.path.isabs(input_path) and is_mac_app():
        input_path = get_resource_file_path(input_path)
    
    if output_path is None:
        output_path = os.path.join(default_imgs_path, "screen_label.png")
    elif not os.path.isabs(output_path) and is_mac_app():
        output_path = get_resource_file_path(output_path)
        
    try:
        if not os.path.exists(input_path):
            return False
        
        image = cv2.imread(input_path)
        if image is None:
            return False
        
        img_height, img_width = image.shape[:2]
        points_to_mark = []
        
        if isinstance(coordinates[0], (list, tuple)):
            for coord in coordinates:
                if len(coord) == 2:
                    x, y = int(coord[0]), int(coord[1])
                    if 0 <= x < img_width and 0 <= y < img_height:
                        points_to_mark.append((x, y))
        else:
            if len(coordinates) == 2:
                x, y = int(coordinates[0]), int(coordinates[1])
                if 0 <= x < img_width and 0 <= y < img_height:
                    points_to_mark.append((x, y))
        
        if not points_to_mark:
            return False
        
        for i, (x, y) in enumerate(points_to_mark):
            cv2.circle(image, (x, y), point_radius, point_color, thickness)
            
            # 添加坐标文本
            text = f"({x}, {y})" if len(points_to_mark) == 1 else f"P{i+1} ({x}, {y})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            offset = i * 40
            text_position = (max(0, x - 30 + offset), max(20, y - 20))
            cv2.putText(image, text, text_position, font, 1, (0, 0, 255), 2)
        
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        cv2.imwrite(output_path, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
        return True
            
    except Exception as e:
        print(f"标记坐标错误: {e}")
        return False

def map_coordinates(x, y, resize_scale, retina_scales, img_width, img_height):
    """
    (核心修改) 将模型输出的归一化坐标映射回屏幕逻辑坐标
    参数:
        x, y: 模型输出的归一化坐标 (0-1000)
        resize_scale: 图片压缩缩放比
        retina_scales: (x_scale, y_scale) 物理/逻辑像素比
        img_width, img_height: 输入给模型的图片实际宽高
    """
    # 1. 归一化坐标 (0-1000) -> 压缩图像素坐标
    pixel_x = (x / 1000.0) * img_width
    pixel_y = (y / 1000.0) * img_height
    
    # 2. 压缩图像素 -> 原始截图(物理)像素
    original_pixel_x = pixel_x / resize_scale
    original_pixel_y = pixel_y / resize_scale
    
    # 3. 原始截图像素 -> 屏幕逻辑坐标 (pyautogui 使用的坐标)
    screen_x = original_pixel_x / retina_scales[0]
    screen_y = original_pixel_y / retina_scales[1]
    
    return int(screen_x), int(screen_y)