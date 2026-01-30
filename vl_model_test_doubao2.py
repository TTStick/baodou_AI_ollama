"""这个版本适配本地OLLAMA多模态模型 - 已针对Retina屏和精准点击优化"""

import os
import base64
import cv2
import numpy as np
from openai import OpenAI
import json
import re
# 核心修改：增加了 draw_grid_on_image 的导入
from cv_shot_doubao import mark_coordinate_on_image, capture_screen_and_save, map_coordinates, draw_grid_on_image
import time
import pyautogui
import pyperclip
import signal
from pydantic import BaseModel
import platform
from mac_app_utils import is_mac_app, get_app_resource_path, get_resource_file_path, get_default_imgs_path

# 全局退出标志
should_exit = False

# 全局回调函数，用于通知主窗口AI输出的坐标
coordinate_callback = None

# 全局客户端实例，用于中断API调用
_global_client = None

current_os = platform.system()

# 尝试导入日志窗口模块
try:
    from log_window import get_log_window
    LOG_WINDOW_AVAILABLE = True
except ImportError:
    LOG_WINDOW_AVAILABLE = False

# 全局信号处理器实例
_signal_handler = None

def get_signal_handler():
    """获取全局信号处理器实例"""
    global _signal_handler
    if _signal_handler is None and LOG_WINDOW_AVAILABLE:
        try:
            log_window = get_log_window()
            if log_window:
                _signal_handler = log_window.signal_handler
        except:
            pass
    return _signal_handler

# 初始化日志窗口函数
def init_log_if_available():
    """如果日志窗口可用，则初始化它"""
    if LOG_WINDOW_AVAILABLE:
        try:
            log_window = get_log_window()
            return log_window
        except:
            return None
    return None

# 日志打印函数，确保日志能显示在日志窗口中
def log_print(*args, **kwargs):
    """自定义打印函数，仅使用原始print函数"""
    import builtins
    original_print = builtins.print
    original_print(*args, **kwargs)

# 设置坐标回调函数
def set_coordinate_callback(callback):
    global coordinate_callback
    coordinate_callback = callback

# 停止客户端连接
def stop_client():
    global _global_client, should_exit
    should_exit = True
    log_print("已设置退出标志，等待API调用完成...")
    
    if _global_client is not None:
        try:
            log_print("正在关闭与AI服务器的连接...")
            _global_client.close()
            log_print("已关闭客户端连接")
        except Exception as e:
            log_print(f"关闭客户端连接时出错: {e}")
        finally:
            _global_client = None

# 信号处理函数
def signal_handler(sig, frame):
    global should_exit, _global_client
    log_print("\n\n收到中断信号 (Ctrl+C)，正在立即停止执行...")
    should_exit = True
    stop_client()
    import sys
    log_print("程序已停止")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# 加载配置文件
def load_config(config_path="config.json"):
    if is_mac_app():
        config_path = get_resource_file_path(config_path)
        log_print(f"检测到Mac App环境，使用资源包中的配置文件: {config_path}")
    
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            # log_print(f"成功加载配置文件: {config_path}")
            return config
        return None
    except Exception as e:
        log_print(f"加载配置文件失败: {e}")
        return None

# 加载配置
config = load_config()

# === 默认配置 ===
DEFAULT_CONFIG = {
    "api_config": {
        "api_key": "ollama",
        "base_url": "http://127.0.01:11434/v1",
        "model_name": "qwen2.5vl:7b"
    },
    "ai_config": {
        "enable_thinking": False,
        "thinking_type": "disabled",
        "vl_high_resolution_images": True
    },
    "execution_config": {
        "max_visual_model_iterations": 80,
        "default_max_iterations": 15
    },
    "screenshot_config": {
        "optimize_for_speed": True,
        "max_png": 1280,
        "input_path": "imgs/screen.png",
        "output_path": "imgs/screen_label.png"
    },
    "mouse_config": {
        "move_duration": 0.1,
        "failsafe": False
    }
}

# 使用配置文件或默认值
if config:
    API_CONFIG = config.get("api_config", DEFAULT_CONFIG["api_config"])
    AI_CONFIG = config.get("ai_config", DEFAULT_CONFIG["ai_config"])
    EXECUTION_CONFIG = config.get("execution_config", DEFAULT_CONFIG["execution_config"])
    SCREENSHOT_CONFIG = config.get("screenshot_config", DEFAULT_CONFIG["screenshot_config"])
    MOUSE_CONFIG = config.get("mouse_config", DEFAULT_CONFIG["mouse_config"])
else:
    API_CONFIG = DEFAULT_CONFIG["api_config"]
    AI_CONFIG = DEFAULT_CONFIG["ai_config"]
    EXECUTION_CONFIG = DEFAULT_CONFIG["execution_config"]
    SCREENSHOT_CONFIG = DEFAULT_CONFIG["screenshot_config"]
    MOUSE_CONFIG = DEFAULT_CONFIG["mouse_config"]

# Mac 路径处理
if is_mac_app():
    if "input_path" in SCREENSHOT_CONFIG and not os.path.isabs(SCREENSHOT_CONFIG["input_path"]):
        SCREENSHOT_CONFIG["input_path"] = get_resource_file_path(SCREENSHOT_CONFIG["input_path"])
    if "output_path" in SCREENSHOT_CONFIG and not os.path.isabs(SCREENSHOT_CONFIG["output_path"]):
        SCREENSHOT_CONFIG["output_path"] = get_resource_file_path(SCREENSHOT_CONFIG["output_path"])

pyautogui.FAILSAFE = MOUSE_CONFIG["failsafe"]

def read_local_image(image_path):
    """
    读取本地图片并转换为base64编码
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise Exception(f"无法读取图片: {image_path}")
        
        # OLLAMA通常不需要太高分辨率，适当压缩可以提高速度
        _, buffer = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        log_print(f"读取图片时出错: {e}")
        return None

class MathResponse(BaseModel):
    current_status: str
    solving_problem: str
    whether_completed: str
    element_info: str
    coordinates: list
    action: str
    type_information: str

def get_next_element(user_content):
    global _global_client, API_CONFIG
    
    # 每次重新加载配置
    config = load_config()
    if config:
        API_CONFIG = config.get("api_config", DEFAULT_CONFIG["api_config"])
    
    image_path = SCREENSHOT_CONFIG["input_path"]
    if not os.path.exists(image_path):
        log_print(f"错误：图片文件不存在 - {os.path.abspath(image_path)}")
        return
    
    image_data_url = read_local_image(image_path)
    if not image_data_url:
        return
    
    api_key = API_CONFIG["api_key"]
    if not api_key:
        api_key = "ollama" 
    
    log_print(f"\n正在调用模型: {API_CONFIG['model_name']}...")
    
    if should_exit:
        return None
    
    client = OpenAI(
        api_key=api_key,
        base_url=API_CONFIG["base_url"],
    )
    
    _global_client = client

    # 读取系统提示词
    txt_filename = "get_next_action_AI_doubao_mac.txt" if (current_os == "Darwin" or is_mac_app()) else "get_next_action_AI_doubao.txt"
    if is_mac_app():
        txt_path = get_resource_file_path(txt_filename)
    else:
        txt_path = txt_filename

    try:
        with open(txt_path, "r", encoding="utf-8") as file:
            system_content = file.read().strip()
    except Exception as e:
        log_print(f"读取提示词文件失败: {e}")
        return None

    try:
        completion = client.chat.completions.create(
            model=API_CONFIG["model_name"],
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": user_content}
                ]}
            ],
            temperature=0.1, 
        )
        
        raw_content = completion.choices[0].message.content
        log_print(f"AI 原始返回内容: {raw_content}")
        
        parsed_json = parse_json(raw_content)
        _global_client = None
        
        if parsed_json:
            return json.dumps(parsed_json, ensure_ascii=False)
        else:
            log_print("手动解析失败，无法处理 AI 返回的内容")
            return None

    except Exception as e:
        log_print(f"API调用出错: {e}")
        if "Connection refused" in str(e):
            log_print("请确认 OLLAMA 是否已启动")
        _global_client = None
        return None

def parse_json(json_str):
    """解析AI输出的JSON字符串"""
    try:
        if json_str.startswith('```json'):
            json_str = json_str[7:]
        if json_str.endswith('```'):
            json_str = json_str[:-3]
        json_str = json_str.strip()
        
        matches = re.findall(r'\{.*\}', json_str, re.DOTALL)
        
        if matches:
            candidate = max(matches, key=len)
            try:
                return json.loads(candidate)
            except:
                pass

        return json.loads(json_str)
            
    except json.JSONDecodeError as e:
        log_print(f"JSON解析错误: {e}")
        try:
            if json_str.count('{') > json_str.count('}'):
                json_str += '}'
            return json.loads(json_str)
        except:
            return None
    except Exception as e:
        log_print(f"解析过程中发生错误: {e}")
        return None

# 控制鼠标函数 (核心修改：接收 retina_scales)
def move_mouse_to_coordinates(coordinates, solving_problem, action, type_information, duration=MOUSE_CONFIG["move_duration"], scale=1, retina_scales=(1.0, 1.0)):
    # 验证坐标有效性的辅助函数
    def validate_coordinate(coord):
        if isinstance(coord, (int, float)):
            return max(-100000, min(100000, coord))
        return coord
    
    def fix_coordinates(coords):
        if isinstance(coords[0], list):
            if len(coords) == 1:
                return [validate_coordinate(coords[0][0]), validate_coordinate(coords[0][1])]
            else:
                return [
                    [validate_coordinate(coords[0][0]), validate_coordinate(coords[0][1])],
                    [validate_coordinate(coords[1][0]), validate_coordinate(coords[1][1])]
                ]
        else:
            return [validate_coordinate(coords[0]), validate_coordinate(coords[1])]
    
    coordinates = fix_coordinates(coordinates)
    
    if action == "page_loading":
        log_print("检测到页面正在加载，暂停3秒...")
        action_str = "检测到页面正在加载，暂停3秒..."+"\n"
        time.sleep(3.0)
        return action_str, None
    
    image_path = SCREENSHOT_CONFIG["input_path"]
    img = cv2.imread(image_path)
    if img is not None:
        img_height, img_width, _ = img.shape
    else:
        # 如果读取失败，给个默认值防止报错，但通常会导致映射错误
        img_width = 1920
        img_height = 1080
    
    action_str = ""
    
    if action == "hotkey":
        if type_information:
            keys = type_information.split()
            if current_os == "Darwin":
                keys = ["command" if key == "win" or key == "meta" else key for key in keys]
                keys = ["command" if key == "cmd" else key for key in keys]
                if len(keys) > 0:
                    pyautogui.keyDown(keys[0])
                    for key in keys[1:]:
                        pyautogui.press(key)
                    pyautogui.keyUp(keys[0])
            else:
                keys = ["win" if key == "meta" else key for key in keys]
                if len(keys) > 0:
                    pyautogui.keyDown(keys[0])
                    for key in keys[1:]:
                        pyautogui.press(key)
                    pyautogui.keyUp(keys[0])
            action_str = f"执行热键操作: {'+'.join(keys)}"+"\n"
        return action_str, None
    
    mapped_coordinates = []

    # 核心修改：在调用 map_coordinates 时传递 retina_scales
    if action == "drag" and isinstance(coordinates[0], list):
        start_x, start_y = coordinates[0]
        end_x, end_y = coordinates[1]
        start_x, start_y = map_coordinates(start_x, start_y, scale, retina_scales, img_width, img_height)
        end_x, end_y = map_coordinates(end_x, end_y, scale, retina_scales, img_width, img_height)
        
        pyautogui.moveTo(start_x, start_y, duration=duration)
        pyautogui.dragTo(end_x, end_y, duration=duration*10, button='left')
        action_str = f"拖拽: ({start_x}, {start_y}) -> ({end_x}, {end_y})\n"
        mapped_coordinates = [[start_x, start_y], [end_x, end_y]]
    else:
        x, y = coordinates
        # 核心修改：调用 map_coordinates
        x, y = map_coordinates(x, y, scale, retina_scales, img_width, img_height)
        
        if coordinate_callback:
            try:
                coordinate_callback((x, y))
            except:
                pass
        
        pyautogui.moveTo(x, y, duration=duration)
        mapped_coordinates = [x, y]
        
        scroll_range = 10 if current_os == "Darwin" else 500
        
        if action == "click":
            pyautogui.click()
            action_str += "已点击\n"
        elif action == "double_click":
            pyautogui.doubleClick()
            action_str += "已双击\n"
        elif action == "long_press":
            pyautogui.mouseDown()
            time.sleep(1)
            pyautogui.mouseUp()
            action_str += "已长按\n"
        elif action == "right_click":
            pyautogui.rightClick()
            action_str += "已右键\n"
        elif action == "scroll_up":
            pyautogui.scroll(scroll_range)
            action_str += "已向上滚动\n"
        elif action == "scroll_down":
            pyautogui.scroll(-scroll_range)
            action_str += "已向下滚动\n"
            
    time.sleep(0.2)
    if type_information != "" and action != "hotkey":
        pyperclip.copy(type_information)
        time.sleep(0.1)
        if action == "type_replace":
            pyautogui.click()
            if current_os == "Darwin":
                pyautogui.hotkey('command', 'a')
            else:
                pyautogui.hotkey('ctrl', 'a')
        
        if current_os == "Darwin":
            pyautogui.hotkey('command', 'v')
        else:
            pyautogui.hotkey('ctrl', 'v')
        
        time.sleep(0.5)
        pyautogui.press('enter')
        action_str += f"已输入: {type_information}\n"

    if solving_problem == "True":   
        pyautogui.moveTo(0, 0, duration=duration)
    time.sleep(1.5)

    return action_str, mapped_coordinates

# 自动控制函数
def auto_control_computer(user_content, max_visual_model_iterations=EXECUTION_CONFIG["default_max_iterations"]):
    global should_exit
    before_output = []
    current_status = "未完成"
    recent_coordinates = []
    same_coordinate_count = 0
    action_str = ""

    label_dir = SCREENSHOT_CONFIG["output_path"]
    if os.path.exists(os.path.dirname(label_dir)):
        try:
            output_dir = os.path.dirname(label_dir)
            for filename in os.listdir(output_dir):
                if filename.startswith("screen_label") and filename.endswith(".png"):
                    os.remove(os.path.join(output_dir, filename))
        except Exception as e:
            log_print(f"清理旧图片失败: {e}")

    for i in range(max_visual_model_iterations):
        if should_exit:
            return "程序已被用户中断"
        
        log_print(f"\n=================第 {i+1} 次循环===============")
        
        if i == 0:
            before_output = []
            before_content = ""
        else:
            if len(before_output) > 5:
                before_output.pop(0)
            before_output_str = "".join(before_output)
            before_content = "之前的AI输出操作为: "+before_output_str+"\n"+"之前已完成的操作为:"+action_str
        
        try:
            # 核心修改：接收3个返回值，包括 retina_scales
            success, resize_scale, retina_scales = capture_screen_and_save(
                save_path=SCREENSHOT_CONFIG["input_path"],
                optimize_for_speed=SCREENSHOT_CONFIG["optimize_for_speed"],
                max_png=SCREENSHOT_CONFIG["max_png"]
            )
            if not success:
                log_print("屏幕截图失败")
                continue

            # 核心修改：截图后立即绘制网格，覆盖原图供AI读取
            draw_grid_on_image(SCREENSHOT_CONFIG["input_path"])

            full_prompt = before_content + "\n" + user_content if before_content else user_content
            response_str = get_next_element(full_prompt)
            
            if should_exit:
                return "程序已被用户中断"

            if response_str:
                next_element = parse_json(response_str)
                if not next_element:
                    log_print("解析JSON失败，跳过本次循环")
                    continue

                current_status = next_element.get('current_status', '未知状态')
                solving_problem = next_element.get('solving_problem', 'False')
                whether_completed = next_element.get('whether_completed', 'difficult')
                element_info = next_element.get('element_info', '未知元素')
                coordinates = next_element.get('coordinates', [0, 0])
                action = next_element.get('action', '未知操作')
                type_information = next_element.get('type_information', '')

                before_output.append(json.dumps({
                    "action": action, 
                    "element": element_info,
                    "status": current_status
                }, ensure_ascii=False) + "; ")

                if whether_completed == "True":
                    log_print("任务标记为完成")
                    return current_status
                
                log_print(f"下一步操作: {action} -> {element_info} (原始坐标: {coordinates})")
                
                # 防抖动逻辑
                coordinates_match = False
                target_coords = coordinates[0] if isinstance(coordinates[0], list) else coordinates
                for coord in recent_coordinates:
                    # 简单的距离判断
                    if isinstance(coord, list) and len(coord)==2 and isinstance(target_coords, list) and len(target_coords)==2:
                         if abs(coord[0] - target_coords[0]) < 5 and abs(coord[1] - target_coords[1]) < 5:
                            coordinates_match = True
                            break
                
                if not coordinates_match:
                    recent_coordinates.append(target_coords)
                    same_coordinate_count = 1
                    if len(recent_coordinates) > 3:
                        recent_coordinates.pop(0)
                else:
                    same_coordinate_count += 1
                
                if same_coordinate_count >= 3:
                    log_print("检测到死循环，清空记忆")
                    before_output = []
                    same_coordinate_count = 0
                    recent_coordinates = []

                # 核心修改：传递 retina_scales
                action_str, mapped_coordinates = move_mouse_to_coordinates(
                    coordinates, solving_problem, action, type_information, 
                    scale=resize_scale, 
                    retina_scales=retina_scales
                )
                
            else:
                log_print("未收到模型响应")
        except Exception as e:
            log_print(f"循环发生错误: {e}")
            if "Connection refused" in str(e):
                log_print("严重错误：无法连接到本地 OLLAMA 服务")
                break
            time.sleep(2)

    return current_status

if __name__ == "__main__":
    log_print("=== OLLAMA 本地模式 (Retina适配版) ===")
    user_content = input("请输入您的需求：")
    auto_control_computer(user_content)