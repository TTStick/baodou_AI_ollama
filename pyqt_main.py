import sys
import time
import ctypes
import os
import json
import platform
import re
from mac_app_utils import is_mac_app, get_app_resource_path, get_resource_file_path

# 仅在开发环境中设置Qt平台插件路径，打包时自动处理
if os.path.exists('venv/Lib/site-packages'):
    sys.path.append('venv/Lib/site-packages')

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                            QTextEdit, QPushButton, QLabel, QLineEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QCursor, QPainter, QPen, QColor, QBrush # Added painting imports

# Windows API常量
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x80000
WS_EX_TRANSPARENT = 0x20
WS_EX_TOOLWINDOW = 0x00000080
WDA_EXCLUDEFROMCAPTURE = 0x00000011

# 定义AI坐标监测常量
AI_COORDINATE_THRESHOLD = 500
WINDOW_MOVE_DISTANCE = 600

# 导入AI控制相关函数
import vl_model_test_doubao2
from vl_model_test_doubao2 import auto_control_computer, set_coordinate_callback

# 导入日志窗口模块
from log_window import init_log_window

# --- 默认 Ollama 配置 ---
DEFAULT_OLLAMA_CONFIG = {
    "api_config": {
        "api_key": "ollama",                  # 本地模型通常不需要真实Key
        "base_url": "http://127.0.0.1:11434/v1", # Ollama 默认地址
        "model_name": "qwen2.5vl:7b"   # 默认视觉模型，可根据需要修改
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

class CoordinateMarker(QWidget):
    """
    Visual marker to show where AI intends to click.
    Transparent, click-through, and always on top.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents) # Critical: lets clicks pass through
        self.setFixedSize(60, 60) # Size of the marker area
        
        # Hide timer
        self.hide_timer = QTimer(self)
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self.hide)

    def show_at(self, x, y):
        """Move the marker to x,y and show it"""
        # Center the widget on the coordinates
        # Note: Qt coordinates might need adjustment based on HighDPI, 
        # but usually map directly to logical screen coordinates used by pyautogui
        self.move(int(x - self.width()/2), int(y - self.height()/2))
        self.show()
        self.update()
        # Keep visible for 1.5 seconds then hide
        self.hide_timer.start(1500)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw target circles
        center_x = self.width() / 2
        center_y = self.height() / 2
        
        # Outer ring (Red, semi-transparent)
        painter.setPen(QPen(QColor(255, 0, 0, 200), 3))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(int(center_x - 20), int(center_y - 20), 40, 40)
        
        # Inner dot (Solid Red)
        painter.setBrush(QBrush(QColor(255, 0, 0, 255)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(int(center_x - 5), int(center_y - 5), 10, 10)
        
        # Crosshairs
        painter.setPen(QPen(QColor(255, 0, 0, 150), 2))
        painter.drawLine(int(center_x - 25), int(center_y), int(center_x + 25), int(center_y))
        painter.drawLine(int(center_x), int(center_y - 25), int(center_x), int(center_y + 25))


class AIWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    ai_coordinate = pyqtSignal(float, float)
    
    def __init__(self, user_content, parent=None):
        super().__init__(parent)
        self.user_content = user_content
        
    def run(self):
        try:
            def coordinate_callback(coords):
                self.ai_coordinate.emit(coords[0], coords[1])
            
            set_coordinate_callback(coordinate_callback)
            
            time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime())
            user_content2 = "当前时间为:"+time_str + "\n" + "用户任务为:"+self.user_content
            print(f"=============用户输入内容为:{user_content2}")
            result = auto_control_computer(user_content2)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class AIWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.ai_thread = None
        self.is_ai_controlling = False
        self.mouse_monitor_timer = None
        
        # Initialize the visual marker
        self.marker = CoordinateMarker()
        
        # 初始化配置（如果不存在则创建）
        self.init_config()
        
        self.initUI()
        
    def init_config(self):
        """检查并初始化 config.json"""
        try:
            config_path = get_resource_file_path('config.json')
            if not os.path.exists(config_path):
                print("未检测到配置文件，正在创建默认 Ollama 配置...")
                # 确保目录存在
                os.makedirs(os.path.dirname(config_path) if os.path.dirname(config_path) else '.', exist_ok=True)
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(DEFAULT_OLLAMA_CONFIG, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"初始化配置失败: {e}")

    def initUI(self):
        self.setWindowTitle('本地AI助手 (Ollama)')
        self.setGeometry(100, 100, 520, 320)
        
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Window | 
                          Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        self.setWindowOpacity(0.9)
        self.show()
        
        # --- 操作系统特定设置 (保持原样) ---
        current_os = platform.system()
        if current_os == "Windows":
            hwnd = self.winId().__int__()
            user32 = ctypes.windll.user32
            current_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            new_style = current_style | WS_EX_LAYERED
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
            user32.SetLayeredWindowAttributes(hwnd, 0, int(255 * 0.9), 0x00000002)
            try:
                user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
            except Exception as e:
                print(f"防截图设置出错: {e}")
        elif current_os == "Darwin":
             print("macOS系统：窗口透明度设置已应用")
        
        # --- UI 布局 ---
        main_layout = QVBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(15, 10, 15, 10)
        
        # 标题标签
        title_label = QLabel('本地 AgentAI 控制台')
        title_font = QFont('Microsoft YaHei', 16, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: #2e7d32; /* 改为绿色系，代表本地运行 */
                padding: 8px;
                margin-bottom: 8px;
                background-color: #e8f5e9;
                border-radius: 10px;
                font-weight: bold;
                border: 2px solid #a5d6a7;
            }
        """)
        main_layout.addWidget(title_label)
        
        # 模型设置区域 (替代了原来的 API Key 区域)
        api_layout = QHBoxLayout()
        api_layout.setSpacing(8)
        
        # 模型名称标签
        model_label = QLabel("模型名称:")
        model_label.setFont(QFont('Microsoft YaHei', 10))
        model_label.setStyleSheet("color: #2e7d32; font-weight: bold;")
        api_layout.addWidget(model_label)

        # 模型名称输入框
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText('例如: llama3.2-vision, qwen2.5-vl')
        self.model_input.textChanged.connect(self.save_model_config) # 实时保存
        input_font = QFont('Microsoft YaHei', 11)
        self.model_input.setFont(input_font)
        self.model_input.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                border: 2px solid #a5d6a7;
                border-radius: 8px;
                background-color: #f1f8e9;
                color: #1b5e20;
                font-size: 11pt;
            }
            QLineEdit:focus {
                border-color: #4caf50;
                background-color: #ffffff;
                outline: none;
            }
        """)
        api_layout.addWidget(self.model_input)
        
        main_layout.addLayout(api_layout)
        
        # 任务输入框
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText('请输入您的需求，本地 AI 将帮您执行...')
        self.input_text.setFixedHeight(120)
        self.input_text.setFont(input_font)
        self.input_text.setStyleSheet("""
            QTextEdit {
                padding: 12px;
                border: 2px solid #a5d6a7;
                border-radius: 10px;
                background-color: #f1f8e9;
                color: #1b5e20;
                font-size: 11pt;
                line-height: 1.5;
            }
            QTextEdit:focus {
                border-color: #4caf50;
                background-color: #ffffff;
                outline: none;
            }
        """)
        main_layout.addWidget(self.input_text)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        
        # 开始按钮
        self.upload_btn = QPushButton('🚀 开始执行')
        self.upload_btn.clicked.connect(self.start_ai)
        upload_font = QFont('Microsoft YaHei', 12, QFont.Bold)
        self.upload_btn.setFont(upload_font)
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #66bb6a, stop:1 #2e7d32);
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                font-weight: bold;
                min-height: 40px;
                font-size: 12pt;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #81c784, stop:1 #388e3c);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2e7d32, stop:1 #1b5e20);
            }
            QPushButton:disabled {
                background: #cfd8dc;
                color: #90a4ae;
            }
        """)
        button_layout.addWidget(self.upload_btn)
        
        # 停止按钮
        self.stop_btn = QPushButton('⏹ 停止执行')
        self.stop_btn.clicked.connect(self.stop_ai)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setFont(upload_font)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ef5350, stop:1 #c62828);
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                font-weight: bold;
                min-height: 40px;
                font-size: 12pt;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e57373, stop:1 #d32f2f);
            }
            QPushButton:disabled {
                background: #cfd8dc;
                color: #90a4ae;
            }
        """)
        button_layout.addWidget(self.stop_btn)
        
        main_layout.addLayout(button_layout)
        
        # 状态标签
        self.status_label = QLabel('🎯 本地模型准备就绪')
        self.status_label.setAlignment(Qt.AlignCenter)
        status_font = QFont('Microsoft YaHei', 11, QFont.Bold)
        self.status_label.setFont(status_font)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #2e7d32;
                background-color: #e8f5e9;
                padding: 12px;
                border-radius: 8px;
                margin-top: 8px;
                font-weight: bold;
                border: 2px solid #a5d6a7;
                font-size: 11pt;
            }
        """)
        main_layout.addWidget(self.status_label)
        
        # 加载现有配置中的模型名
        self.load_model_config()
        
        self.setLayout(main_layout)
        self.setStyleSheet("""
            QWidget {
                background-color: #f1f8e9;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f9fbe7, stop:1 #eff7e9);
            }
        """)
    
    def load_model_config(self):
        """加载模型名称"""
        try:
            config_path = get_resource_file_path('config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                model_name = config.get('api_config', {}).get('model_name', '')
                self.model_input.setText(model_name)
        except Exception as e:
            print(f"加载配置失败: {e}")
    
    def save_model_config(self, text):
        """保存模型名称到配置文件"""
        try:
            config_path = get_resource_file_path('config.json')
            config = {}
            
            # 读取现有配置
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                config = DEFAULT_OLLAMA_CONFIG.copy()
            
            # 更新模型名称
            if 'api_config' not in config:
                config['api_config'] = {}
            config['api_config']['model_name'] = text
            # 确保 Base URL 也是本地的
            if 'base_url' not in config['api_config'] or not config['api_config']['base_url']:
                 config['api_config']['base_url'] = "http://localhost:11434/v1"
            config['api_config']['api_key'] = "ollama" # 强制写为 ollama

            # 保存
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
                
        except Exception as e:
            print(f"保存配置失败: {e}")
    
    def start_ai(self):
        model_name = self.model_input.text().strip()
        user_input = self.input_text.toPlainText().strip()
        
        if not model_name or not user_input:
            self.status_label.setText('⚠️ 请填写模型名称和需求')
            return
        
        # 确保配置已保存
        self.save_model_config(model_name)
        
        # 重置退出标志
        vl_model_test_doubao2.should_exit = False
        
        self.status_label.setText(f'🤖 AI ({model_name}) 正在执行中...')
        self.upload_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.model_input.setEnabled(False) # 运行时锁定模型修改
        
        self.is_ai_controlling = True
        
        self.ai_thread = AIWorker(user_input)
        self.ai_thread.finished.connect(self.ai_finished)
        self.ai_thread.error.connect(self.ai_error)
        self.ai_thread.start()
        
        self.ai_thread.ai_coordinate.connect(self.handle_ai_coordinate)
        
    def stop_ai(self):
        try:
            vl_model_test_doubao2.stop_client()
        except Exception as e:
            print(f"停止失败: {e}")
            vl_model_test_doubao2.should_exit = True
        
        self.status_label.setText('⏹️ 正在停止...')
        
    def ai_finished(self, result):
        self.status_label.setText('✅ 执行完成')
        self.reset_ui_state()
        
    def ai_error(self, error):
        self.status_label.setText(f'❌ 错误: {error}')
        self.reset_ui_state()
        
    def reset_ui_state(self):
        """重置UI控件状态"""
        self.upload_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.model_input.setEnabled(True)
        vl_model_test_doubao2.should_exit = False
        self.is_ai_controlling = False

    def handle_ai_coordinate(self, x, y):
        """
        Handle coordinates received from AI Worker.
        1. Draw the marker on screen.
        2. Move the main window away if it obstructs.
        """
        if not self.is_ai_controlling: return

        # === 1. Draw Coordinate Marker ===
        self.marker.show_at(x, y)

        # === 2. Avoid Obstruction Logic ===
        window_rect = self.geometry()
        center = window_rect.center()
        distance = ((x - center.x()) ** 2 + (y - center.y()) ** 2) ** 0.5
        
        if distance < AI_COORDINATE_THRESHOLD:
            ai_pos = QPoint(int(x), int(y))
            self.move_window_away(ai_pos, window_rect)
    
    def move_window_away(self, ai_pos, window_rect):
        screen = QApplication.desktop().availableGeometry()
        win_w, win_h = window_rect.width(), window_rect.height()
        new_x, new_y = window_rect.center().x(), window_rect.center().y()
        
        if (new_x + WINDOW_MOVE_DISTANCE) < (screen.width() - win_w/2):
            new_x += WINDOW_MOVE_DISTANCE
        else:
            new_x -= WINDOW_MOVE_DISTANCE
            
        if (new_y + WINDOW_MOVE_DISTANCE) < (screen.height() - win_h/2):
            new_y += WINDOW_MOVE_DISTANCE
        else:
            new_y -= WINDOW_MOVE_DISTANCE
            
        new_x = max(0, min(new_x - win_w // 2, screen.width() - win_w))
        new_y = max(0, min(new_y - win_h // 2, screen.height() - win_h))
        self.move(new_x, new_y)
    
    def closeEvent(self, event):
        self.stop_ai()
        if self.marker:
            self.marker.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    log_window = init_log_window()
    window = AIWindow()
    sys.exit(app.exec_())