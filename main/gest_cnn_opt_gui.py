from PyQt5.QtWidgets import QApplication, QMainWindow,QVBoxLayout, QLabel, QWidget, QHBoxLayout, QScrollArea, QPushButton, QGridLayout, QLineEdit, QTextEdit
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import os
import tensorflow as tf
import joblib
import numpy as np
from collections import deque
from threading import Lock
import myo
import threading
import sys
import pyautogui

# pyautogui 설정
pyautogui.PAUSE = 0.05

# Step 1: 손동작과 레이블 매핑
gesture_to_label = {
    'back': 0, 'space': 1, 'neutral': 2, 'ja1': 3, 'mo1': 4,
    'ja2': 5, 'mo2': 6, 'ja3': 7, 'ja4': 8, 'ja5': 9, 'ja6': 10, 'mo3':11
}

label_to_gesture = {v: k for k, v in gesture_to_label.items()}


# 손동작과 한글 자음/모음 매핑
gesture_to_char = {
    0: 'back', 1: 'space', 2: 'neutral', 3: 'ja1', 4: 'mo1',
    5: 'ja2', 6: 'mo2', 7: 'ja3', 8: 'ja4', 9: 'ja5', 10: 'ja6', 11: 'mo3'
}

# 자음/모음 변환 규칙
char_transitions = {
    'ja1': ['r', 'z', 'R'],  # ㄱ, ㅋ, ㄲ
    'ja2': ['s', 'f', 'd'],  # ㄴ, ㄹ, ㅇ
    'ja3': ['e', 'x', 'E'],  # ㄷ, ㅌ, ㄸ
    'ja4': ['q', 'v', 'Q'],  # ㅂ, ㅍ, ㅃ
    'ja5': ['t', 'g', 'a'],  # ㅅ, ㅎ, ㅁ
    'ja6': ['w', 'c', 'W'],  # ㅈ, ㅊ, ㅉ
    'mo1': ['l', 'j', 'k', 'u', 'i'],  # ㅣ, ㅓ, ㅏ, ㅕ, ㅑ
    'mo2': ['m', 'h', 'n', 'y', 'b'],  # ㅡ, ㅗ, ㅜ, ㅛ, ㅠ
    'mo3': ['p', 'o', 'P', 'O']  # ㅔ, ㅐ, ㅖ, ㅒ
}

class EmgCollector(myo.DeviceListener):
    def __init__(self, frame_size, hop_length, time_steps, scaler_path):
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.time_steps = time_steps
        self.lock = Lock()
        self.emg_data_queue = deque(maxlen=frame_size)
        self.rms_data_queue = deque(maxlen=time_steps)
        self.count = 0

        # Load scaler for normalization
        if not scaler_path or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}.")
        self.scaler = joblib.load(scaler_path)

    def get_time_series_rms_data(self):
        with self.lock:
            if len(self.rms_data_queue) == self.time_steps:
                return np.array(self.rms_data_queue)
            return None

    def get_raw_emg_data(self):
        with self.lock:
            data = np.array(self.emg_data_queue)
            if data.ndim == 1:
                data = data.reshape(-1, 8)
            return data

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_emg(self, event):
        with self.lock:
            self.emg_data_queue.append(event.emg)
            self.count += 1
            if self.count >= self.hop_length and len(self.emg_data_queue) == self.frame_size:
                self.calculate_rms()
                self.count = 0

    def calculate_rms(self):
        emg_array = np.array(self.emg_data_queue)
        rms = np.sqrt(np.mean(np.square(emg_array), axis=0))
        normalized_rms = self.scaler.transform(rms.reshape(1, -1)).flatten()
        self.rms_data_queue.append(normalized_rms)

class CustomLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent  # Parent application reference

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Backspace:
            self.setText(self.text()[:-1])
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.handle_enter_key()
        else:
            super().keyPressEvent(event)

    def handle_enter_key(self):
        # Call parent app's enter_clicked method
        if self.parent_app:
            self.parent_app.enter_clicked()


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)

class GestureApp(QMainWindow):
    def __init__(self, emg_collector, model, assets_path):
        super().__init__()
        self.emg_collector = emg_collector
        self.model = model
        self.assets_path = assets_path

        self.previous_gesture = None
        self.same_gesture_count = 0  # 동일한 손동작 카운트
        self.required_same_count = 2  # 두 번 연속해야 인식됨
        self.neutral_state_count = 0
        self.neutral_threshold = 2  # 중립 상태를 몇 번 연속 인식해야 하는지
        self.neutral_mode_active = False  # 중립 모드 활성화 플래그

        # 추가된 char_count 초기화 (자음/모음 순환을 추적)
        self.char_count = {char: 0 for char in gesture_to_char.values() if char not in ['back', 'space', 'neutral']}
        self.last_input_char = None  # 마지막 입력된 문자
        self.last_char_type = None  # 마지막 문자 타입 (자음/모음)

        # Set up UI
        self.setWindowTitle("Real-Time Gesture Recognition")
        self.resize(2560, 1440)
        self.setMinimumSize(1200, 800)
        self.setStyleSheet("background-color: black;")

        # Main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Left: Scrollable Graph Container
        self.left_scroll_area = QScrollArea()
        self.left_scroll_area.setWidgetResizable(True)
        self.left_container = QWidget()
        self.left_scroll_area.setWidget(self.left_container)

        self.left_layout = QVBoxLayout(self.left_container)
        self.raw_graphs = [MplCanvas(self, width=4, height=1) for _ in range(8)]
        for graph in self.raw_graphs:
            self.left_layout.addWidget(graph)

        self.left_scroll_area.setFixedWidth(1000)
        self.main_layout.addWidget(self.left_scroll_area)

        # Center: RMS Graph and Image
        self.center_layout = QVBoxLayout()
        self.rms_canvas = MplCanvas(self, width=6, height=3)
        self.rms_canvas.setFixedWidth(600)
        self.center_layout.addWidget(self.rms_canvas, stretch=6)

        self.center_image = QLabel()
        self.center_image.setStyleSheet("border: 2px solid white;")
        self.center_image.setAlignment(Qt.AlignCenter)
        self.center_image.setFixedWidth(600)
        self.center_layout.addWidget(self.center_image, stretch=4)

        self.main_layout.addLayout(self.center_layout)

        # Right: Gesture Label and Button Grid
        self.right_layout = QVBoxLayout()

        # Gesture Label
        self.gesture_label = QLabel("Detected Gesture: None")
        self.gesture_label.setStyleSheet("""
            font-size: 24px; 
            color: white; 
            border: 2px solid #4caf50; 
            padding: 10px;
            background-color: #333;
            border-radius: 10px;
        """)
        self.gesture_label.setFixedSize(850, 100)
        self.gesture_label.setAlignment(Qt.AlignTop | Qt.AlignCenter)
        self.right_layout.addWidget(self.gesture_label)

        # Button Grid
        self.button_grid = QWidget()
        self.button_layout = QGridLayout(self.button_grid)
        self.buttons = []  # 버튼 저장 리스트

        button_texts = [
            ["ㅣㅓㅏㅕㅑ", "ㅔㅐㅖㅒ", "ㅡㅜㅗㅠㅛ"],
            ["ㄱㅋㄲ", "ㄴㄹㅇ", "ㄷㅌㄸ"],
            ["ㅂㅍㅃ", "ㅅㅎㅁ", "ㅈㅊㅉ"]
        ]

        for row, texts in enumerate(button_texts):
            for col, text in enumerate(texts):
                button = QPushButton(text)
                button.setFixedSize(150, 150)
                button.setStyleSheet("""
                    font-size: 24px;
                    color: black;
                    background-color: white;
                    border: 1px solid #aaa;
                    border-radius: 10px;
                """)
                self.button_layout.addWidget(button, row, col)
                self.buttons.append(button)  # 버튼 저장

        self.right_layout.addWidget(self.button_grid)

        #Chat Input Section
        self.chat_widget = QWidget()
        self.chat_layout = QHBoxLayout(self.chat_widget)

        # Input Field
        self.chat_input = CustomLineEdit()
        self.chat_input.setPlaceholderText("Type your message here...")
        self.chat_input.setStyleSheet("""
            font-size: 48px;
            color: white;
            padding: 5px;
            border: 2px solid #aaa;
            border-radius: 10px;
        """)
        self.chat_layout.addWidget(self.chat_input, stretch=4)

        # Backspace Button
        self.backspace_button = QPushButton("⌫")
        self.backspace_button.setFixedSize(50, 50)
        self.backspace_button.setStyleSheet("""
            font-size: 18px;
            background-color: #f44336;
            color: white;
            border: none;
            border-radius: 10px;
        """)
        self.backspace_button.clicked.connect(self.backspace_clicked)
        self.chat_layout.addWidget(self.backspace_button)

        # Enter Button
        self.enter_button = QPushButton("⏎")
        self.enter_button.setFixedSize(50, 50)
        self.enter_button.setStyleSheet("""
            font-size: 18px;
            background-color: #4caf50;
            color: white;
            border: none;
            border-radius: 10px;
        """)
        self.enter_button.clicked.connect(self.enter_clicked)
        self.chat_layout.addWidget(self.enter_button)

        self.right_layout.addWidget(self.chat_widget)

        # Chat Log Section
        self.chat_log = QTextEdit()
        self.chat_log.setReadOnly(True)
        self.chat_log.setStyleSheet("""
            font-size: 48px;
            background-color: #333;
            color: white;
            border: 2px solid #aaa;
            border-radius: 10px;
        """)
        self.right_layout.addWidget(self.chat_log, stretch=3)
    
        QTimer.singleShot(100, self.set_initial_focus)

        self.main_layout.addLayout(self.right_layout)

        # Gesture to Button Map
        self.gesture_to_button_map = {
            'mo1': 0, 'mo3': 1, 'mo2' : 2,
            'ja1': 3, 'ja2': 4, 'ja3': 5,  
            'ja4': 6, 'ja5': 7, 'ja6': 8  
        }

       # Dictionary to track active timers
        self.active_timers = {}

        # Timer for UI Updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(500)

    def set_initial_focus(self):
        self.chat_input.setFocus()

    def simulate_button_click(self, button_index):
        """
        Simulate button click effect
        """
        if 0 <= button_index < len(self.buttons):
            button = self.buttons[button_index]

            # Change button style to simulate click
            original_style = button.styleSheet()
            button.setStyleSheet("""
                font-size: 24px;
                color: white;
                background-color: #4caf50;  /* Highlight color */
                border: 2px solid #388e3c;
                border-radius: 10px;
            """)

            # Cancel any existing timer for the button
            if button in self.active_timers:
                self.active_timers[button].stop()

            # Create a new timer to restore the original style
            timer = QTimer()
            timer.setSingleShot(True)
            timer.timeout.connect(lambda: button.setStyleSheet(original_style))
            timer.start(200)  # 200ms delay to restore the style

            # Store the new timer for this button
            self.active_timers[button] = timer

    def backspace_clicked(self):
        current_text = self.chat_input.text()
        self.chat_input.setText(current_text[:-1])

    def enter_clicked(self):
        message = self.chat_input.text().strip()
        if message:
            self.chat_log.append(message)
            self.chat_input.clear()

    def process_character(self, char, char_count, reset=False):
        """
        자음/모음 순환 변환.
        """
        if reset:
            char_count[char] = 0  # 순환 초기화
        if char in char_transitions:
            transitions = char_transitions[char]
            next_index = char_count[char] % len(transitions)
            char_count[char] += 1
            return transitions[next_index]
        return char
            
    def type_character(self, new_char, last_input_char, last_char_type, current_char_type, neutral_mode=False):
        """
        PyAutoGUI를 사용해 글자 입력 및 교체.
        neutral_mode: 중립 상태 후 새 글자 입력 모드 여부.
        """
        if new_char == 'back':
            pyautogui.press('backspace')  # 백스페이스 처리
            return
        elif neutral_mode:
            pyautogui.write(new_char, interval=0.1)
        elif last_char_type == current_char_type:
            # 동일한 유형의 문자(자음/모음)가 연속 입력될 때 순환
            pyautogui.press('backspace')
            pyautogui.write(new_char, interval=0.1)
        else:
            # 새로운 입력 처리
            pyautogui.write(new_char, interval=0.1)


    def update_ui(self):
        # Update raw EMG graphs
        raw_data = self.emg_collector.get_raw_emg_data()
        if raw_data is not None:
            for i, graph in enumerate(self.raw_graphs):
                graph.axes.clear()
                graph.axes.plot(raw_data[:, i], color="blue")
                graph.axes.set_ylim([-128, 128])
                graph.draw()

        # Update RMS graph
        rms_data = self.emg_collector.get_time_series_rms_data()
        if rms_data is not None:
            self.rms_canvas.axes.clear()
            self.rms_canvas.axes.bar(range(8), rms_data[-1], color="black")
            self.rms_canvas.axes.set_ylim(0, 1)
            self.rms_canvas.draw()

        # Update gesture image and label
        gesture_idx = self.predict_hand_gesture(rms_data)
        
        if gesture_idx is not None:
            gesture_name = label_to_gesture.get(gesture_idx, "Unknown Gesture")

            if gesture_name == self.previous_gesture:
                self.same_gesture_count += 1
            else:
                self.same_gesture_count = 1


            if gesture_to_char.get(gesture_idx) == "neutral":
                self.neutral_state_count += 1
                if self.neutral_state_count >= self.neutral_threshold:
                    self.neutral_mode_active = True
            else:
                self.neutral_state_count = 0


            if self.same_gesture_count >= self.required_same_count:
                self.gesture_label.setText(f"Gesture Recognized: {gesture_name}")

                mapped_char = gesture_to_char.get(gesture_idx, None)


                if mapped_char in ["ja1", "ja2", "ja3", "ja4", "ja5", "ja6"]:
                    current_char_type = "ja"  # 자음
                    new_char = self.process_character(mapped_char, self.char_count)
                    self.type_character(new_char, self.last_input_char, self.last_char_type, current_char_type, self.neutral_mode_active)
                    self.last_input_char = new_char  # 마지막 입력 문자 갱신
                    self.last_char_type = current_char_type  # 문자 유형 갱신
                    self.neutral_mode_active = False  # 첫 입력 후 중립 모드 해제

                elif mapped_char in ["mo1", "mo2", "mo3"]:
                    current_char_type = "mo"  # 모음
                    new_char = self.process_character(mapped_char, self.char_count)
                    self.type_character(new_char, self.last_input_char, self.last_char_type, current_char_type, self.neutral_mode_active)
                    self.last_input_char = new_char  # 마지막 입력 문자 갱신
                    self.last_char_type = current_char_type  # 문자 유형 갱신
                    self.neutral_mode_active = False  # 첫 입력 후 중립 모드 해제
                                
                elif mapped_char == 'back':
                    self.type_character('back', None, self.last_char_type, None)
                    self.last_input_char = None  # 입력 초기화
                    self.last_char_type = None  # 문자 유형 초기화
                    self.neutral_mode_active = False  # 첫 입력 후 중립 모드 해제
                                
                elif mapped_char == 'space':
                    self.type_character(' ', None, self.last_char_type, None)
                    self.last_input_char = None  # 입력 초기화
                    self.last_char_type = None  # 문자 유형 초기화
                    self.neutral_mode_active = False  # 첫 입력 후 중립 모드 해제

                # 이미지 경로를 업데이트
                image_path = os.path.join(self.assets_path, f"{gesture_name}.png")
                if os.path.exists(image_path):
                    pixmap = QPixmap(image_path).scaled(500, 400, aspectRatioMode=1)
                    self.center_image.setPixmap(pixmap)
                else:
                    self.center_image.clear()

                # 버튼 클릭 시뮬레이션
                if gesture_name in self.gesture_to_button_map:
                    button_index = self.gesture_to_button_map[gesture_name]
                    self.simulate_button_click(button_index)
            
                self.same_gesture_count = 0

            self.previous_gesture = gesture_name

    def predict_hand_gesture(self, time_series_data):
        if time_series_data is None:
            return None
        input_data = time_series_data.reshape(1, time_series_data.shape[0], time_series_data.shape[1], 1)
        prediction = self.model.predict(input_data, verbose=0)
        confidence = np.max(prediction)
        if confidence < 0.90:
            return None
        return np.argmax(prediction)


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "cnn_model.h5")
    scaler_path = os.path.join(current_dir, "scaler.pkl")
    assets_path = os.path.join(current_dir, "../assets")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Model or scaler not found. Please train the model first.")
        return

    # Load model and initialize Myo
    model = tf.keras.models.load_model(model_path)
    myo.init()
    hub = myo.Hub()
    emg_collector = EmgCollector(frame_size=40, hop_length=20, time_steps=10, scaler_path=scaler_path)

    # Start PyQt application
    app = QApplication(sys.argv)
    gui = GestureApp(emg_collector, model, assets_path)

    def run_myo():
        try:
            hub.run_forever(emg_collector.on_event)
        except Exception as e:
            print(f"Myo run error: {e}")

    threading.Thread(target=run_myo, daemon=True).start()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()