import sys
import cv2
import numpy as np
import requests
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
    QLineEdit, QMessageBox, QSlider, QHBoxLayout, QTextEdit,
    QRadioButton, QButtonGroup
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
from datetime import datetime


class RTSPFaceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition")

        # ---- Title ----
        self.title_label = QLabel("Face Recognition App")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Arial", 16, QFont.Bold))

        # ---- Mode Selector ----
        self.recognize_radio = QRadioButton("Recognize")
        self.recognize_radio.setChecked(True)
        self.register_radio = QRadioButton("Register")
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.recognize_radio)
        self.mode_group.addButton(self.register_radio)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        mode_layout.addWidget(self.recognize_radio)
        mode_layout.addWidget(self.register_radio)

        # ---- RTSP Input & Button ----
        self.rtsp_input = QLineEdit()
        self.rtsp_input.setText("rtsp://Tapocam:Tapocam123@192.168.0.18:554/stream1")
        self.start_button = QPushButton("▶️ Start Stream")
        self.start_button.clicked.connect(self.start_stream)

        # ---- Frame Sending Control ----
        self.send_slider = QSlider(Qt.Horizontal)
        self.send_slider.setMinimum(1)
        self.send_slider.setMaximum(60)
        self.send_slider.setValue(30)
        self.slider_label = QLabel("Send every 30th frame")
        self.send_slider.valueChanged.connect(
            lambda: self.slider_label.setText(f"Send every {self.send_slider.value()}th frame")
        )

        # ---- Image Views ----
        self.live_label = QLabel("Live Stream")
        self.live_label.setFixedSize(640, 360)
        self.live_label.setStyleSheet("border: 2px solid black;")

        self.sent_label = QLabel("Last Sent Frame")
        self.sent_label.setFixedSize(640, 360)
        self.sent_label.setStyleSheet("border: 2px solid black;")

        # ---- Log Area ----
        self.status_console = QTextEdit()
        self.status_console.setReadOnly(True)
        self.status_console.setFixedHeight(120)
        self.status_console.setStyleSheet("background-color: #f5f5f5;")

        # ---- Layout Setup ----
        video_layout = QHBoxLayout()
        video_layout.addWidget(self.live_label)
        video_layout.addWidget(self.sent_label)

        controls = QVBoxLayout()
        controls.addLayout(mode_layout)
        controls.addWidget(self.rtsp_input)
        controls.addWidget(self.start_button)
        controls.addWidget(self.slider_label)
        controls.addWidget(self.send_slider)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.title_label)
        main_layout.addLayout(video_layout)
        main_layout.addLayout(controls)
        main_layout.addWidget(QLabel("📊 Progress Console"))
        main_layout.addWidget(self.status_console)

        self.setLayout(main_layout)

        # ---- Streaming Variables ----
        self.cap = None
        self.current_frame = None
        self.frame_counter = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_console.append(f"[{timestamp}] {message}")

    def start_stream(self):
        url = self.rtsp_input.text().strip()
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Unable to open RTSP stream.")
            self.log("❌ Failed to open RTSP stream.")
            return

        self.frame_counter = 0
        self.timer.start(30)
        self.log(f"✅ Stream started in {'Register' if self.register_radio.isChecked() else 'Recognize'} mode.")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.log("⚠️ Failed to grab frame.")
            return

        self.frame_counter += 1
        resized = cv2.resize(frame, (640, 360))
        self.show_frame_in_label(resized, self.live_label)

        if self.frame_counter % self.send_slider.value() == 0:
            self.send_frame_to_api(resized)

    def show_frame_in_label(self, frame, label):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qt_img))

    def send_frame_to_api(self, frame):
        mode = "register" if self.register_radio.isChecked() else "recognize"
        endpoint = f"http://zahangir.pythonanywhere.com/{mode}-frame"

        self.log(f"📤 Sending frame #{self.frame_counter} to `{mode}` API...")
        _, buffer = cv2.imencode('.jpg', frame)

        try:
            response = requests.post(
                endpoint,
                files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
            )
            if response.ok:
                result = response.json()
                self.log(f"✅ Success: {result}")
                self.show_frame_in_label(frame, self.sent_label)
            else:
                self.log(f"❌ Error: {response.status_code} - {response.text}")
        except Exception as e:
            self.log(f"❌ Exception: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RTSPFaceApp()
    window.show()
    sys.exit(app.exec_())
