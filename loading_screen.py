from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
from PySide6.QtCore import Qt

class LoadingScreen(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setFixedSize(520, 260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(34, 30, 34, 30)
        layout.setSpacing(18)

        self.title = QLabel("Численные методы")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("""
            color: #ff008f;
            font-size: 34px;
            font-weight: 900;
            background: transparent;
            letter-spacing: 1px;
        """)

        self.subtitle = QLabel("Загрузка приложения...")
        self.subtitle.setAlignment(Qt.AlignCenter)
        self.subtitle.setStyleSheet("""
            color: white;
            font-size: 18px;
            font-weight: 700;
            background: transparent;
        """)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(14)
        self.progress.setStyleSheet("""
            QProgressBar {
                background-color: #171717;
                border: 1px solid #383838;
                border-radius: 7px;
            }

            QProgressBar::chunk {
                background-color: #ff008f;
                border-radius: 7px;
            }
        """)

        self.status = QLabel("Инициализация...")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet("""
            color: #d0d0d0;
            font-size: 14px;
            font-weight: 600;
            background: transparent;
        """)

        layout.addStretch()
        layout.addWidget(self.title)
        layout.addWidget(self.subtitle)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addStretch()

        self.setStyleSheet("""
            QWidget {
                background-color: #0d0d0d;
                border: 1px solid #383838;
                border-radius: 24px;
            }
        """)

    def set_progress(self, value, text):
        self.progress.setValue(value)
        self.status.setText(text)