import sys
import os

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QStackedWidget,
    QFrame,
)
from PySide6.QtCore import Qt

from loading_screen import LoadingScreen
from PySide6.QtGui import QIcon

from tasks.task1 import Task1Page
from tasks.task2 import Task2Page
from tasks.task3 import Task3Page
from tasks.task4 import Task4Page
from tasks.task5 import Task5Page
from tasks.task6 import Task6Page
from tasks.task7 import Task7Page
from tasks.task8 import Task8Page
from tasks.task9 import Task9Page
from tasks.task10 import Task10Page
from tasks.task11 import Task11Page


TASK_NAMES = {
    1: "1. Гаусс и простая итерация",
    2: "2. Прогонка и Зейдель",
    3: "3. Метод вращения",
    4: "4. Корни уравнения",
    5: "5. Лагранж и Ньютон",
    6: "6. Кубические сплайны",
    7: "7. МНК",
    8: "8. Производные",
    9: "9. Метод Симпсона",
    10: "10. Рунге-Кутта и Адамс",
    11: "11. Краевая задача",
}

def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path).replace("\\", "/")

class PlaceholderPage(QWidget):
    def __init__(self, number: int, title: str):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        label = QLabel(f"{title}\n\nЭто задание подключим следующим шагом.")
        label.setAlignment(Qt.AlignCenter)
        label.setObjectName("PlaceholderLabel")

        layout.addWidget(label)

class WelcomePage(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("Численные методы")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            color: #ff008f;
            font-size: 42px;
            font-weight: 900;
            background: transparent;
        """)

        subtitle = QLabel("Выберите задание слева")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("""
            color: white;
            font-size: 26px;
            font-weight: 700;
            background: transparent;
            margin-top: 12px;
        """)

        layout.addWidget(title)
        layout.addWidget(subtitle)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Численные методы: Маев - Вариант 8")
        self.setWindowIcon(QIcon(resource_path("assets/app_icon.ico")))
        self.resize(1500, 900)

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(14, 18, 14, 18)
        sidebar_layout.setSpacing(8)

        self.logo = QLabel("Численные\nметоды")
        self.logo.setAlignment(Qt.AlignCenter)
        self.logo.setObjectName("LogoLabel")
        sidebar_layout.addWidget(self.logo)

        self.buttons = {}

        for number, title in TASK_NAMES.items():
            button = QPushButton(title)
            button.setObjectName("TaskButton")
            button.clicked.connect(lambda checked=False, n=number: self.open_task(n))
            sidebar_layout.addWidget(button)
            self.buttons[number] = button

        sidebar_layout.addStretch()

        self.stack = QStackedWidget()
        self.stack.setObjectName("ContentStack")

        self.welcome_page = WelcomePage()
        self.stack.addWidget(self.welcome_page)

        self.pages = {}

        for number, title in TASK_NAMES.items():
            if number == 1:
                page = Task1Page()
            elif number == 2:
                page = Task2Page()
            elif number == 3:
                page = Task3Page()
            elif number == 4:
                page = Task4Page()
            elif number == 5:
                page = Task5Page()
            elif number == 6:
                page = Task6Page()
            elif number == 7:
                page = Task7Page()
            elif number == 8:
                page = Task8Page()
            elif number == 9:
                page = Task9Page()
            elif number == 10:
                page = Task10Page()
            elif number == 11:
                page = Task11Page()
            else:
                page = PlaceholderPage(number, title)

            self.pages[number] = page
            self.stack.addWidget(page)

        root.addWidget(self.sidebar, 1)
        root.addWidget(self.stack, 4)

        self.setCentralWidget(central)

        self.apply_style()
        self.open_welcome()

    def open_welcome(self):
        self.stack.setCurrentWidget(self.welcome_page)

        for button in self.buttons.values():
            button.setProperty("active", False)
            button.style().unpolish(button)
            button.style().polish(button)

    def open_task(self, number: int):
        self.stack.setCurrentWidget(self.pages[number])

        for n, button in self.buttons.items():
            button.setProperty("active", n == number)
            button.style().unpolish(button)
            button.style().polish(button)

    def apply_style(self):
        plus_icon = resource_path("assets/plus.svg")
        minus_icon = resource_path("assets/minus.svg")

        qss = """
            QMainWindow {
                background-color: #000000;
            }

            QWidget {
                background-color: #0d0d0d;
                color: #ffffff;
                font-family: Segoe UI;
                font-size: 16px;
                font-weight: 600;
            }
            
            QLabel {
                background: transparent;
                color: white;
            }

            #Sidebar {
                background-color: #070707;
                border: 1px solid #222222;
                border-radius: 22px;
            }

            #LogoLabel {
                color: #ff008f;
                font-size: 28px;
                font-weight: bold;
                padding: 16px;
                letter-spacing: 1px;
            }

            #TaskButton {
                background-color: #111111;
                color: #ffffff;
                border: 1px solid #2a2a2a;
                border-radius: 14px;
                padding: 14px;
                text-align: left;

                font-family: Segoe UI;
                font-size: 17px;
                font-weight: bold;
                letter-spacing: 0.3px;
            }

            #TaskButton:hover {
                background-color: #171717;
                border: 1px solid #e04c91;
                color: #e04c91;
            }

            #TaskButton[active="true"] {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff008f,
                    stop:1 #e04c91
                );
                color: white;
                border: 1px solid #ff66c4;
            }

            #ContentStack {
                background-color: #0d0d0d;
                border: 1px solid #222222;
                border-radius: 22px;
            }

            #PlaceholderLabel {
                color: #ffffff;
                font-size: 28px;
                font-weight: bold;
            }

            QLabel#PageTitle {
                color: #ff008f;
                font-size: 30px;
                font-weight: bold;
                padding-bottom: 10px;
            }

            QLabel#BlockTitle {
                color: #ffffff;
                font-size: 21px;
                font-weight: bold;
                padding-top: 14px;
                padding-bottom: 4px;
            }

            QLabel#SmallHint {
                color: #bfbfbf;
                font-size: 13px;
            }

            QLabel#ConditionText {
                background-color: #111111;
                color: #ffffff;
                border: 1px solid #252525;
                border-radius: 18px;
                padding: 18px;
                font-size: 17px;
                line-height: 140%;
            }

            QLabel#FormulaText {
                background-color: #111111;
                color: #ffffff;
                border: 1px solid #252525;
                border-radius: 18px;
                padding: 18px;
                font-size: 18px;
            }

            QLabel#ResultCard {
                background-color: #111111;
                color: #ffffff;
                border: 1px solid #252525;
                border-radius: 20px;
                padding: 18px;
                font-size: 17px;
                font-weight: 600;
                line-height: 150%;
            }

            QFrame#Panel {
                background-color: #111111;
                border: 1px solid #252525;
                border-radius: 20px;
            }

            QFrame#GraphPanel {
                background-color: #111111;
                border: 1px solid #252525;
                border-radius: 20px;
            }

            QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color: #111111;
                color: #ffffff;
                border: 1px solid #383838;
                border-radius: 10px;
                padding: 8px;
                font-size: 18px;
                font-weight: 700;
                selection-background-color: #666666;
            }
            
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #ffffff;
            }
            
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 26px;
                background-color: #171717;
                border-left: 1px solid #383838;
                border-top-right-radius: 8px;
            }
            
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 26px;
                background-color: #171717;
                border-left: 1px solid #383838;
                border-bottom-right-radius: 8px;
            }
            
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #383838;
            }
            
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                image: url("__PLUS_ICON__");
                width: 12px;
                height: 12px;
            }
            
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                image: url("__MINUS_ICON__");
                width: 12px;
                height: 12px;
            }
            

            
            QPushButton#ActionButton {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff008f,
                    stop:1 #e04c91
                );
                color: white;
                border: none;
                border-radius: 14px;
                padding: 12px 18px;
                font-size: 16px;
                font-weight: bold;
            }

            QPushButton#ActionButton:hover {
                background-color: #ff33a6;
            }

            QPushButton#SecondaryButton {
                background-color: #181818;
                color: white;
                border: 1px solid #383838;
                border-radius: 14px;
                padding: 12px 18px;
                font-size: 15px;
                font-weight: bold;
            }

            QPushButton#SecondaryButton:hover {
                border: 1px solid #e04c91;
                color: #e04c91;
            }

            QTextBrowser, QTextEdit {
                background-color: #111111;
                color: #ffffff;
                border: 1px solid #252525;
                border-radius: 16px;
                padding: 12px;
                font-size: 16px;
            }

            QTableWidget {
                background-color: #111111;
                alternate-background-color: #171717;
                color: white;
                gridline-color: #383838;
                border: 1px solid #252525;
                border-radius: 14px;
                selection-background-color: #e04c91;
                selection-color: white;
            }

            QTableWidget::item {
                padding: 7px;
                border-bottom: 1px solid #222222;
            }

            QHeaderView::section {
                background-color: #383838;
                color: white;
                font-weight: bold;
                padding: 8px;
                border: 1px solid #2a2a2a;
            }

            QScrollArea {
                border: none;
                background-color: transparent;
            }

            QScrollBar:vertical {
                background: #383838;
                width: 8px;
                margin: 8px 2px 8px 2px;
                border-radius: 4px;
            }
            
            QScrollBar::handle:vertical {
                background: #e04c91;
                min-height: 36px;
                border-radius: 4px;
            }
            
            QScrollBar::handle:vertical:hover {
                background: #ff008f;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
                background: none;
            }

            QScrollBar:horizontal {
                background: #383838;
                height: 12px;
                margin: 0px 4px 0px 4px;
                border-radius: 6px;
            }

            QScrollBar::handle:horizontal {
                background: #e04c91;
                min-width: 28px;
                border-radius: 6px;
            }

            QScrollBar::handle:horizontal:hover {
                background: #ff008f;
            }

            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                width: 0px;
                background: none;
            }
        """

        qss = qss.replace("__PLUS_ICON__", plus_icon)
        qss = qss.replace("__MINUS_ICON__", minus_icon)

        self.setStyleSheet(qss)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setWindowIcon(QIcon(resource_path("assets/app_icon.ico")))
    app.setApplicationName("Численные методы: Маев - Вариант 8")

    loading = LoadingScreen()

    screen = app.primaryScreen().availableGeometry()
    x = screen.center().x() - loading.width() // 2
    y = screen.center().y() - loading.height() // 2
    loading.move(x, y)

    loading.show()
    app.processEvents()

    loading.set_progress(15, "Загрузка интерфейса...")
    app.processEvents()

    window = MainWindow()

    loading.set_progress(70, "Подготовка заданий...")
    app.processEvents()

    loading.set_progress(100, "Готово")
    app.processEvents()

    window.show()
    loading.close()

    sys.exit(app.exec())