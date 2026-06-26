import numpy as np

from PySide6.QtWidgets import (
    QWidget,
    QFrame,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSizePolicy,
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPainter, QPen, QColor, QPainterPath


class SectionPanel(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("Panel")
        self.setStyleSheet("""
            QFrame#Panel {
                background-color: #111111;
                border: 1px solid #252525;
                border-radius: 20px;
            }
        """)


class ResultCard(QFrame):
    def __init__(self, title: str, value: str = "—"):
        super().__init__()

        self.setObjectName("ResultCard")
        self.setStyleSheet("""
            QFrame#ResultCard {
                background-color: #111111;
                border: 1px solid #252525;
                border-radius: 18px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            color: white;
            font-size: 18px;
            font-weight: 700;
            background: transparent;
        """)

        self.value_label = QLabel(value)
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setStyleSheet("""
            color: white;
            font-size: 27px;
            font-weight: 900;
            letter-spacing: 1px;
            background: transparent;
        """)

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value):
        if isinstance(value, float):
            self.value_label.setText(f"{value:.6f}")
        else:
            self.value_label.setText(str(value))


class ParenthesisWidget(QWidget):
    def __init__(self, side="left", height=130):
        super().__init__()
        self.side = side
        self.setFixedSize(42, height)

    def set_bracket_height(self, height):
        self.setFixedSize(42, height)

    def sizeHint(self):
        return QSize(42, self.height())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        pen = QPen(QColor("#ffffff"))
        pen.setWidth(3)
        painter.setPen(pen)

        w = self.width()
        h = self.height()

        top = 8
        bottom = h - 8

        path = QPainterPath()

        if self.side == "left":
            path.moveTo(w - 6, top)
            path.cubicTo(8, top + h * 0.18, 8, bottom - h * 0.18, w - 6, bottom)
        else:
            path.moveTo(6, top)
            path.cubicTo(w - 8, top + h * 0.18, w - 8, bottom - h * 0.18, 6, bottom)

        painter.drawPath(path)
        painter.end()


class MatrixDisplayWidget(QWidget):
    def __init__(self, title: str = "A =", matrix=None):
        super().__init__()

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        root.setAlignment(Qt.AlignCenter)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("""
            color: white;
            font-size: 20px;
            font-weight: 900;
            background: transparent;
        """)

        self.left_bracket = ParenthesisWidget("left", 130)
        self.right_bracket = ParenthesisWidget("right", 130)

        self.grid_widget = QWidget()
        self.grid_widget.setStyleSheet("background: transparent;")

        self.grid = QGridLayout(self.grid_widget)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(30)
        self.grid.setVerticalSpacing(10)

        root.addWidget(self.title_label)
        root.addWidget(self.left_bracket)
        root.addWidget(self.grid_widget)
        root.addWidget(self.right_bracket)

        if matrix is not None:
            self.set_matrix(matrix)

    def clear_grid(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def set_matrix(self, matrix):
        self.clear_grid()

        matrix = np.array(matrix)
        rows, cols = matrix.shape

        bracket_height = max(120, rows * 42 + 28)
        self.left_bracket.set_bracket_height(bracket_height)
        self.right_bracket.set_bracket_height(bracket_height)

        for i in range(rows):
            for j in range(cols):
                value = matrix[i][j]

                if isinstance(value, (float, np.floating)):
                    text = f"{value:.6g}"
                else:
                    text = str(value)

                label = QLabel(text)
                label.setAlignment(Qt.AlignCenter)

                width = max(105, min(190, len(text) * 13))
                label.setMinimumWidth(width)

                label.setStyleSheet("""
                    color: white;
                    font-size: 18px;
                    font-weight: 700;
                    background: transparent;
                    padding: 2px;
                """)

                self.grid.addWidget(label, i, j)


class MatrixInputWidget(QWidget):
    def __init__(self, rows: int, cols: int, title: str = "A ="):
        super().__init__()

        self.rows = rows
        self.cols = cols
        self.inputs = []

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        root.setAlignment(Qt.AlignCenter)

        title_label = QLabel(title)
        title_label.setStyleSheet("""
            color: white;
            font-size: 20px;
            font-weight: 900;
            background: transparent;
        """)

        bracket_height = max(130, rows * 46 + 28)

        left_bracket = ParenthesisWidget("left", bracket_height)
        right_bracket = ParenthesisWidget("right", bracket_height)

        grid_widget = QWidget()
        grid_widget.setStyleSheet("background: transparent;")

        grid = QGridLayout(grid_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        for i in range(rows):
            row_inputs = []

            for j in range(cols):
                edit = QLineEdit()
                edit.setAlignment(Qt.AlignCenter)
                edit.setFixedSize(92, 38)
                edit.setStyleSheet("""
                    QLineEdit {
                        background-color: #171717;
                        color: white;
                        border: 1px solid #383838;
                        border-radius: 10px;
                        font-size: 18px;
                        font-weight: 800;
                    }

                    QLineEdit:focus {
                        border: 1px solid #ffffff;
                    }
                """)

                grid.addWidget(edit, i, j)
                row_inputs.append(edit)

            self.inputs.append(row_inputs)

        root.addWidget(title_label)
        root.addWidget(left_bracket)
        root.addWidget(grid_widget)
        root.addWidget(right_bracket)

    def set_matrix(self, matrix):
        matrix = np.array(matrix)

        for i in range(self.rows):
            for j in range(self.cols):
                self.inputs[i][j].setText(f"{matrix[i][j]:.6g}")

    def get_matrix(self):
        matrix = np.zeros((self.rows, self.cols), dtype=float)

        for i in range(self.rows):
            for j in range(self.cols):
                text = self.inputs[i][j].text().replace(",", ".")

                if text.strip() == "":
                    raise ValueError(f"Пустая ячейка матрицы ({i + 1}; {j + 1})")

                matrix[i][j] = float(text)

        return matrix


class AutoTableWidget(QTableWidget):
    def __init__(self):
        super().__init__()

        self.setAlternatingRowColors(True)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectRows)

        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.setStyleSheet("""
            QTableWidget {
                background-color: #111111;
                alternate-background-color: #171717;
                color: white;
                gridline-color: #383838;
                border: 1px solid #252525;
                border-radius: 14px;
                selection-background-color: #383838;
                selection-color: white;
                font-size: 15px;
                font-weight: 600;
            }

            QTableWidget::item {
                padding: 6px;
                border-bottom: 1px solid #222222;
            }

            QHeaderView::section {
                background-color: #383838;
                color: white;
                font-size: 15px;
                font-weight: bold;
                padding: 8px;
                border: 1px solid #2a2a2a;
            }
        """)

    def fill_from_dataframe(self, table):
        table = table.copy()

        rename_map = {
            "element": "el",
            "a_ij": "aij",
            "phi": "φ",
        }

        table = table.rename(columns=rename_map)

        headers = list(table.columns)

        self.clear()
        self.setRowCount(len(table))
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)

        for row_index, (_, row) in enumerate(table.iterrows()):
            for col_index, header in enumerate(headers):
                value = row[header]

                if isinstance(value, (float, np.floating)):
                    text = f"{value:.6f}"
                else:
                    text = str(value)

                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.setItem(row_index, col_index, item)

        self.resizeColumnsToContents()
        self.resizeRowsToContents()

        min_widths = {
            "k": 55,
            "el": 70,
            "aij": 100,
            "φ": 100,
            "a11": 110,
            "a22": 110,
            "a33": 110,
            "S": 110,
        }

        for col, header in enumerate(headers):
            current = self.columnWidth(col)
            needed = min_widths.get(header, 100)
            self.setColumnWidth(col, max(current + 18, needed))

        total_width = self.verticalHeader().width()
        for column in range(self.columnCount()):
            total_width += self.columnWidth(column)
        total_width += 14

        total_height = self.horizontalHeader().height()
        for row in range(self.rowCount()):
            total_height += self.rowHeight(row)
        total_height += 24

        self.setFixedSize(total_width, total_height)

from io import BytesIO

from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


class MathFormula(QLabel):
    def __init__(self, formula: str, font_size: int = 14):
        super().__init__()

        self.setStyleSheet("background: transparent;")
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.set_formula(formula, font_size)

    def set_formula(self, formula: str, font_size: int = 14):
        figure = Figure(figsize=(7, 0.55), dpi=100)
        figure.patch.set_alpha(0)

        canvas = FigureCanvasAgg(figure)

        axes = figure.add_axes([0, 0, 1, 1])
        axes.axis("off")

        axes.text(
            0.0,
            0.5,
            formula,
            color="white",
            fontsize=font_size,
            fontweight="bold",
            ha="left",
            va="center"
        )

        buffer = BytesIO()

        figure.savefig(
            buffer,
            format="png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.02
        )

        pixmap = QPixmap()
        pixmap.loadFromData(buffer.getvalue())

        self.setPixmap(pixmap)
        self.setFixedHeight(pixmap.height() + 4)