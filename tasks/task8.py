import numpy as np
import pandas as pd

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QDoubleSpinBox,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ui_components import SectionPanel, ResultCard, AutoTableWidget


ACCENT_SOFT = "#e04c91"
PANEL = "#111111"


DEFAULT_X = [0.110, 0.116, 0.122, 0.128, 0.134, 0.140]
DEFAULT_Y = [8.65729, 8.29329, 7.95829, 7.64893, 7.36235, 7.09613]
DEFAULT_X_STAR = 0.1315


def select_three_nodes(x_nodes, x_value):
    """
    Автоматически выбирает три соседних узла для построения
    квадратичного интерполяционного многочлена.
    """

    n = len(x_nodes)

    if n < 3:
        raise ValueError("Для вычисления производных нужно минимум 3 узла.")

    candidates = []

    for i in range(n - 2):
        left = x_nodes[i]
        right = x_nodes[i + 2]
        middle = x_nodes[i + 1]

        if left <= x_value <= right:
            candidates.append((abs(x_value - middle), i))

    if candidates:
        candidates.sort()
        return candidates[0][1]

    if x_value < x_nodes[0]:
        return 0

    return n - 3


def quadratic_derivatives(x_nodes, y_nodes, x_value):
    start_index = select_three_nodes(x_nodes, x_value)

    xs = x_nodes[start_index:start_index + 3]
    ys = y_nodes[start_index:start_index + 3]

    coefficients = np.polyfit(xs, ys, 2)

    a = coefficients[0]
    b = coefficients[1]
    c = coefficients[2]

    first_derivative = 2 * a * x_value + b
    second_derivative = 2 * a

    y_value = a * x_value ** 2 + b * x_value + c

    return {
        "start_index": start_index,
        "nodes": xs,
        "y_value": y_value,
        "first_derivative": first_derivative,
        "second_derivative": second_derivative,
    }


def create_result_table(x_nodes, y_nodes, x_star):
    x2 = x_nodes[1]

    result_star = quadratic_derivatives(x_nodes, y_nodes, x_star)
    result_x2 = quadratic_derivatives(x_nodes, y_nodes, x2)

    rows = []

    rows.append({
        "Точка": "x*",
        "x": x_star,
        "Узлы": f"{result_star['nodes'][0]:.3f}; {result_star['nodes'][1]:.3f}; {result_star['nodes'][2]:.3f}",
        "y": result_star["y_value"],
        "y'": result_star["first_derivative"],
        "y''": result_star["second_derivative"],
    })

    rows.append({
        "Точка": "x₂",
        "x": x2,
        "Узлы": f"{result_x2['nodes'][0]:.3f}; {result_x2['nodes'][1]:.3f}; {result_x2['nodes'][2]:.3f}",
        "y": result_x2["y_value"],
        "y'": result_x2["first_derivative"],
        "y''": result_x2["second_derivative"],
    })

    return pd.DataFrame(rows), result_star, result_x2


class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(7, 3.4), dpi=100)
        self.axes = self.figure.add_subplot(111)

        super().__init__(self.figure)

        self.figure.patch.set_facecolor(PANEL)
        self.axes.set_facecolor(PANEL)

    def plot_table(self, x_nodes, y_nodes, x_star, y_star, x2, y_x2):
        self.axes.clear()

        self.axes.plot(
            x_nodes,
            y_nodes,
            color=ACCENT_SOFT,
            linewidth=2.5,
            marker="o",
            label="Табличные значения"
        )

        self.axes.scatter(
            [x_star],
            [y_star],
            s=90,
            color="white",
            label="x*",
            zorder=4
        )

        self.axes.scatter(
            [x2],
            [y_x2],
            s=90,
            color="#e04c91",
            label="x₂",
            zorder=4
        )

        self.axes.set_title("Табличная функция из задания №5", color="white", fontsize=13, pad=12)
        self.axes.set_xlabel("x", color="white")
        self.axes.set_ylabel("y", color="white")

        self.axes.grid(True, color="#333333", linewidth=0.8)

        self.axes.tick_params(axis="x", colors="white")
        self.axes.tick_params(axis="y", colors="white")

        for spine in self.axes.spines.values():
            spine.set_color("#555555")

        legend = self.axes.legend()
        legend.get_frame().set_facecolor("#111111")
        legend.get_frame().set_edgecolor("#383838")

        for text in legend.get_texts():
            text.set_color("white")

        self.figure.tight_layout()
        self.draw()


class Task8Page(QWidget):
    def __init__(self):
        super().__init__()

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(18)

        title = QLabel("Задание 8 — Производные по таблице")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        condition_panel = SectionPanel()
        condition_layout = QVBoxLayout(condition_panel)
        condition_layout.setContentsMargins(18, 18, 18, 18)
        condition_layout.setSpacing(12)

        condition_title = QLabel("Условие задачи")
        condition_title.setObjectName("BlockTitle")

        condition_text = QLabel()
        condition_text.setWordWrap(True)
        condition_text.setTextFormat(Qt.RichText)
        condition_text.setText("""
            <span style="font-size:19px; color:white; font-weight:600;">
                Используя таблицу задания №5, найти значения первой и второй производных
                в точках x* и x₂.
            </span>
        """)

        formulas_text = QLabel()
        formulas_text.setWordWrap(True)
        formulas_text.setTextFormat(Qt.RichText)
        formulas_text.setText("""
            <span style="font-size:18px; color:white; font-weight:600;">
                Для вычисления производных используется квадратичный интерполяционный многочлен
                по трём ближайшим узлам. После построения многочлена находятся значения
                y′(x) и y″(x).
            </span>
        """)

        condition_layout.addWidget(condition_title)
        condition_layout.addWidget(condition_text)
        condition_layout.addWidget(formulas_text)

        layout.addWidget(condition_panel)

        input_panel = SectionPanel()
        input_layout = QVBoxLayout(input_panel)
        input_layout.setContentsMargins(18, 18, 18, 18)
        input_layout.setSpacing(12)

        input_title = QLabel("Исходные данные")
        input_title.setObjectName("BlockTitle")
        input_layout.addWidget(input_title)

        row = QHBoxLayout()

        self.x_star_spin = QDoubleSpinBox()
        self.x_star_spin.setDecimals(6)
        self.x_star_spin.setMinimum(-1000)
        self.x_star_spin.setMaximum(1000)
        self.x_star_spin.setSingleStep(0.001)
        self.x_star_spin.setValue(DEFAULT_X_STAR)
        self.x_star_spin.setMinimumWidth(170)

        x_star_label = QLabel("x*:")
        x_star_label.setStyleSheet("""
            color: white;
            font-size: 19px;
            font-weight: 800;
            background: transparent;
        """)

        reset_button = QPushButton("Сбросить таблицу")
        reset_button.setObjectName("SecondaryButton")
        reset_button.clicked.connect(self.fill_default_table)

        calculate_button = QPushButton("Рассчитать")
        calculate_button.setObjectName("ActionButton")
        calculate_button.clicked.connect(self.calculate)

        row.addWidget(x_star_label)
        row.addWidget(self.x_star_spin)
        row.addWidget(reset_button)
        row.addWidget(calculate_button)
        row.addStretch()

        input_layout.addLayout(row)

        hint = QLabel("Таблицу можно редактировать вручную. x₂ берётся как второй узел таблицы.")
        hint.setObjectName("SmallHint")
        input_layout.addWidget(hint)

        table_row = QHBoxLayout()
        table_row.addStretch()

        self.input_table = QTableWidget(6, 2)
        self.input_table.setHorizontalHeaderLabels(["x_i", "y_i"])
        self.input_table.verticalHeader().setVisible(True)
        self.input_table.setAlternatingRowColors(True)
        self.input_table.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.input_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.input_table.setColumnWidth(0, 150)
        self.input_table.setColumnWidth(1, 150)
        self.input_table.setFixedSize(365, 335)

        table_row.addWidget(self.input_table)
        table_row.addStretch()

        input_layout.addLayout(table_row)

        layout.addWidget(input_panel)

        result_title = QLabel("Результат")
        result_title.setObjectName("BlockTitle")
        layout.addWidget(result_title)

        cards_row_1 = QHBoxLayout()

        self.y1_star_card = ResultCard("y′(x*)")
        self.y2_star_card = ResultCard("y″(x*)")

        cards_row_1.addWidget(self.y1_star_card)
        cards_row_1.addWidget(self.y2_star_card)

        layout.addLayout(cards_row_1)

        cards_row_2 = QHBoxLayout()

        self.y1_x2_card = ResultCard("y′(x₂)")
        self.y2_x2_card = ResultCard("y″(x₂)")

        cards_row_2.addWidget(self.y1_x2_card)
        cards_row_2.addWidget(self.y2_x2_card)

        layout.addLayout(cards_row_2)

        table_title = QLabel("Таблица расчёта производных")
        table_title.setObjectName("BlockTitle")
        layout.addWidget(table_title)

        result_panel = SectionPanel()
        result_layout = QHBoxLayout(result_panel)
        result_layout.setContentsMargins(18, 18, 18, 18)
        result_layout.addStretch()

        self.result_table = AutoTableWidget()

        result_layout.addWidget(self.result_table)
        result_layout.addStretch()

        layout.addWidget(result_panel)

        graph_title = QLabel("График")
        graph_title.setObjectName("BlockTitle")
        layout.addWidget(graph_title)

        graph_panel = SectionPanel()
        graph_layout = QVBoxLayout(graph_panel)
        graph_layout.setContentsMargins(16, 16, 16, 16)

        self.canvas = PlotCanvas()
        self.canvas.setMinimumHeight(330)

        graph_layout.addWidget(self.canvas)

        layout.addWidget(graph_panel)

        scroll.setWidget(content)
        root.addWidget(scroll)

        self.fill_default_table()
        self.calculate()

    def fill_default_table(self):
        for i in range(6):
            x_item = QTableWidgetItem(f"{DEFAULT_X[i]:.6f}")
            y_item = QTableWidgetItem(f"{DEFAULT_Y[i]:.6f}")

            x_item.setTextAlignment(Qt.AlignCenter)
            y_item.setTextAlignment(Qt.AlignCenter)

            self.input_table.setItem(i, 0, x_item)
            self.input_table.setItem(i, 1, y_item)

    def read_input_table(self):
        x_values = []
        y_values = []

        for i in range(6):
            x_item = self.input_table.item(i, 0)
            y_item = self.input_table.item(i, 1)

            if x_item is None or y_item is None:
                raise ValueError(f"Пустая строка таблицы под номером {i + 1}")

            x_text = x_item.text().replace(",", ".")
            y_text = y_item.text().replace(",", ".")

            x_values.append(float(x_text))
            y_values.append(float(y_text))

        return np.array(x_values, dtype=float), np.array(y_values, dtype=float)

    def calculate(self):
        try:
            x_nodes, y_nodes = self.read_input_table()
            x_star = self.x_star_spin.value()

            if len(x_nodes) < 3:
                raise ValueError("Для вычисления производных нужно минимум 3 точки.")

            if np.any(np.diff(x_nodes) <= 0):
                raise ValueError("Узлы x_i должны идти строго по возрастанию.")

            result_table, result_star, result_x2 = create_result_table(x_nodes, y_nodes, x_star)

            self.y1_star_card.set_value(result_star["first_derivative"])
            self.y2_star_card.set_value(result_star["second_derivative"])
            self.y1_x2_card.set_value(result_x2["first_derivative"])
            self.y2_x2_card.set_value(result_x2["second_derivative"])

            self.result_table.fill_from_dataframe(result_table)

            x2 = x_nodes[1]

            self.canvas.plot_table(
                x_nodes,
                y_nodes,
                x_star,
                result_star["y_value"],
                x2,
                result_x2["y_value"]
            )

        except Exception as error:
            error_table = pd.DataFrame([{
                "Ошибка": str(error)
            }])
            self.result_table.fill_from_dataframe(error_table)