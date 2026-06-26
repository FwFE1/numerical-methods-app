import math
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


def lagrange_polynomial(x_value, x_nodes, y_nodes):
    n = len(x_nodes)
    result = 0.0

    for i in range(n):
        basis = 1.0

        for j in range(n):
            if i != j:
                basis *= (x_value - x_nodes[j]) / (x_nodes[i] - x_nodes[j])

        result += y_nodes[i] * basis

    return result


def finite_differences(y_nodes):
    differences = [np.array(y_nodes, dtype=float)]

    for _ in range(1, len(y_nodes)):
        differences.append(np.diff(differences[-1]))

    return differences


def newton_polynomial(x_value, x_nodes, y_nodes):
    h = x_nodes[1] - x_nodes[0]
    t = (x_value - x_nodes[0]) / h

    differences = finite_differences(y_nodes)

    result = y_nodes[0]
    multiplier = 1.0

    for k in range(1, len(y_nodes)):
        multiplier *= t - (k - 1)
        result += multiplier * differences[k][0] / math.factorial(k)

    return result


def create_difference_table(x_nodes, y_nodes):
    differences = finite_differences(y_nodes)

    rows = []

    for i in range(len(x_nodes)):
        row = {
            "i": i,
            "x_i": x_nodes[i],
            "y_i": differences[0][i]
        }

        for k in range(1, len(differences)):
            if i < len(differences[k]):
                row[f"Δ^{k}y_i"] = differences[k][i]
            else:
                row[f"Δ^{k}y_i"] = "—"

        rows.append(row)

    return pd.DataFrame(rows)


def newton_polynomial_text(x_nodes, y_nodes):
    h = x_nodes[1] - x_nodes[0]
    differences = finite_differences(y_nodes)

    parts = []
    parts.append(f"{differences[0][0]:.5f}")

    for k in range(1, len(y_nodes)):
        coefficient = differences[k][0]
        denominator = math.factorial(k)

        if k == 1:
            term = f"{coefficient:+.5f}t"
        else:
            factors = "".join([f"(t-{j})" for j in range(k)])
            term = f"{coefficient:+.5f}{factors}/{denominator}"

        parts.append(term)

    text = " ".join(parts)
    return f"t = (x - {x_nodes[0]:.3f}) / {h:.3f}\nP₅(x) = {text}"


class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(7, 3.4), dpi=100)
        self.axes = self.figure.add_subplot(111)

        super().__init__(self.figure)

        self.figure.patch.set_facecolor(PANEL)
        self.axes.set_facecolor(PANEL)

    def plot_interpolation(self, x_nodes, y_nodes, x_star, y_star):
        self.axes.clear()

        x_min = min(x_nodes)
        x_max = max(x_nodes)

        x_plot = np.linspace(x_min, x_max, 500)
        y_plot = [newton_polynomial(x, x_nodes, y_nodes) for x in x_plot]

        self.axes.plot(
            x_plot,
            y_plot,
            color=ACCENT_SOFT,
            linewidth=2.5,
            label="Интерполяционный многочлен"
        )

        self.axes.scatter(
            x_nodes,
            y_nodes,
            s=70,
            color="white",
            label="Узлы интерполяции",
            zorder=3
        )

        self.axes.scatter(
            [x_star],
            [y_star],
            s=90,
            color=ACCENT_SOFT,
            label="x*",
            zorder=4
        )

        self.axes.set_title("Интерполяционный многочлен", color="white", fontsize=13, pad=12)
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


class Task5Page(QWidget):
    def __init__(self):
        super().__init__()

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(18)

        title = QLabel("Задание 5 — Многочлены Лагранжа и Ньютона")
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
                Выписать интерполяционные многочлены Лагранжа и Ньютона для узловых значений
                {x<sub>i</sub>, y<sub>i</sub>}. Найти значение в точке x* = 0.1315.
            </span>
        """)

        formulas_text = QLabel()
        formulas_text.setWordWrap(True)
        formulas_text.setTextFormat(Qt.RichText)
        formulas_text.setText("""
            <span style="font-size:18px; color:white; font-weight:600;">
                Многочлен Лагранжа:
                P<sub>n</sub>(x) = Σ y<sub>i</sub>L<sub>i</sub>(x).
                <br>
                Многочлен Ньютона для равноотстоящих узлов:
                P<sub>n</sub>(x) = y<sub>0</sub> + tΔy<sub>0</sub> +
                t(t−1)Δ²y<sub>0</sub>/2! + ...
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

        hint = QLabel("Таблицу можно редактировать вручную. Значения по умолчанию соответствуют варианту 8.")
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

        cards_layout = QHBoxLayout()

        self.lagrange_card = ResultCard("Pₗ(x*)")
        self.newton_card = ResultCard("Pₙ(x*)")
        self.diff_card = ResultCard("|Pₗ − Pₙ|")

        cards_layout.addWidget(self.lagrange_card)
        cards_layout.addWidget(self.newton_card)
        cards_layout.addWidget(self.diff_card)

        layout.addLayout(cards_layout)

        polynomial_panel = SectionPanel()
        polynomial_layout = QVBoxLayout(polynomial_panel)
        polynomial_layout.setContentsMargins(18, 18, 18, 18)

        polynomial_title = QLabel("Многочлен Ньютона")
        polynomial_title.setObjectName("BlockTitle")

        self.polynomial_label = QLabel()
        self.polynomial_label.setWordWrap(True)
        self.polynomial_label.setTextFormat(Qt.RichText)

        polynomial_layout.addWidget(polynomial_title)
        polynomial_layout.addWidget(self.polynomial_label)

        layout.addWidget(polynomial_panel)

        diff_title = QLabel("Таблица конечных разностей")
        diff_title.setObjectName("BlockTitle")
        layout.addWidget(diff_title)

        diff_panel = SectionPanel()
        diff_layout = QHBoxLayout(diff_panel)
        diff_layout.setContentsMargins(18, 18, 18, 18)
        diff_layout.addStretch()

        self.diff_table = AutoTableWidget()

        diff_layout.addWidget(self.diff_table)
        diff_layout.addStretch()

        layout.addWidget(diff_panel)

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

            p_lagrange = lagrange_polynomial(x_star, x_nodes, y_nodes)
            p_newton = newton_polynomial(x_star, x_nodes, y_nodes)
            difference = abs(p_lagrange - p_newton)

            self.lagrange_card.set_value(p_lagrange)
            self.newton_card.set_value(p_newton)
            self.diff_card.set_value(difference)

            text = newton_polynomial_text(x_nodes, y_nodes)
            text = text.replace("\n", "<br>")

            self.polynomial_label.setText(f"""
                <span style="font-size:18px; color:white; font-weight:600;">
                    {text}
                </span>
            """)

            difference_table = create_difference_table(x_nodes, y_nodes)
            self.diff_table.fill_from_dataframe(difference_table)

            self.canvas.plot_interpolation(x_nodes, y_nodes, x_star, p_newton)

        except Exception as error:
            self.polynomial_label.setText(f"""
                <span style="font-size:18px; color:white; font-weight:700;">
                    Ошибка: {error}
                </span>
            """)