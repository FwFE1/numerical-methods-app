import numpy as np
import pandas as pd

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
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


DEFAULT_X = [0, 0.12, 0.19, 0.35, 0.40, 0.45, 0.62, 0.71, 0.84, 0.91, 1.00]
DEFAULT_Y = [2.1, 2.0, 2.0, 2.5, 2.4, 3.1, 3.2, 3.8, 4.0, 4.6, 5.4]


def least_squares_linear(x, y):
    matrix = np.column_stack([np.ones(len(x)), x])
    coefficients = np.linalg.solve(matrix.T @ matrix, matrix.T @ y)

    a0 = coefficients[0]
    a1 = coefficients[1]

    y_approx = a0 + a1 * x
    error_sum = np.sum((y - y_approx) ** 2)

    return a0, a1, y_approx, error_sum


def least_squares_quadratic(x, y):
    matrix = np.column_stack([np.ones(len(x)), x, x ** 2])
    coefficients = np.linalg.solve(matrix.T @ matrix, matrix.T @ y)

    a0 = coefficients[0]
    a1 = coefficients[1]
    a2 = coefficients[2]

    y_approx = a0 + a1 * x + a2 * x ** 2
    error_sum = np.sum((y - y_approx) ** 2)

    return a0, a1, a2, y_approx, error_sum


def create_sums_table(x, y):
    rows = [
        {"Величина": "n", "Значение": len(x)},
        {"Величина": "Σx", "Значение": np.sum(x)},
        {"Величина": "Σy", "Значение": np.sum(y)},
        {"Величина": "Σx²", "Значение": np.sum(x ** 2)},
        {"Величина": "Σxy", "Значение": np.sum(x * y)},
        {"Величина": "Σx³", "Значение": np.sum(x ** 3)},
        {"Величина": "Σx⁴", "Значение": np.sum(x ** 4)},
        {"Величина": "Σx²y", "Значение": np.sum((x ** 2) * y)},
    ]

    return pd.DataFrame(rows)


def create_values_table(x, y, y_linear, y_quadratic):
    rows = []

    for i in range(len(x)):
        rows.append({
            "i": i + 1,
            "x_i": x[i],
            "y_i": y[i],
            "P1(x_i)": y_linear[i],
            "y_i-P1": y[i] - y_linear[i],
            "P2(x_i)": y_quadratic[i],
            "y_i-P2": y[i] - y_quadratic[i],
        })

    return pd.DataFrame(rows)


class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(7, 3.4), dpi=100)
        self.axes = self.figure.add_subplot(111)

        super().__init__(self.figure)

        self.figure.patch.set_facecolor(PANEL)
        self.axes.set_facecolor(PANEL)

    def plot_approximation(self, x, y, a0_l, a1_l, a0_q, a1_q, a2_q):
        self.axes.clear()

        x_plot = np.linspace(min(x), max(x), 500)

        y_linear = a0_l + a1_l * x_plot
        y_quadratic = a0_q + a1_q * x_plot + a2_q * x_plot ** 2

        self.axes.scatter(
            x,
            y,
            s=70,
            color="white",
            label="Исходные точки",
            zorder=3
        )

        self.axes.plot(
            x_plot,
            y_linear,
            color=ACCENT_SOFT,
            linewidth=2.5,
            label="Линейная аппроксимация"
        )

        self.axes.plot(
            x_plot,
            y_quadratic,
            color="#ffffff",
            linewidth=2.0,
            linestyle="--",
            label="Квадратичная аппроксимация"
        )

        self.axes.set_title("Аппроксимация методом наименьших квадратов", color="white", fontsize=13, pad=12)
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


class Task7Page(QWidget):
    def __init__(self):
        super().__init__()

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(18)

        title = QLabel("Задание 7 — Метод наименьших квадратов")
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
                Методом наименьших квадратов аппроксимировать заданную таблицу
                линейным и квадратичным многочленом.
            </span>
        """)

        formulas_text = QLabel()
        formulas_text.setWordWrap(True)
        formulas_text.setTextFormat(Qt.RichText)
        formulas_text.setText("""
            <span style="font-size:18px; color:white; font-weight:600;">
                Линейный многочлен: P₁(x) = a₀ + a₁x.
                <br>
                Квадратичный многочлен: P₂(x) = a₀ + a₁x + a₂x².
                <br>
                Качество аппроксимации оценивается суммой квадратов отклонений:
                S = Σ(yᵢ − P(xᵢ))².
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

        reset_button = QPushButton("Сбросить таблицу")
        reset_button.setObjectName("SecondaryButton")
        reset_button.clicked.connect(self.fill_default_table)

        calculate_button = QPushButton("Рассчитать")
        calculate_button.setObjectName("ActionButton")
        calculate_button.clicked.connect(self.calculate)

        row.addWidget(reset_button)
        row.addWidget(calculate_button)
        row.addStretch()

        input_layout.addLayout(row)

        hint = QLabel("Таблицу можно редактировать вручную. Значения по умолчанию соответствуют варианту 8.")
        hint.setObjectName("SmallHint")
        input_layout.addWidget(hint)

        table_row = QHBoxLayout()
        table_row.addStretch()

        self.input_table = QTableWidget(11, 2)
        self.input_table.setHorizontalHeaderLabels(["x_i", "y_i"])
        self.input_table.verticalHeader().setVisible(True)
        self.input_table.setAlternatingRowColors(True)
        self.input_table.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.input_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.input_table.setColumnWidth(0, 150)
        self.input_table.setColumnWidth(1, 150)
        self.input_table.setFixedSize(365, 520)

        table_row.addWidget(self.input_table)
        table_row.addStretch()

        input_layout.addLayout(table_row)

        layout.addWidget(input_panel)

        result_title = QLabel("Результат")
        result_title.setObjectName("BlockTitle")
        layout.addWidget(result_title)

        cards_layout = QHBoxLayout()

        self.s1_card = ResultCard("S₁")
        self.s2_card = ResultCard("S₂")
        self.best_card = ResultCard("Лучше")

        cards_layout.addWidget(self.s1_card)
        cards_layout.addWidget(self.s2_card)
        cards_layout.addWidget(self.best_card)

        layout.addLayout(cards_layout)

        polynomials_panel = SectionPanel()
        polynomials_layout = QVBoxLayout(polynomials_panel)
        polynomials_layout.setContentsMargins(18, 18, 18, 18)

        polynomials_title = QLabel("Полученные многочлены")
        polynomials_title.setObjectName("BlockTitle")

        self.polynomials_label = QLabel()
        self.polynomials_label.setWordWrap(True)
        self.polynomials_label.setTextFormat(Qt.RichText)

        polynomials_layout.addWidget(polynomials_title)
        polynomials_layout.addWidget(self.polynomials_label)

        layout.addWidget(polynomials_panel)

        sums_title = QLabel("Суммы для нормальных уравнений")
        sums_title.setObjectName("BlockTitle")
        layout.addWidget(sums_title)

        sums_panel = SectionPanel()
        sums_layout = QHBoxLayout(sums_panel)
        sums_layout.setContentsMargins(18, 18, 18, 18)
        sums_layout.addStretch()

        self.sums_table = AutoTableWidget()

        sums_layout.addWidget(self.sums_table)
        sums_layout.addStretch()

        layout.addWidget(sums_panel)

        values_title = QLabel("Таблица значений")
        values_title.setObjectName("BlockTitle")
        layout.addWidget(values_title)

        values_panel = SectionPanel()
        values_layout = QHBoxLayout(values_panel)
        values_layout.setContentsMargins(18, 18, 18, 18)
        values_layout.addStretch()

        self.values_table = AutoTableWidget()

        values_layout.addWidget(self.values_table)
        values_layout.addStretch()

        layout.addWidget(values_panel)

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
        for i in range(11):
            x_item = QTableWidgetItem(f"{DEFAULT_X[i]:.6f}")
            y_item = QTableWidgetItem(f"{DEFAULT_Y[i]:.6f}")

            x_item.setTextAlignment(Qt.AlignCenter)
            y_item.setTextAlignment(Qt.AlignCenter)

            self.input_table.setItem(i, 0, x_item)
            self.input_table.setItem(i, 1, y_item)

    def read_input_table(self):
        x_values = []
        y_values = []

        for i in range(11):
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
            x, y = self.read_input_table()

            a0_l, a1_l, y_linear, s1 = least_squares_linear(x, y)
            a0_q, a1_q, a2_q, y_quadratic, s2 = least_squares_quadratic(x, y)

            self.s1_card.set_value(s1)
            self.s2_card.set_value(s2)

            if s2 < s1:
                self.best_card.set_value("P₂")
            else:
                self.best_card.set_value("P₁")

            self.polynomials_label.setText(f"""
                <span style="font-size:18px; color:white; font-weight:600;">
                    P₁(x) = {a0_l:.6f} + {a1_l:.6f}x
                    <br><br>
                    P₂(x) = {a0_q:.6f} + {a1_q:.6f}x + {a2_q:.6f}x²
                    <br><br>
                    Так как S₂ = {s2:.6f}, а S₁ = {s1:.6f},
                    квадратичный многочлен аппроксимирует таблицу точнее.
                </span>
            """)

            sums_table = create_sums_table(x, y)
            values_table = create_values_table(x, y, y_linear, y_quadratic)

            self.sums_table.fill_from_dataframe(sums_table)
            self.values_table.fill_from_dataframe(values_table)

            self.canvas.plot_approximation(x, y, a0_l, a1_l, a0_q, a1_q, a2_q)

        except Exception as error:
            self.polynomials_label.setText(f"""
                <span style="font-size:18px; color:white; font-weight:700;">
                    Ошибка: {error}
                </span>
            """)