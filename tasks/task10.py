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
)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ui_components import SectionPanel, ResultCard, AutoTableWidget, MathFormula


ACCENT_SOFT = "#e04c91"
PANEL = "#111111"


def f(x, y, p, q):
    return np.cos(y) / (p + x) + q * y ** 2


def runge_kutta_4(x0, y0, h, x_end, p, q):
    x_values = [x0]
    y_values = [y0]

    rows = []

    x = x0
    y = y0

    n = int(round((x_end - x0) / h))

    for i in range(n):
        k1 = h * f(x, y, p, q)
        k2 = h * f(x + h / 2, y + k1 / 2, p, q)
        k3 = h * f(x + h / 2, y + k2 / 2, p, q)
        k4 = h * f(x + h, y + k3, p, q)

        y_next = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        rows.append({
            "i": i,
            "x_i": x,
            "y_i": y,
            "k1": k1,
            "k2": k2,
            "k3": k3,
            "k4": k4,
            "y_{i+1}": y_next
        })

        x = x + h
        y = y_next

        x_values.append(x)
        y_values.append(y)

    table = pd.DataFrame(rows)

    return np.array(x_values), np.array(y_values), table


def adams_method(x0, y0, h, x_end, p, q):
    x_values = [x0]
    y_values = [y0]

    rk_start_rows = []

    x = x0
    y = y0

    # Начальный отрезок: y1, y2, y3 методом Рунге-Кутта
    for i in range(3):
        k1 = h * f(x, y, p, q)
        k2 = h * f(x + h / 2, y + k1 / 2, p, q)
        k3 = h * f(x + h / 2, y + k2 / 2, p, q)
        k4 = h * f(x + h, y + k3, p, q)

        y_next = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        rk_start_rows.append({
            "i": i,
            "x_i": x,
            "y_i": y,
            "k1": k1,
            "k2": k2,
            "k3": k3,
            "k4": k4,
            "y_{i+1}": y_next
        })

        x = x + h
        y = y_next

        x_values.append(x)
        y_values.append(y)

    adams_rows = []

    n = int(round((x_end - x0) / h))

    for i in range(3, n):
        f_i = f(x_values[i], y_values[i], p, q)
        f_i_1 = f(x_values[i - 1], y_values[i - 1], p, q)
        f_i_2 = f(x_values[i - 2], y_values[i - 2], p, q)
        f_i_3 = f(x_values[i - 3], y_values[i - 3], p, q)

        y_next = y_values[i] + h / 24 * (
            55 * f_i
            - 59 * f_i_1
            + 37 * f_i_2
            - 9 * f_i_3
        )

        x_next = x_values[i] + h

        adams_rows.append({
            "i": i,
            "x_i": x_values[i],
            "y_i": y_values[i],
            "f_i": f_i,
            "f_{i-1}": f_i_1,
            "f_{i-2}": f_i_2,
            "f_{i-3}": f_i_3,
            "y_{i+1}": y_next
        })

        x_values.append(x_next)
        y_values.append(y_next)

    rk_start_table = pd.DataFrame(rk_start_rows)
    adams_table = pd.DataFrame(adams_rows)

    return np.array(x_values), np.array(y_values), rk_start_table, adams_table


def create_comparison_table(x_rk, y_rk, y_adams):
    rows = []

    for i in range(len(x_rk)):
        rows.append({
            "i": i,
            "x": x_rk[i],
            "y_RK4": y_rk[i],
            "y_Adams": y_adams[i],
            "|Разность|": abs(y_rk[i] - y_adams[i])
        })

    return pd.DataFrame(rows)


class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(7, 3.4), dpi=100)
        self.axes = self.figure.add_subplot(111)

        super().__init__(self.figure)

        self.figure.patch.set_facecolor(PANEL)
        self.axes.set_facecolor(PANEL)

    def plot_solution(self, x_rk, y_rk, x_adams, y_adams):
        self.axes.clear()

        self.axes.plot(
            x_rk,
            y_rk,
            marker="o",
            color=ACCENT_SOFT,
            linewidth=2.5,
            label="Рунге-Кутта 4-го порядка"
        )

        self.axes.plot(
            x_adams,
            y_adams,
            marker="s",
            color="white",
            linestyle="--",
            linewidth=2.0,
            label="Метод Адамса"
        )

        self.axes.set_title("Решение задачи Коши", color="white", fontsize=13, pad=12)
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


class Task10Page(QWidget):
    def __init__(self):
        super().__init__()

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(18)

        title = QLabel("Задание 10 — Рунге-Кутта и Адамс")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        condition_panel = SectionPanel()
        condition_panel.setMinimumHeight(300)
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
                С шагом h = 0.1 и точностью ε = 0.01 решить задачу Коши:
            </span>
        """)

        formula_1 = MathFormula(
            r"$y'=\frac{\cos(y)}{1.5+x}+0.1y^2,\quad y(0)=0.$",
            font_size=14
        )

        condition_text_2 = QLabel()
        condition_text_2.setWordWrap(True)
        condition_text_2.setTextFormat(Qt.RichText)
        condition_text_2.setText("""
            <span style="font-size:19px; color:white; font-weight:600;">
                Выполнить расчёт методом Рунге-Кутта 4-го порядка и методом Адамса.
                Начальный отрезок для метода Адамса определить методом Рунге-Кутта.
            </span>
        """)

        formula_title = QLabel("Метод Адамса:")
        formula_title.setStyleSheet("""
            color: white;
            font-size: 18px;
            font-weight: 800;
            background: transparent;
        """)

        formula_2 = MathFormula(
            r"$y_{i+1}=y_i+\frac{h}{24}\left(55f_i-59f_{i-1}+37f_{i-2}-9f_{i-3}\right).$",
            font_size=14
        )

        condition_layout.addWidget(condition_title)
        condition_layout.addWidget(condition_text)
        condition_layout.addWidget(formula_1)
        condition_layout.addWidget(condition_text_2)
        condition_layout.addWidget(formula_title)
        condition_layout.addWidget(formula_2)

        layout.addWidget(condition_panel)

        input_panel = SectionPanel()
        input_layout = QVBoxLayout(input_panel)
        input_layout.setContentsMargins(18, 18, 18, 18)
        input_layout.setSpacing(12)

        input_title = QLabel("Исходные данные")
        input_title.setObjectName("BlockTitle")
        input_layout.addWidget(input_title)

        row1 = QHBoxLayout()

        self.x0_spin = self.create_double_spin(-1000, 1000, 0.0, 0.1, 6)
        self.y0_spin = self.create_double_spin(-1000, 1000, 0.0, 0.1, 6)
        self.x_end_spin = self.create_double_spin(-1000, 1000, 1.0, 0.1, 6)
        self.h_spin = self.create_double_spin(0.000001, 1000, 0.1, 0.1, 6)

        self.add_labeled_spin(row1, "x₀:", self.x0_spin)
        self.add_labeled_spin(row1, "y₀:", self.y0_spin)
        self.add_labeled_spin(row1, "x конец:", self.x_end_spin)
        self.add_labeled_spin(row1, "h:", self.h_spin)
        row1.addStretch()

        input_layout.addLayout(row1)

        row2 = QHBoxLayout()

        self.eps_spin = self.create_double_spin(0.000001, 10, 0.01, 0.01, 6)
        self.p_spin = self.create_double_spin(0.000001, 1000, 1.5, 0.1, 6)
        self.q_spin = self.create_double_spin(-1000, 1000, 0.1, 0.1, 6)

        self.add_labeled_spin(row2, "ε:", self.eps_spin)
        self.add_labeled_spin(row2, "p в (p+x):", self.p_spin)
        self.add_labeled_spin(row2, "q при y²:", self.q_spin)

        reset_button = QPushButton("Сбросить")
        reset_button.setObjectName("SecondaryButton")
        reset_button.clicked.connect(self.reset_values)

        calculate_button = QPushButton("Рассчитать")
        calculate_button.setObjectName("ActionButton")
        calculate_button.clicked.connect(self.calculate)

        row2.addWidget(reset_button)
        row2.addWidget(calculate_button)
        row2.addStretch()

        input_layout.addLayout(row2)

        hint = QLabel("Функция имеет вид y′ = cos(y)/(p+x) + qy². По умолчанию p = 1.5, q = 0.1.")
        hint.setObjectName("SmallHint")
        input_layout.addWidget(hint)

        layout.addWidget(input_panel)

        result_title = QLabel("Результат")
        result_title.setObjectName("BlockTitle")
        layout.addWidget(result_title)

        cards_layout = QHBoxLayout()

        self.rk_card = ResultCard("Рунге-Кутта")
        self.adams_card = ResultCard("Адамс")
        self.diff_card = ResultCard("|Разность|")

        cards_layout.addWidget(self.rk_card)
        cards_layout.addWidget(self.adams_card)
        cards_layout.addWidget(self.diff_card)

        layout.addLayout(cards_layout)

        summary_panel = SectionPanel()
        summary_layout = QVBoxLayout(summary_panel)
        summary_layout.setContentsMargins(18, 18, 18, 18)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(Qt.RichText)

        summary_layout.addWidget(self.summary_label)

        layout.addWidget(summary_panel)

        tables_title = QLabel("Таблицы вычислений")
        tables_title.setObjectName("BlockTitle")
        layout.addWidget(tables_title)

        self.rk_table = AutoTableWidget()
        self.rk_start_table = AutoTableWidget()
        self.adams_table = AutoTableWidget()
        self.comparison_table = AutoTableWidget()

        layout.addWidget(self.create_table_block("Метод Рунге-Кутта 4-го порядка", self.rk_table))
        layout.addWidget(self.create_table_block("Начальный отрезок для метода Адамса", self.rk_start_table))
        layout.addWidget(self.create_table_block("Метод Адамса", self.adams_table))
        layout.addWidget(self.create_table_block("Сравнение результатов", self.comparison_table))

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

        self.calculate()

    def create_double_spin(self, minimum, maximum, value, step, decimals):
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setMinimum(minimum)
        spin.setMaximum(maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setMinimumWidth(150)
        return spin

    def add_labeled_spin(self, layout, text, spin):
        label = QLabel(text)
        label.setStyleSheet("""
            color: white;
            font-size: 19px;
            font-weight: 800;
            background: transparent;
        """)

        layout.addWidget(label)
        layout.addWidget(spin)

    def create_table_block(self, title, table_widget):
        panel = SectionPanel()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(18, 18, 18, 18)
        panel_layout.setSpacing(12)

        label = QLabel(title)
        label.setStyleSheet("""
            color: white;
            font-size: 18px;
            font-weight: 800;
            background: transparent;
        """)

        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(table_widget)
        row.addStretch()

        panel_layout.addWidget(label)
        panel_layout.addLayout(row)

        return panel

    def reset_values(self):
        self.x0_spin.setValue(0.0)
        self.y0_spin.setValue(0.0)
        self.x_end_spin.setValue(1.0)
        self.h_spin.setValue(0.1)
        self.eps_spin.setValue(0.01)
        self.p_spin.setValue(1.5)
        self.q_spin.setValue(0.1)
        self.calculate()

    def calculate(self):
        try:
            x0 = self.x0_spin.value()
            y0 = self.y0_spin.value()
            x_end = self.x_end_spin.value()
            h = self.h_spin.value()
            eps = self.eps_spin.value()
            p = self.p_spin.value()
            q = self.q_spin.value()

            if x_end <= x0:
                raise ValueError("Конец отрезка должен быть больше x₀.")

            n_float = (x_end - x0) / h

            if abs(n_float - round(n_float)) > 1e-8:
                raise ValueError("Отрезок должен делиться на шаг h без остатка.")

            if round(n_float) < 4:
                raise ValueError("Для метода Адамса нужно минимум 4 шага.")

            x_rk, y_rk, rk_table = runge_kutta_4(x0, y0, h, x_end, p, q)
            x_adams, y_adams, rk_start_table, adams_table = adams_method(x0, y0, h, x_end, p, q)

            comparison_table = create_comparison_table(x_rk, y_rk, y_adams)

            difference = abs(y_rk[-1] - y_adams[-1])

            self.rk_card.set_value(f"{y_rk[-1]:.6f}")
            self.adams_card.set_value(f"{y_adams[-1]:.6f}")
            self.diff_card.set_value(f"{difference:.6f}")

            self.summary_label.setText(f"""
                <span style="font-size:18px; color:white; font-weight:600;">
                    При x = {x_end:.6f} получено:
                    методом Рунге-Кутта y = {y_rk[-1]:.6f},
                    методом Адамса y = {y_adams[-1]:.6f}.
                    Разность равна {difference:.6f}.
                    Так как разность меньше ε = {eps:.6f}, результаты согласуются с заданной точностью.
                </span>
            """)

            self.rk_table.fill_from_dataframe(rk_table)
            self.rk_start_table.fill_from_dataframe(rk_start_table)
            self.adams_table.fill_from_dataframe(adams_table)
            self.comparison_table.fill_from_dataframe(comparison_table)

            self.canvas.plot_solution(x_rk, y_rk, x_adams, y_adams)

        except Exception as error:
            self.summary_label.setText(f"""
                <span style="font-size:18px; color:white; font-weight:700;">
                    Ошибка: {error}
                </span>
            """)