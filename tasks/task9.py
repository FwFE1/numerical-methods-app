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


def f(x):
    return np.log10(x ** 2 + 1) / x


def simpson_method(a, b, n):
    h = (b - a) / n

    x = np.linspace(a, b, n + 1)
    y = f(x)

    result = y[0] + y[-1]

    for i in range(1, n):
        if i % 2 == 1:
            result += 4 * y[i]
        else:
            result += 2 * y[i]

    result *= h / 3

    return result


def solve_simpson(a, b, eps):
    if a >= b:
        raise ValueError("Должно быть a < b.")

    if a <= 0 <= b:
        raise ValueError("Отрезок не должен содержать x = 0, так как функция содержит деление на x.")

    n = 2
    integral_old = simpson_method(a, b, n)

    rows = []

    rows.append({
        "n": n,
        "h": (b - a) / n,
        "I_n": integral_old,
        "R": "—"
    })

    while True:
        n *= 2
        integral_new = simpson_method(a, b, n)

        R = abs(integral_new - integral_old) / 15

        rows.append({
            "n": n,
            "h": (b - a) / n,
            "I_n": integral_new,
            "R": R
        })

        if R < eps:
            break

        if n > 1_000_000:
            raise RuntimeError("Метод не достиг заданной точности.")

        integral_old = integral_new

    table = pd.DataFrame(rows)

    return integral_new, R, n, table


class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(7, 3.4), dpi=100)
        self.axes = self.figure.add_subplot(111)

        super().__init__(self.figure)

        self.figure.patch.set_facecolor(PANEL)
        self.axes.set_facecolor(PANEL)

    def plot_function(self, a, b):
        self.axes.clear()

        x_plot = np.linspace(a, b, 500)
        y_plot = f(x_plot)

        self.axes.plot(
            x_plot,
            y_plot,
            color=ACCENT_SOFT,
            linewidth=2.5,
            label="f(x)"
        )

        self.axes.fill_between(
            x_plot,
            y_plot,
            alpha=0.25,
            color=ACCENT_SOFT,
            label="Площадь под графиком"
        )

        self.axes.set_title("График функции f(x) = lg(x² + 1) / x", color="white", fontsize=13, pad=12)
        self.axes.set_xlabel("x", color="white")
        self.axes.set_ylabel("f(x)", color="white")

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


class Task9Page(QWidget):
    def __init__(self):
        super().__init__()

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(18)

        title = QLabel("Задание 9 — Метод Симпсона")
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
                Методом Симпсона с точностью ε = 10⁻⁴ вычислить интеграл:
            </span>
        """)

        formula_integral = MathFormula(
            r"$I=\int_{0.5}^{1}\frac{\lg(x^2+1)}{x}\,dx.$",
            font_size=14
        )

        formula_simpson_title = QLabel("Формула Симпсона:")
        formula_simpson_title.setStyleSheet("""
            color: white;
            font-size: 18px;
            font-weight: 800;
            background: transparent;
        """)

        formula_simpson = MathFormula(
            r"$I_n=\frac{h}{3}\left[f(x_0)+f(x_n)+4\sum f(x_{2i-1})+2\sum f(x_{2i})\right].$",
            font_size=14
        )

        formula_runge_title = QLabel("Оценка точности по правилу Рунге:")
        formula_runge_title.setStyleSheet("""
            color: white;
            font-size: 18px;
            font-weight: 800;
            background: transparent;
        """)

        formula_runge = MathFormula(
            r"$R=\frac{|I_{2n}-I_n|}{15}.$",
            font_size=14
        )

        condition_layout.addWidget(condition_title)
        condition_layout.addWidget(condition_text)
        condition_layout.addWidget(formula_integral)
        condition_layout.addWidget(formula_simpson_title)
        condition_layout.addWidget(formula_simpson)
        condition_layout.addWidget(formula_runge_title)
        condition_layout.addWidget(formula_runge)

        layout.addWidget(condition_panel)

        input_panel = SectionPanel()
        input_layout = QVBoxLayout(input_panel)
        input_layout.setContentsMargins(18, 18, 18, 18)
        input_layout.setSpacing(12)

        input_title = QLabel("Исходные данные")
        input_title.setObjectName("BlockTitle")
        input_layout.addWidget(input_title)

        row = QHBoxLayout()

        self.a_spin = self.create_double_spin(-1000, 1000, 0.5, 0.1, 6)
        self.b_spin = self.create_double_spin(-1000, 1000, 1.0, 0.1, 6)
        self.eps_spin = self.create_double_spin(0.00000001, 10, 0.0001, 0.0001, 8)

        self.add_labeled_spin(row, "a:", self.a_spin)
        self.add_labeled_spin(row, "b:", self.b_spin)
        self.add_labeled_spin(row, "ε:", self.eps_spin)

        reset_button = QPushButton("Сбросить")
        reset_button.setObjectName("SecondaryButton")
        reset_button.clicked.connect(self.reset_values)

        calculate_button = QPushButton("Рассчитать")
        calculate_button.setObjectName("ActionButton")
        calculate_button.clicked.connect(self.calculate)

        row.addWidget(reset_button)
        row.addWidget(calculate_button)
        row.addStretch()

        input_layout.addLayout(row)

        hint = QLabel("Функция фиксирована: f(x) = lg(x² + 1) / x. Пределы интегрирования и точность можно изменить.")
        hint.setObjectName("SmallHint")
        input_layout.addWidget(hint)

        layout.addWidget(input_panel)

        result_title = QLabel("Результат")
        result_title.setObjectName("BlockTitle")
        layout.addWidget(result_title)

        cards_layout = QHBoxLayout()

        self.integral_card = ResultCard("I")
        self.n_card = ResultCard("n")
        self.r_card = ResultCard("R")

        cards_layout.addWidget(self.integral_card)
        cards_layout.addWidget(self.n_card)
        cards_layout.addWidget(self.r_card)

        layout.addLayout(cards_layout)

        summary_panel = SectionPanel()
        summary_layout = QVBoxLayout(summary_panel)
        summary_layout.setContentsMargins(18, 18, 18, 18)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(Qt.RichText)

        summary_layout.addWidget(self.summary_label)

        layout.addWidget(summary_panel)

        table_title = QLabel("Таблица уточнений")
        table_title.setObjectName("BlockTitle")
        layout.addWidget(table_title)

        table_panel = SectionPanel()
        table_layout = QHBoxLayout(table_panel)
        table_layout.setContentsMargins(18, 18, 18, 18)
        table_layout.addStretch()

        self.result_table = AutoTableWidget()

        table_layout.addWidget(self.result_table)
        table_layout.addStretch()

        layout.addWidget(table_panel)

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
        spin.setMinimumWidth(160)
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

    def reset_values(self):
        self.a_spin.setValue(0.5)
        self.b_spin.setValue(1.0)
        self.eps_spin.setValue(0.0001)
        self.calculate()

    def calculate(self):
        try:
            a = self.a_spin.value()
            b = self.b_spin.value()
            eps = self.eps_spin.value()

            answer, R, n, table = solve_simpson(a, b, eps)

            self.integral_card.set_value(f"{answer:.8f}")
            self.n_card.set_value(str(n))
            self.r_card.set_value(f"{R:.8f}")

            self.summary_label.setText(f"""
                <span style="font-size:18px; color:white; font-weight:600;">
                    Получено значение интеграла I = {answer:.8f}.
                    Последняя оценка погрешности R = {R:.8f}.
                    Так как R &lt; ε, заданная точность достигнута при n = {n}.
                </span>
            """)

            self.result_table.fill_from_dataframe(table)
            self.canvas.plot_function(a, b)

        except Exception as error:
            self.summary_label.setText(f"""
                <span style="font-size:18px; color:white; font-weight:700;">
                    Ошибка: {error}
                </span>
            """)