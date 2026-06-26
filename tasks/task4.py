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


def f(x, c):
    return (x - 1) ** 2 - c * np.exp(x)


def df(x, c):
    return 2 * (x - 1) - c * np.exp(x)


def phi(x, c):
    return 1 - np.sqrt(c * np.exp(x))


def simple_iteration_method(c, eps, x0, a, b):
    q = 0.5 * np.sqrt(c * np.exp(max(a, b)))

    rows = []

    x_old = x0

    rows.append({
        "k": 0,
        "x": x_old,
        "f(x)": f(x_old, c),
        "delta": "—",
        "error": "—"
    })

    k = 0

    while True:
        k += 1

        x_new = phi(x_old, c)

        delta = abs(x_new - x_old)

        if q < 1:
            error_estimate = q / (1 - q) * delta
        else:
            error_estimate = delta

        rows.append({
            "k": k,
            "x": x_new,
            "f(x)": f(x_new, c),
            "delta": delta,
            "error": error_estimate
        })

        if error_estimate < eps:
            break

        if k > 100:
            raise RuntimeError("Метод простой итерации не сошёлся за 100 итераций.")

        x_old = x_new

    return x_new, q, pd.DataFrame(rows)


def bisection_method(c, eps, a, b):
    if f(a, c) * f(b, c) > 0:
        raise ValueError("На выбранном отрезке f(a) и f(b) имеют одинаковые знаки.")

    rows = []
    k = 0

    while True:
        mid = (a + b) / 2

        rows.append({
            "k": k,
            "a": a,
            "b": b,
            "c": mid,
            "f(c)": f(mid, c),
            "(b-a)/2": (b - a) / 2
        })

        if (b - a) / 2 < eps:
            break

        if f(a, c) * f(mid, c) < 0:
            b = mid
        else:
            a = mid

        k += 1

        if k > 100:
            raise RuntimeError("Метод половинного деления не сошёлся за 100 итераций.")

    return mid, pd.DataFrame(rows)


def newton_method(c, eps, x0):
    rows = []

    x_old = x0

    rows.append({
        "k": 0,
        "x": x_old,
        "f(x)": f(x_old, c),
        "f'(x)": df(x_old, c),
        "delta": "—"
    })

    k = 0

    while True:
        k += 1

        derivative = df(x_old, c)

        if abs(derivative) < 1e-12:
            raise ZeroDivisionError("Производная слишком близка к нулю.")

        x_new = x_old - f(x_old, c) / derivative
        delta = abs(x_new - x_old)

        rows.append({
            "k": k,
            "x": x_new,
            "f(x)": f(x_new, c),
            "f'(x)": df(x_new, c),
            "delta": delta
        })

        if delta < eps:
            break

        if k > 100:
            raise RuntimeError("Метод Ньютона не сошёлся за 100 итераций.")

        x_old = x_new

    return x_new, pd.DataFrame(rows)


class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(7, 3.4), dpi=100)
        self.axes = self.figure.add_subplot(111)

        super().__init__(self.figure)

        self.figure.patch.set_facecolor(PANEL)
        self.axes.set_facecolor(PANEL)

    def plot_function(self, c, a, b, roots):
        self.axes.clear()

        left = min(a, b) - 0.1
        right = max(a, b) + 0.1

        x_values = np.linspace(left, right, 400)
        y_values = f(x_values, c)

        self.axes.plot(
            x_values,
            y_values,
            color=ACCENT_SOFT,
            linewidth=2.5,
            label="f(x)"
        )

        self.axes.axhline(0, color="white", linewidth=1)

        for root_name, root_value in roots.items():
            self.axes.scatter(
                [root_value],
                [f(root_value, c)],
                s=70,
                label=root_name
            )

        self.axes.set_title("График функции f(x)", color="white", fontsize=13, pad=12)
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


class Task4Page(QWidget):
    def __init__(self):
        super().__init__()

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(18)

        title = QLabel("Задание 4 — Уточнение корня уравнения")
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
                С точностью ε = 0.01 уточнить один из корней уравнения:
            </span>
        """)

        formula_equation = MathFormula(
            r"$(x-1)^2=0.5e^x.$",
            font_size=14
        )

        methods_text = QLabel()
        methods_text.setWordWrap(True)
        methods_text.setTextFormat(Qt.RichText)
        methods_text.setText("""
            <span style="font-size:19px; color:white; font-weight:600;">
                Выполнить расчёт методом простой итерации, методом половинного деления
                и методом Ньютона.
            </span>
        """)

        formula_function_title = QLabel("Функция:")
        formula_function_title.setStyleSheet("""
            color: white;
            font-size: 18px;
            font-weight: 800;
            background: transparent;
        """)

        formula_function = MathFormula(
            r"$f(x)=(x-1)^2-ce^x.$",
            font_size=14
        )

        formula_iteration_title = QLabel("Метод простой итерации:")
        formula_iteration_title.setStyleSheet("""
            color: white;
            font-size: 18px;
            font-weight: 800;
            background: transparent;
        """)

        formula_iteration = MathFormula(
            r"$x_{k+1}=1-\sqrt{ce^{x_k}}.$",
            font_size=14
        )

        formula_newton_title = QLabel("Метод Ньютона:")
        formula_newton_title.setStyleSheet("""
            color: white;
            font-size: 18px;
            font-weight: 800;
            background: transparent;
        """)

        formula_newton = MathFormula(
            r"$x_{k+1}=x_k-\frac{f(x_k)}{f'(x_k)}.$",
            font_size=14
        )

        condition_layout.addWidget(condition_title)
        condition_layout.addWidget(condition_text)
        condition_layout.addWidget(formula_equation)
        condition_layout.addWidget(methods_text)
        condition_layout.addWidget(formula_function_title)
        condition_layout.addWidget(formula_function)
        condition_layout.addWidget(formula_iteration_title)
        condition_layout.addWidget(formula_iteration)
        condition_layout.addWidget(formula_newton_title)
        condition_layout.addWidget(formula_newton)

        layout.addWidget(condition_panel)

        input_panel = SectionPanel()
        input_layout = QVBoxLayout(input_panel)
        input_layout.setContentsMargins(18, 18, 18, 18)
        input_layout.setSpacing(12)

        input_title = QLabel("Исходные данные")
        input_title.setObjectName("BlockTitle")
        input_layout.addWidget(input_title)

        row1 = QHBoxLayout()

        self.c_spin = self.create_double_spin(0.000001, 100, 0.5, 0.1, 6)
        self.eps_spin = self.create_double_spin(0.000001, 10, 0.01, 0.01, 6)
        self.a_spin = self.create_double_spin(-100, 100, 0.0, 0.1, 6)
        self.b_spin = self.create_double_spin(-100, 100, 0.5, 0.1, 6)

        self.add_labeled_spin(row1, "c:", self.c_spin)
        self.add_labeled_spin(row1, "ε:", self.eps_spin)
        self.add_labeled_spin(row1, "a:", self.a_spin)
        self.add_labeled_spin(row1, "b:", self.b_spin)
        row1.addStretch()

        input_layout.addLayout(row1)

        row2 = QHBoxLayout()

        self.x0_iter_spin = self.create_double_spin(-100, 100, 0.25, 0.1, 6)
        self.x0_newton_spin = self.create_double_spin(-100, 100, 0.0, 0.1, 6)

        self.add_labeled_spin(row2, "x₀ простой итерации:", self.x0_iter_spin)
        self.add_labeled_spin(row2, "x₀ Ньютона:", self.x0_newton_spin)

        calculate_button = QPushButton("Рассчитать")
        calculate_button.setObjectName("ActionButton")
        calculate_button.clicked.connect(self.calculate)

        row2.addWidget(calculate_button)
        row2.addStretch()

        input_layout.addLayout(row2)

        layout.addWidget(input_panel)

        result_title = QLabel("Результат")
        result_title.setObjectName("BlockTitle")
        layout.addWidget(result_title)

        cards_layout = QHBoxLayout()

        self.iter_card = ResultCard("Простая итерация")
        self.bisection_card = ResultCard("Дихотомия")
        self.newton_card = ResultCard("Ньютон")

        cards_layout.addWidget(self.iter_card)
        cards_layout.addWidget(self.bisection_card)
        cards_layout.addWidget(self.newton_card)

        layout.addLayout(cards_layout)

        summary_panel = SectionPanel()
        summary_layout = QVBoxLayout(summary_panel)
        summary_layout.setContentsMargins(18, 18, 18, 18)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(Qt.RichText)

        summary_layout.addWidget(self.summary_label)

        layout.addWidget(summary_panel)

        tables_title = QLabel("Таблицы итераций")
        tables_title.setObjectName("BlockTitle")
        layout.addWidget(tables_title)

        self.iter_table = AutoTableWidget()
        self.bisection_table = AutoTableWidget()
        self.newton_table = AutoTableWidget()

        layout.addWidget(self.create_table_block("Метод простой итерации", self.iter_table))
        layout.addWidget(self.create_table_block("Метод половинного деления", self.bisection_table))
        layout.addWidget(self.create_table_block("Метод Ньютона", self.newton_table))

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
        spin.setMinimum(minimum)
        spin.setMaximum(maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setMinimumWidth(140)
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

    def calculate(self):
        try:
            c = self.c_spin.value()
            eps = self.eps_spin.value()
            a = self.a_spin.value()
            b = self.b_spin.value()
            x0_iter = self.x0_iter_spin.value()
            x0_newton = self.x0_newton_spin.value()

            iter_root, q, iter_table = simple_iteration_method(c, eps, x0_iter, a, b)
            bisection_root, bisection_table = bisection_method(c, eps, a, b)
            newton_root, newton_table = newton_method(c, eps, x0_newton)

            self.iter_card.set_value(iter_root)
            self.bisection_card.set_value(bisection_root)
            self.newton_card.set_value(newton_root)

            self.summary_label.setText(f"""
                <span style="font-size:18px; color:white; font-weight:600;">
                    Уравнение приведено к виду f(x) = (x − 1)<sup>2</sup> − ce<sup>x</sup>.
                    При c = {c:.6f}, ε = {eps:.6f} получены значения:
                    простая итерация — {iter_root:.6f},
                    дихотомия — {bisection_root:.6f},
                    метод Ньютона — {newton_root:.6f}.
                    С точностью ε можно записать x ≈ {newton_root:.2f}.
                    Для метода простой итерации q = {q:.6f}.
                </span>
            """)

            self.iter_table.fill_from_dataframe(iter_table)
            self.bisection_table.fill_from_dataframe(bisection_table)
            self.newton_table.fill_from_dataframe(newton_table)

            self.canvas.plot_function(
                c,
                a,
                b,
                {
                    "Простая итерация": iter_root,
                    "Дихотомия": bisection_root,
                    "Ньютон": newton_root,
                }
            )

        except Exception as error:
            self.summary_label.setText(f"""
                <span style="font-size:18px; color:white; font-weight:700;">
                    Ошибка: {error}
                </span>
            """)