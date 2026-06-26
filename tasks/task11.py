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


def g(x, r):
    return r * x ** 2


def build_system(a, b, h, p, q, r, left_alpha, left_gamma, right_value):
    """
    Уравнение:
    y'' + p y' + qy = r*x^2

    Левое условие:
    y(a) + left_alpha * y'(a) = left_gamma

    Правое условие:
    y(b) = right_value
    """

    if abs(left_alpha) < 1e-12:
        raise ValueError("Коэффициент при y'(a) в левом условии не должен быть равен нулю.")

    n_float = (b - a) / h

    if abs(n_float - round(n_float)) > 1e-8:
        raise ValueError("Отрезок должен делиться на шаг h без остатка.")

    n = int(round(n_float))

    if n < 2:
        raise ValueError("Для краевой задачи нужно минимум 2 шага.")

    # Узлы: x0, x1, ..., xn
    x_nodes = np.array([a + i * h for i in range(n + 1)], dtype=float)

    # Правое значение y_n известно, неизвестны y_0 ... y_{n-1}
    unknown_count = n

    rows = []

    # Левая граница через разложение Тейлора с точностью O(h^2)
    #
    # y_1 = y_0 + h*y'_0 + h^2/2*y''_0
    # y'_0 = (gamma - y_0) / alpha
    # y''_0 = g(a) - p*y'_0 - q*y_0
    #
    # Получается:
    # A*y_0 - y_1 = -B
    A = 1 - h / left_alpha + (h ** 2 / 2) * (p / left_alpha - q)
    B = h * left_gamma / left_alpha + (h ** 2 / 2) * (g(a, r) - p * left_gamma / left_alpha)

    scale = 2 / h ** 2

    rows.append({
        "i": 1,
        "x_i": x_nodes[0],
        "a_i": 0.0,
        "b_i": scale * A,
        "c_i": -scale,
        "d_i": -scale * B
    })

    # Внутренние узлы x1 ... x_{n-1}
    for j in range(1, unknown_count):
        x = x_nodes[j]

        a_i = -(1 / h ** 2 - p / (2 * h))
        b_i = 2 / h ** 2 - q
        c_i = -(1 / h ** 2 + p / (2 * h))
        d_i = -g(x, r)

        # Если справа стоит известное значение y_n
        if j == unknown_count - 1:
            d_i -= c_i * right_value
            c_i = 0.0

        rows.append({
            "i": j + 1,
            "x_i": x,
            "a_i": a_i,
            "b_i": b_i,
            "c_i": c_i,
            "d_i": d_i
        })

    return x_nodes, pd.DataFrame(rows)


def thomas_method(system_table):
    rows = system_table.copy()

    alphas = []
    betas = []
    denominators = []

    for index, row in rows.iterrows():
        a_i = row["a_i"]
        b_i = row["b_i"]
        c_i = row["c_i"]
        d_i = row["d_i"]

        if index == 0:
            denominator = b_i
            alpha = -c_i / denominator
            beta = d_i / denominator
        else:
            denominator = b_i + a_i * alphas[index - 1]

            if abs(denominator) < 1e-12:
                raise ZeroDivisionError("Нулевой знаменатель в методе прогонки.")

            alpha = -c_i / denominator
            beta = (d_i - a_i * betas[index - 1]) / denominator

        alphas.append(alpha)
        betas.append(beta)
        denominators.append(denominator)

    rows["denominator"] = denominators
    rows["alpha"] = alphas
    rows["beta"] = betas

    m = len(rows)
    y_unknown = np.zeros(m)

    y_unknown[-1] = betas[-1]

    for i in range(m - 2, -1, -1):
        y_unknown[i] = alphas[i] * y_unknown[i + 1] + betas[i]

    return y_unknown, rows


def solve_boundary_problem(a, b, h, p, q, r, left_alpha, left_gamma, right_value):
    x_nodes, system_table = build_system(a, b, h, p, q, r, left_alpha, left_gamma, right_value)

    y_unknown, sweep_table = thomas_method(system_table)

    y_values = np.append(y_unknown, right_value)

    solution_rows = []

    for x, y in zip(x_nodes, y_values):
        solution_rows.append({
            "x_i": x,
            "y_i": y
        })

    solution_table = pd.DataFrame(solution_rows)

    return x_nodes, y_values, sweep_table, solution_table


class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(7, 3.4), dpi=100)
        self.axes = self.figure.add_subplot(111)

        super().__init__(self.figure)

        self.figure.patch.set_facecolor(PANEL)
        self.axes.set_facecolor(PANEL)

    def plot_solution(self, x_nodes, y_values):
        self.axes.clear()

        self.axes.plot(
            x_nodes,
            y_values,
            marker="o",
            color=ACCENT_SOFT,
            linewidth=2.5,
            label="Решение методом прогонки"
        )

        self.axes.set_title("Решение краевой задачи", color="white", fontsize=13, pad=12)
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


class Task11Page(QWidget):
    def __init__(self):
        super().__init__()

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(18)

        title = QLabel("Задание 11 — Краевая задача методом прогонки")
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
                Методом прогонки с шагом h = 0.1 и точностью O(h²)
                решить краевую задачу:
            </span>
        """)

        formula_equation = MathFormula(
            r"$y''-\frac{y'}{2}+3y=2x^2,$",
            font_size=14
        )

        formula_conditions = MathFormula(
            r"$y(1)+2y'(1)=0.6,\quad y(1.3)=1.$",
            font_size=14
        )

        general_text = QLabel()
        general_text.setWordWrap(True)
        general_text.setTextFormat(Qt.RichText)
        general_text.setText("""
            <span style="font-size:18px; color:white; font-weight:600;">
                Общий вид в программе:
            </span>
        """)

        formula_general = MathFormula(
            r"$y''+py'+qy=rx^2.$",
            font_size=14
        )

        boundary_text = QLabel()
        boundary_text.setWordWrap(True)
        boundary_text.setTextFormat(Qt.RichText)
        boundary_text.setText("""
            <span style="font-size:18px; color:white; font-weight:600;">
                Левое условие: y(a) + αy′(a) = γ.
                Правое условие: y(b) = δ.
                <br>
                Для левого условия используется разложение Тейлора с точностью O(h²).
            </span>
        """)

        condition_layout.addWidget(condition_title)
        condition_layout.addWidget(condition_text)
        condition_layout.addWidget(formula_equation)
        condition_layout.addWidget(formula_conditions)
        condition_layout.addWidget(general_text)
        condition_layout.addWidget(formula_general)
        condition_layout.addWidget(boundary_text)

        layout.addWidget(condition_panel)

        input_panel = SectionPanel()
        input_layout = QVBoxLayout(input_panel)
        input_layout.setContentsMargins(18, 18, 18, 18)
        input_layout.setSpacing(12)

        input_title = QLabel("Исходные данные")
        input_title.setObjectName("BlockTitle")
        input_layout.addWidget(input_title)

        row1 = QHBoxLayout()

        self.a_spin = self.create_double_spin(-1000, 1000, 1.0, 0.1, 6)
        self.b_spin = self.create_double_spin(-1000, 1000, 1.3, 0.1, 6)
        self.h_spin = self.create_double_spin(0.000001, 1000, 0.1, 0.1, 6)

        self.add_labeled_spin(row1, "a:", self.a_spin)
        self.add_labeled_spin(row1, "b:", self.b_spin)
        self.add_labeled_spin(row1, "h:", self.h_spin)
        row1.addStretch()

        input_layout.addLayout(row1)

        row2 = QHBoxLayout()

        self.p_spin = self.create_double_spin(-1000, 1000, -0.5, 0.1, 6)
        self.q_spin = self.create_double_spin(-1000, 1000, 3.0, 0.1, 6)
        self.r_spin = self.create_double_spin(-1000, 1000, 2.0, 0.1, 6)

        self.add_labeled_spin(row2, "p при y′:", self.p_spin)
        self.add_labeled_spin(row2, "q при y:", self.q_spin)
        self.add_labeled_spin(row2, "r при x²:", self.r_spin)
        row2.addStretch()

        input_layout.addLayout(row2)

        row3 = QHBoxLayout()

        self.left_alpha_spin = self.create_double_spin(-1000, 1000, 2.0, 0.1, 6)
        self.left_gamma_spin = self.create_double_spin(-1000, 1000, 0.6, 0.1, 6)
        self.right_value_spin = self.create_double_spin(-1000, 1000, 1.0, 0.1, 6)

        self.add_labeled_spin(row3, "α:", self.left_alpha_spin)
        self.add_labeled_spin(row3, "γ:", self.left_gamma_spin)
        self.add_labeled_spin(row3, "δ:", self.right_value_spin)

        reset_button = QPushButton("Сбросить")
        reset_button.setObjectName("SecondaryButton")
        reset_button.clicked.connect(self.reset_values)

        calculate_button = QPushButton("Рассчитать")
        calculate_button.setObjectName("ActionButton")
        calculate_button.clicked.connect(self.calculate)

        row3.addWidget(reset_button)
        row3.addWidget(calculate_button)
        row3.addStretch()

        input_layout.addLayout(row3)

        hint = QLabel("По умолчанию: y″ − y′/2 + 3y = 2x², y(1)+2y′(1)=0.6, y(1.3)=1.")
        hint.setObjectName("SmallHint")
        input_layout.addWidget(hint)

        layout.addWidget(input_panel)

        result_title = QLabel("Результат")
        result_title.setObjectName("BlockTitle")
        layout.addWidget(result_title)

        cards_layout = QHBoxLayout()

        self.y_a_card = ResultCard("y(a)")
        self.y_mid_card = ResultCard("y(a+h)")
        self.y_b_card = ResultCard("y(b)")

        cards_layout.addWidget(self.y_a_card)
        cards_layout.addWidget(self.y_mid_card)
        cards_layout.addWidget(self.y_b_card)

        layout.addLayout(cards_layout)

        summary_panel = SectionPanel()
        summary_layout = QVBoxLayout(summary_panel)
        summary_layout.setContentsMargins(18, 18, 18, 18)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(Qt.RichText)

        summary_layout.addWidget(self.summary_label)

        layout.addWidget(summary_panel)

        sweep_title = QLabel("Таблица прогонки")
        sweep_title.setObjectName("BlockTitle")
        layout.addWidget(sweep_title)

        sweep_panel = SectionPanel()
        sweep_layout = QHBoxLayout(sweep_panel)
        sweep_layout.setContentsMargins(18, 18, 18, 18)
        sweep_layout.addStretch()

        self.sweep_table = AutoTableWidget()

        sweep_layout.addWidget(self.sweep_table)
        sweep_layout.addStretch()

        layout.addWidget(sweep_panel)

        solution_title = QLabel("Решение краевой задачи")
        solution_title.setObjectName("BlockTitle")
        layout.addWidget(solution_title)

        solution_panel = SectionPanel()
        solution_layout = QHBoxLayout(solution_panel)
        solution_layout.setContentsMargins(18, 18, 18, 18)
        solution_layout.addStretch()

        self.solution_table = AutoTableWidget()

        solution_layout.addWidget(self.solution_table)
        solution_layout.addStretch()

        layout.addWidget(solution_panel)

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

    def reset_values(self):
        self.a_spin.setValue(1.0)
        self.b_spin.setValue(1.3)
        self.h_spin.setValue(0.1)

        self.p_spin.setValue(-0.5)
        self.q_spin.setValue(3.0)
        self.r_spin.setValue(2.0)

        self.left_alpha_spin.setValue(2.0)
        self.left_gamma_spin.setValue(0.6)
        self.right_value_spin.setValue(1.0)

        self.calculate()

    def calculate(self):
        try:
            a = self.a_spin.value()
            b = self.b_spin.value()
            h = self.h_spin.value()

            p = self.p_spin.value()
            q = self.q_spin.value()
            r = self.r_spin.value()

            left_alpha = self.left_alpha_spin.value()
            left_gamma = self.left_gamma_spin.value()
            right_value = self.right_value_spin.value()

            x_nodes, y_values, sweep_table, solution_table = solve_boundary_problem(
                a, b, h, p, q, r, left_alpha, left_gamma, right_value
            )

            self.y_a_card.set_value(f"{y_values[0]:.6f}")
            self.y_mid_card.set_value(f"{y_values[1]:.6f}")
            self.y_b_card.set_value(f"{y_values[-1]:.6f}")

            self.summary_label.setText(f"""
                <span style="font-size:18px; color:white; font-weight:600;">
                    Получено решение на сетке с шагом h = {h:.6f}.
                    Значение в левой точке: y({x_nodes[0]:.6f}) = {y_values[0]:.6f}.
                    Значение в правой точке: y({x_nodes[-1]:.6f}) = {y_values[-1]:.6f}.
                </span>
            """)

            self.sweep_table.fill_from_dataframe(sweep_table)
            self.solution_table.fill_from_dataframe(solution_table)
            self.canvas.plot_solution(x_nodes, y_values)

        except Exception as error:
            self.summary_label.setText(f"""
                <span style="font-size:18px; color:white; font-weight:700;">
                    Ошибка: {error}
                </span>
            """)