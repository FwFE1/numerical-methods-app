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

from ui_components import (
    SectionPanel,
    ResultCard,
    MatrixInputWidget,
    MatrixDisplayWidget,
    AutoTableWidget,
    MathFormula,
)


ACCENT_SOFT = "#e04c91"
PANEL = "#111111"


DEFAULT_A = np.array([
    [0.71, 0.10, 0.12],
    [0.10, 0.34, -0.04],
    [0.12, -0.04, 0.90]
], dtype=float)

DEFAULT_B = np.array([0.29, 0.32, -0.10], dtype=float)


def gauss_method(A, b):
    A = A.copy().astype(float)
    b = b.copy().astype(float)

    n = len(b)

    for k in range(n - 1):
        if abs(A[k][k]) < 1e-12:
            raise ZeroDivisionError("Нулевой ведущий элемент в методе Гаусса.")

        for i in range(k + 1, n):
            m = A[i][k] / A[k][k]

            for j in range(k, n):
                A[i][j] -= m * A[k][j]

            b[i] -= m * b[k]

    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        s = 0.0

        for j in range(i + 1, n):
            s += A[i][j] * x[j]

        if abs(A[i][i]) < 1e-12:
            raise ZeroDivisionError("Нулевой элемент на диагонали при обратном ходе.")

        x[i] = (b[i] - s) / A[i][i]

    return x, A, b


def build_iteration_form(A, b):
    n = len(b)

    B = np.zeros((n, n))
    c = np.zeros(n)

    for i in range(n):
        if abs(A[i][i]) < 1e-12:
            raise ZeroDivisionError("На диагонали матрицы есть нулевой элемент.")

        c[i] = b[i] / A[i][i]

        for j in range(n):
            if i != j:
                B[i][j] = -A[i][j] / A[i][i]

    q = np.linalg.norm(B, ord=np.inf)

    return B, c, q


def simple_iteration_method(A, b, eps):
    B, c, q = build_iteration_form(A, b)

    x_old = np.zeros(len(b))

    rows = []

    rows.append({
        "k": 0,
        "x1": x_old[0],
        "x2": x_old[1],
        "x3": x_old[2],
        "delta": "—",
        "error": "—"
    })

    k = 0

    while True:
        k += 1

        x_new = B @ x_old + c

        delta = np.linalg.norm(x_new - x_old, ord=np.inf)

        if q < 1:
            error = q / (1 - q) * delta
        else:
            error = delta

        rows.append({
            "k": k,
            "x1": x_new[0],
            "x2": x_new[1],
            "x3": x_new[2],
            "delta": delta,
            "error": error
        })

        if error < eps:
            break

        if k > 100:
            raise RuntimeError("Метод простой итерации не сошёлся за 100 итераций.")

        x_old = x_new

    table = pd.DataFrame(rows)

    return x_new, table, B, c, q


class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(7, 3.4), dpi=100)
        self.axes = self.figure.add_subplot(111)

        super().__init__(self.figure)

        self.figure.patch.set_facecolor(PANEL)
        self.axes.set_facecolor(PANEL)

    def plot_iterations(self, table):
        self.axes.clear()

        k = table["k"].astype(float)

        self.axes.plot(k, table["x1"].astype(float), marker="o", color=ACCENT_SOFT, linewidth=2.5, label="x₁")
        self.axes.plot(k, table["x2"].astype(float), marker="s", color="white", linewidth=2.0, label="x₂")
        self.axes.plot(k, table["x3"].astype(float), marker="^", color="#888888", linewidth=2.0, label="x₃")

        self.axes.set_title("Сходимость метода простой итерации", color="white", fontsize=13, pad=12)
        self.axes.set_xlabel("Номер итерации k", color="white")
        self.axes.set_ylabel("Значение", color="white")

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


class Task1Page(QWidget):
    def __init__(self):
        super().__init__()

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(18)

        title = QLabel("Задание 1 — Гаусс и простая итерация")
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
                Решить систему линейных алгебраических уравнений с точностью ε = 0.01:
            </span>
        """)

        formula_system = QLabel()
        formula_system.setTextFormat(Qt.RichText)
        formula_system.setStyleSheet("""
            color: white;
            font-size: 19px;
            font-weight: 600;
            background: transparent;
        """)
        formula_system.setText("""
            0.71x<sub>1</sub> + 0.10x<sub>2</sub> + 0.12x<sub>3</sub> = 0.29,<br>
            0.10x<sub>1</sub> + 0.34x<sub>2</sub> − 0.04x<sub>3</sub> = 0.32,<br>
            0.12x<sub>1</sub> − 0.04x<sub>2</sub> + 0.90x<sub>3</sub> = −0.10.
        """)

        methods_text = QLabel()
        methods_text.setWordWrap(True)
        methods_text.setTextFormat(Qt.RichText)
        methods_text.setText("""
            <span style="font-size:19px; color:white; font-weight:600;">
                Выполнить расчёт методом Гаусса и методом простой итерации.
            </span>
        """)

        iteration_form_title = QLabel("Итерационная форма:")
        iteration_form_title.setStyleSheet("""
            color: white;
            font-size: 18px;
            font-weight: 800;
            background: transparent;
        """)

        formula_iteration = MathFormula(
            r"$x=Bx+c.$",
            font_size=15
        )

        stop_title = QLabel("Критерий остановки:")
        stop_title.setStyleSheet("""
            color: white;
            font-size: 18px;
            font-weight: 800;
            background: transparent;
        """)

        formula_stop = MathFormula(
            r"$\frac{q}{1-q}\cdot\left\|x^{(k)}-x^{(k-1)}\right\|<\varepsilon.$",
            font_size=15
        )

        condition_layout.addWidget(condition_title)
        condition_layout.addWidget(condition_text)
        condition_layout.addWidget(formula_system)
        condition_layout.addWidget(methods_text)
        condition_layout.addWidget(iteration_form_title)
        condition_layout.addWidget(formula_iteration)
        condition_layout.addWidget(stop_title)
        condition_layout.addWidget(formula_stop)

        layout.addWidget(condition_panel)

        input_panel = SectionPanel()
        input_layout = QVBoxLayout(input_panel)
        input_layout.setContentsMargins(18, 18, 18, 18)
        input_layout.setSpacing(12)

        input_title = QLabel("Исходные данные")
        input_title.setObjectName("BlockTitle")
        input_layout.addWidget(input_title)

        row = QHBoxLayout()

        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setDecimals(6)
        self.eps_spin.setMinimum(0.000001)
        self.eps_spin.setMaximum(10)
        self.eps_spin.setSingleStep(0.01)
        self.eps_spin.setValue(0.01)
        self.eps_spin.setMinimumWidth(160)

        eps_label = QLabel("ε:")
        eps_label.setStyleSheet("""
            color: white;
            font-size: 19px;
            font-weight: 800;
            background: transparent;
        """)

        reset_button = QPushButton("Сбросить")
        reset_button.setObjectName("SecondaryButton")
        reset_button.clicked.connect(self.reset_values)

        calculate_button = QPushButton("Рассчитать")
        calculate_button.setObjectName("ActionButton")
        calculate_button.clicked.connect(self.calculate)

        row.addWidget(eps_label)
        row.addWidget(self.eps_spin)
        row.addWidget(reset_button)
        row.addWidget(calculate_button)
        row.addStretch()

        input_layout.addLayout(row)

        hint = QLabel("Матрицу A и вектор b можно редактировать вручную.")
        hint.setObjectName("SmallHint")
        input_layout.addWidget(hint)

        matrices_row = QHBoxLayout()
        matrices_row.addStretch()

        self.matrix_input = MatrixInputWidget(3, 3, "A =")
        self.vector_input = MatrixInputWidget(3, 1, "b =")

        matrices_row.addWidget(self.matrix_input)
        matrices_row.addSpacing(40)
        matrices_row.addWidget(self.vector_input)
        matrices_row.addStretch()

        input_layout.addLayout(matrices_row)

        layout.addWidget(input_panel)

        result_title = QLabel("Результат")
        result_title.setObjectName("BlockTitle")
        layout.addWidget(result_title)

        gauss_label = QLabel("Метод Гаусса")
        gauss_label.setObjectName("BlockTitle")
        layout.addWidget(gauss_label)

        gauss_cards = QHBoxLayout()

        self.gauss_x1_card = ResultCard("x₁")
        self.gauss_x2_card = ResultCard("x₂")
        self.gauss_x3_card = ResultCard("x₃")

        gauss_cards.addWidget(self.gauss_x1_card)
        gauss_cards.addWidget(self.gauss_x2_card)
        gauss_cards.addWidget(self.gauss_x3_card)

        layout.addLayout(gauss_cards)

        iter_label = QLabel("Метод простой итерации")
        iter_label.setObjectName("BlockTitle")
        layout.addWidget(iter_label)

        iter_cards = QHBoxLayout()

        self.iter_x1_card = ResultCard("x₁")
        self.iter_x2_card = ResultCard("x₂")
        self.iter_x3_card = ResultCard("x₃")
        self.q_card = ResultCard("q")

        iter_cards.addWidget(self.iter_x1_card)
        iter_cards.addWidget(self.iter_x2_card)
        iter_cards.addWidget(self.iter_x3_card)
        iter_cards.addWidget(self.q_card)

        layout.addLayout(iter_cards)

        bc_panel = SectionPanel()
        bc_layout = QVBoxLayout(bc_panel)
        bc_layout.setContentsMargins(18, 18, 18, 18)
        bc_layout.setSpacing(18)

        bc_title = QLabel("Итерационная форма")
        bc_title.setObjectName("BlockTitle")

        self.B_widget = MatrixDisplayWidget("B =")
        self.c_widget = MatrixDisplayWidget("c =")

        bc_layout.addWidget(bc_title)
        bc_layout.addWidget(self.B_widget, alignment=Qt.AlignCenter)
        bc_layout.addWidget(self.c_widget, alignment=Qt.AlignCenter)

        layout.addWidget(bc_panel)

        table_title = QLabel("Таблица итераций")
        table_title.setObjectName("BlockTitle")
        layout.addWidget(table_title)

        table_panel = SectionPanel()
        table_layout = QHBoxLayout(table_panel)
        table_layout.setContentsMargins(18, 18, 18, 18)
        table_layout.addStretch()

        self.iter_table = AutoTableWidget()

        table_layout.addWidget(self.iter_table)
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

        self.reset_values()

    def reset_values(self):
        self.eps_spin.setValue(0.01)
        self.matrix_input.set_matrix(DEFAULT_A)
        self.vector_input.set_matrix(DEFAULT_B.reshape(3, 1))
        self.calculate()

    def calculate(self):
        try:
            A = self.matrix_input.get_matrix()
            b = self.vector_input.get_matrix().flatten()
            eps = self.eps_spin.value()

            x_gauss, upper_A, upper_b = gauss_method(A, b)
            x_iter, table, B, c, q = simple_iteration_method(A, b, eps)

            self.gauss_x1_card.set_value(f"{x_gauss[0]:.6f}")
            self.gauss_x2_card.set_value(f"{x_gauss[1]:.6f}")
            self.gauss_x3_card.set_value(f"{x_gauss[2]:.6f}")

            self.iter_x1_card.set_value(f"{x_iter[0]:.6f}")
            self.iter_x2_card.set_value(f"{x_iter[1]:.6f}")
            self.iter_x3_card.set_value(f"{x_iter[2]:.6f}")
            self.q_card.set_value(f"{q:.6f}")

            self.B_widget.set_matrix(B)
            self.c_widget.set_matrix(c.reshape(3, 1))

            self.iter_table.fill_from_dataframe(table)
            self.canvas.plot_iterations(table)

        except Exception as error:
            error_table = pd.DataFrame([{
                "Ошибка": str(error)
            }])
            self.iter_table.fill_from_dataframe(error_table)