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
)


ACCENT_SOFT = "#e04c91"
PANEL = "#111111"


DEFAULT_A = np.array([
    [8.5, -2.0, 0.0, 0.0],
    [-1.5, 6.5, -2.0, 0.0],
    [0.0, 2.0, 10.5, -4.0],
    [0.0, 0.0, -1.0, 5.5],
], dtype=float)

DEFAULT_B = np.array([13.5, 5.0, 15.0, 9.5], dtype=float)


def check_tridiagonal(A):
    n = A.shape[0]

    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and abs(A[i][j]) > 1e-12:
                raise ValueError("Метод прогонки применим только к трёхдиагональной матрице.")


def thomas_method(A, d):
    A = A.copy().astype(float)
    d = d.copy().astype(float)

    check_tridiagonal(A)

    n = len(d)

    lower = np.zeros(n)
    main = np.zeros(n)
    upper = np.zeros(n)

    for i in range(n):
        main[i] = A[i][i]

        if i > 0:
            lower[i] = A[i][i - 1]

        if i < n - 1:
            upper[i] = A[i][i + 1]

    rows = []
    alpha = np.zeros(n)
    beta = np.zeros(n)

    for i in range(n):
        if i == 0:
            denominator = main[i]

            if abs(denominator) < 1e-12:
                raise ZeroDivisionError("Нулевой знаменатель в методе прогонки.")

            alpha[i] = -upper[i] / denominator
            beta[i] = d[i] / denominator
        else:
            denominator = main[i] + lower[i] * alpha[i - 1]

            if abs(denominator) < 1e-12:
                raise ZeroDivisionError("Нулевой знаменатель в методе прогонки.")

            alpha[i] = -upper[i] / denominator
            beta[i] = (d[i] - lower[i] * beta[i - 1]) / denominator

        rows.append({
            "i": i + 1,
            "a_i": lower[i],
            "b_i": main[i],
            "c_i": upper[i],
            "d_i": d[i],
            "denominator": denominator,
            "alpha": alpha[i],
            "beta": beta[i],
        })

    x = np.zeros(n)
    x[-1] = beta[-1]

    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]

    table = pd.DataFrame(rows)

    return x, table


def gauss_seidel_method(A, b, eps):
    A = A.copy().astype(float)
    b = b.copy().astype(float)

    n = len(b)

    x_old = np.zeros(n)
    x_new = np.zeros(n)

    rows = []

    rows.append({
        "k": 0,
        "x1": x_old[0],
        "x2": x_old[1],
        "x3": x_old[2],
        "x4": x_old[3],
        "delta": "—",
        "residual": np.linalg.norm(A @ x_old - b, ord=np.inf),
    })

    k = 0

    while True:
        k += 1

        x_previous = x_old.copy()

        for i in range(n):
            if abs(A[i][i]) < 1e-12:
                raise ZeroDivisionError("На диагонали матрицы есть нулевой элемент.")

            left_sum = 0.0
            right_sum = 0.0

            for j in range(i):
                left_sum += A[i][j] * x_new[j]

            for j in range(i + 1, n):
                right_sum += A[i][j] * x_old[j]

            x_new[i] = (b[i] - left_sum - right_sum) / A[i][i]

        delta = np.linalg.norm(x_new - x_previous, ord=np.inf)
        residual = np.linalg.norm(A @ x_new - b, ord=np.inf)

        rows.append({
            "k": k,
            "x1": x_new[0],
            "x2": x_new[1],
            "x3": x_new[2],
            "x4": x_new[3],
            "delta": delta,
            "residual": residual,
        })

        if delta < eps:
            break

        if k > 500:
            raise RuntimeError("Метод Зейделя не сошёлся за 500 итераций.")

        x_old = x_new.copy()

    table = pd.DataFrame(rows)

    return x_new, table


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
        self.axes.plot(k, table["x4"].astype(float), marker="D", color="#cfcfcf", linewidth=2.0, label="x₄")

        self.axes.set_title("Сходимость метода Зейделя", color="white", fontsize=13, pad=12)
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


class Task2Page(QWidget):
    def __init__(self):
        super().__init__()

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(18)

        title = QLabel("Задание 2 — Прогонка и Зейдель")
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
                <br><br>
                8.5x₁ − 2x₂ = 13.5
                <br>
                −1.5x₁ + 6.5x₂ − 2x₃ = 5.0
                <br>
                2x₂ + 10.5x₃ − 4x₄ = 15.0
                <br>
                −x₃ + 5.5x₄ = 9.5
                <br><br>
                Выполнить расчёт методом прогонки и методом Зейделя.
            </span>
        """)

        formulas_text = QLabel()
        formulas_text.setWordWrap(True)
        formulas_text.setTextFormat(Qt.RichText)
        formulas_text.setText("""
            <span style="font-size:18px; color:white; font-weight:600;">
                Метод прогонки применяется к трёхдиагональной системе.
                <br>
                Метод Зейделя использует уже найденные значения на текущей итерации.
                Критерий остановки: ||x⁽ᵏ⁾ − x⁽ᵏ⁻¹⁾|| &lt; ε.
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

        self.matrix_input = MatrixInputWidget(4, 4, "A =")
        self.vector_input = MatrixInputWidget(4, 1, "b =")

        matrices_row.addWidget(self.matrix_input)
        matrices_row.addSpacing(40)
        matrices_row.addWidget(self.vector_input)
        matrices_row.addStretch()

        input_layout.addLayout(matrices_row)

        layout.addWidget(input_panel)

        result_title = QLabel("Результат")
        result_title.setObjectName("BlockTitle")
        layout.addWidget(result_title)

        sweep_label = QLabel("Метод прогонки")
        sweep_label.setObjectName("BlockTitle")
        layout.addWidget(sweep_label)

        sweep_cards = QHBoxLayout()

        self.sweep_x1_card = ResultCard("x₁")
        self.sweep_x2_card = ResultCard("x₂")
        self.sweep_x3_card = ResultCard("x₃")
        self.sweep_x4_card = ResultCard("x₄")

        sweep_cards.addWidget(self.sweep_x1_card)
        sweep_cards.addWidget(self.sweep_x2_card)
        sweep_cards.addWidget(self.sweep_x3_card)
        sweep_cards.addWidget(self.sweep_x4_card)

        layout.addLayout(sweep_cards)

        seidel_label = QLabel("Метод Зейделя")
        seidel_label.setObjectName("BlockTitle")
        layout.addWidget(seidel_label)

        seidel_cards = QHBoxLayout()

        self.seidel_x1_card = ResultCard("x₁")
        self.seidel_x2_card = ResultCard("x₂")
        self.seidel_x3_card = ResultCard("x₃")
        self.seidel_x4_card = ResultCard("x₄")

        seidel_cards.addWidget(self.seidel_x1_card)
        seidel_cards.addWidget(self.seidel_x2_card)
        seidel_cards.addWidget(self.seidel_x3_card)
        seidel_cards.addWidget(self.seidel_x4_card)

        layout.addLayout(seidel_cards)

        table_sweep_title = QLabel("Таблица прогонки")
        table_sweep_title.setObjectName("BlockTitle")
        layout.addWidget(table_sweep_title)

        sweep_panel = SectionPanel()
        sweep_layout = QHBoxLayout(sweep_panel)
        sweep_layout.setContentsMargins(18, 18, 18, 18)
        sweep_layout.addStretch()

        self.sweep_table = AutoTableWidget()

        sweep_layout.addWidget(self.sweep_table)
        sweep_layout.addStretch()

        layout.addWidget(sweep_panel)

        table_seidel_title = QLabel("Таблица итераций метода Зейделя")
        table_seidel_title.setObjectName("BlockTitle")
        layout.addWidget(table_seidel_title)

        seidel_panel = SectionPanel()
        seidel_layout = QHBoxLayout(seidel_panel)
        seidel_layout.setContentsMargins(18, 18, 18, 18)
        seidel_layout.addStretch()

        self.seidel_table = AutoTableWidget()

        seidel_layout.addWidget(self.seidel_table)
        seidel_layout.addStretch()

        layout.addWidget(seidel_panel)

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
        self.vector_input.set_matrix(DEFAULT_B.reshape(4, 1))
        self.calculate()

    def calculate(self):
        try:
            A = self.matrix_input.get_matrix()
            b = self.vector_input.get_matrix().flatten()
            eps = self.eps_spin.value()

            x_sweep, sweep_table = thomas_method(A, b)
            x_seidel, seidel_table = gauss_seidel_method(A, b, eps)

            self.sweep_x1_card.set_value(f"{x_sweep[0]:.6f}")
            self.sweep_x2_card.set_value(f"{x_sweep[1]:.6f}")
            self.sweep_x3_card.set_value(f"{x_sweep[2]:.6f}")
            self.sweep_x4_card.set_value(f"{x_sweep[3]:.6f}")

            self.seidel_x1_card.set_value(f"{x_seidel[0]:.6f}")
            self.seidel_x2_card.set_value(f"{x_seidel[1]:.6f}")
            self.seidel_x3_card.set_value(f"{x_seidel[2]:.6f}")
            self.seidel_x4_card.set_value(f"{x_seidel[3]:.6f}")

            self.sweep_table.fill_from_dataframe(sweep_table)
            self.seidel_table.fill_from_dataframe(seidel_table)

            self.canvas.plot_iterations(seidel_table)

        except Exception as error:
            error_table = pd.DataFrame([{
                "Ошибка": str(error)
            }])
            self.seidel_table.fill_from_dataframe(error_table)