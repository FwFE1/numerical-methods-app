import numpy as np
import pandas as pd

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QFrame,
    QScrollArea,
)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ui_components import (
    SectionPanel,
    ResultCard,
    MatrixDisplayWidget,
    MatrixInputWidget,
    AutoTableWidget,
)


ACCENT = "#ff008f"
ACCENT_SOFT = "#e04c91"
PANEL = "#111111"


def off_diagonal_norm(matrix):
    n = matrix.shape[0]
    s = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            s += matrix[i][j] ** 2

    return np.sqrt(s)


def max_off_diagonal_element(matrix):
    n = matrix.shape[0]

    max_value = 0.0
    p = 0
    q = 1

    for i in range(n):
        for j in range(i + 1, n):
            if abs(matrix[i][j]) > abs(max_value):
                max_value = matrix[i][j]
                p = i
                q = j

    return p, q, max_value


def rotation_method(matrix, eps):
    matrix = matrix.copy()
    n = matrix.shape[0]

    eigenvectors = np.eye(n)
    rows = []

    rows.append({
        "k": 0,
        "element": "—",
        "a_ij": "—",
        "phi": "—",
        "a11": matrix[0][0],
        "a22": matrix[1][1],
        "a33": matrix[2][2],
        "S": off_diagonal_norm(matrix)
    })

    k = 0

    while off_diagonal_norm(matrix) >= eps:
        k += 1

        p, q, a_pq = max_off_diagonal_element(matrix)

        if matrix[p][p] == matrix[q][q]:
            phi = np.pi / 4
        else:
            phi = 0.5 * np.arctan(2 * matrix[p][q] / (matrix[p][p] - matrix[q][q]))

        rotation = np.eye(n)

        rotation[p][p] = np.cos(phi)
        rotation[q][q] = np.cos(phi)
        rotation[p][q] = -np.sin(phi)
        rotation[q][p] = np.sin(phi)

        matrix = rotation.T @ matrix @ rotation
        eigenvectors = eigenvectors @ rotation

        rows.append({
            "k": k,
            "element": f"a{p + 1}{q + 1}",
            "a_ij": a_pq,
            "phi": phi,
            "a11": matrix[0][0],
            "a22": matrix[1][1],
            "a33": matrix[2][2],
            "S": off_diagonal_norm(matrix)
        })

        if k > 100:
            raise RuntimeError("Метод не сошёлся за 100 итераций.")

    table = pd.DataFrame(rows)

    return np.diag(matrix), eigenvectors, matrix, table


class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(7, 3.4), dpi=100)
        self.axes = self.figure.add_subplot(111)

        super().__init__(self.figure)

        self.figure.patch.set_facecolor(PANEL)
        self.axes.set_facecolor(PANEL)

    def plot_s(self, table):
        self.axes.clear()

        k_values = table["k"].astype(float)
        s_values = table["S"].astype(float)

        self.axes.plot(
            k_values,
            s_values,
            marker="o",
            color=ACCENT_SOFT,
            linewidth=2.5,
            markersize=7,
        )

        self.axes.set_title("Уменьшение внедиагональных элементов", color="white", fontsize=13, pad=12)
        self.axes.set_xlabel("Номер итерации k", color="white")
        self.axes.set_ylabel("S", color="white")

        self.axes.grid(True, color="#333333", linewidth=0.8)

        self.axes.tick_params(axis="x", colors="white")
        self.axes.tick_params(axis="y", colors="white")

        for spine in self.axes.spines.values():
            spine.set_color("#555555")

        self.figure.tight_layout()
        self.draw()


class Task3Page(QWidget):
    def __init__(self):
        super().__init__()

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(18)

        title = QLabel("Задание 3 — Метод вращения")
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
                Методом вращения с точностью ε = 0.01 вычислить собственные значения
                и собственные векторы симметричной матрицы.
            </span>
        """)

        symbolic_matrix = MatrixDisplayWidget("A =", np.array([
            ["50 + 3n", "10 - n", "3"],
            ["10 - n", "20 + 2n", "10 - n"],
            ["3", "10 - n", "90 - n"],
        ], dtype=object))

        criterion = QLabel()
        criterion.setWordWrap(True)
        criterion.setTextFormat(Qt.RichText)
        criterion.setText("""
            <span style="font-size:18px; color:white; font-weight:600;">
                Критерий остановки: S = √Σa<sub>ij</sub><sup>2</sup> &lt; ε,
                где сумма берётся по внедиагональным элементам.
            </span>
        """)

        condition_layout.addWidget(condition_title)
        condition_layout.addWidget(condition_text)
        condition_layout.addWidget(symbolic_matrix)
        condition_layout.addWidget(criterion)

        layout.addWidget(condition_panel)

        input_panel = SectionPanel()
        input_layout = QVBoxLayout(input_panel)
        input_layout.setContentsMargins(18, 18, 18, 18)
        input_layout.setSpacing(12)

        input_title = QLabel("Исходные данные")
        input_title.setObjectName("BlockTitle")
        input_layout.addWidget(input_title)

        input_row = QHBoxLayout()

        self.n_spin = QSpinBox()
        self.n_spin.setMinimum(-100)
        self.n_spin.setMaximum(100)
        self.n_spin.setValue(8)

        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setDecimals(6)
        self.eps_spin.setMinimum(0.000001)
        self.eps_spin.setMaximum(10)
        self.eps_spin.setValue(0.01)
        self.eps_spin.setSingleStep(0.01)

        update_button = QPushButton("Заполнить матрицу по n")
        update_button.setObjectName("SecondaryButton")
        update_button.clicked.connect(self.fill_matrix_by_variant)

        calculate_button = QPushButton("Рассчитать")
        calculate_button.setObjectName("ActionButton")
        calculate_button.clicked.connect(self.calculate)

        n_label = QLabel("n:")
        n_label.setStyleSheet("color: white; font-size: 19px; font-weight: 800;")

        eps_label = QLabel("ε:")
        eps_label.setStyleSheet("color: white; font-size: 19px; font-weight: 800;")

        self.n_spin.setMinimumWidth(130)
        self.eps_spin.setMinimumWidth(170)

        input_row.addWidget(n_label)
        input_row.addWidget(self.n_spin)
        input_row.addWidget(eps_label)
        input_row.addWidget(self.eps_spin)
        input_row.addWidget(update_button)
        input_row.addWidget(calculate_button)
        input_row.addStretch()

        input_layout.addLayout(input_row)

        hint = QLabel("Матрица изначально заполнена по варианту 8. Значения можно изменить вручную.")
        hint.setObjectName("SmallHint")
        input_layout.addWidget(hint)

        self.matrix_input = MatrixInputWidget(3, 3, "A =")
        input_layout.addWidget(self.matrix_input, alignment=Qt.AlignCenter)

        layout.addWidget(input_panel)

        result_title = QLabel("Результат")
        result_title.setObjectName("BlockTitle")
        layout.addWidget(result_title)

        result_cards_layout = QHBoxLayout()

        self.lambda1_card = ResultCard("λ₁")
        self.lambda2_card = ResultCard("λ₂")
        self.lambda3_card = ResultCard("λ₃")

        result_cards_layout.addWidget(self.lambda1_card)
        result_cards_layout.addWidget(self.lambda2_card)
        result_cards_layout.addWidget(self.lambda3_card)

        layout.addLayout(result_cards_layout)

        matrix_result_panel = SectionPanel()
        matrix_result_layout = QVBoxLayout(matrix_result_panel)
        matrix_result_layout.setContentsMargins(18, 18, 18, 18)
        matrix_result_layout.setSpacing(16)

        matrix_result_title = QLabel("Матрица после вращений и собственные векторы")
        matrix_result_title.setObjectName("BlockTitle")

        self.diagonal_matrix_widget = MatrixDisplayWidget("A⁽ᵏ⁾ =")
        self.eigenvectors_matrix_widget = MatrixDisplayWidget("V =")

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(Qt.RichText)

        matrix_result_layout.addWidget(matrix_result_title)
        matrix_result_layout.addWidget(self.diagonal_matrix_widget, alignment=Qt.AlignCenter)

        vectors_title = QLabel("Собственные векторы являются столбцами матрицы V")
        vectors_title.setAlignment(Qt.AlignCenter)
        vectors_title.setStyleSheet("""
            color: white;
            font-size: 19px;
            font-weight: 800;
            background: transparent;
        """)

        matrix_result_layout.addWidget(vectors_title)
        matrix_result_layout.addWidget(self.eigenvectors_matrix_widget, alignment=Qt.AlignCenter)
        matrix_result_layout.addWidget(self.summary_label)

        layout.addWidget(matrix_result_panel)

        table_title = QLabel("Таблица итераций")
        table_title.setObjectName("BlockTitle")
        layout.addWidget(table_title)

        table_panel = SectionPanel()
        table_layout = QHBoxLayout(table_panel)
        table_layout.setContentsMargins(18, 18, 18, 18)
        table_layout.addStretch()

        self.iteration_table = AutoTableWidget()

        table_layout.addWidget(self.iteration_table, alignment=Qt.AlignCenter)
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

        self.fill_matrix_by_variant()
        self.calculate()

    def fill_matrix_by_variant(self):
        n = self.n_spin.value()

        matrix = np.array([
            [50 + 3 * n, 10 - n, 3],
            [10 - n, 20 + 2 * n, 10 - n],
            [3, 10 - n, 90 - n]
        ], dtype=float)

        self.matrix_input.set_matrix(matrix)

    def calculate(self):
        try:
            matrix = self.matrix_input.get_matrix()
            eps = self.eps_spin.value()

            eigenvalues, eigenvectors, diagonal_matrix, table = rotation_method(matrix, eps)

            self.lambda1_card.set_value(eigenvalues[0])
            self.lambda2_card.set_value(eigenvalues[1])
            self.lambda3_card.set_value(eigenvalues[2])

            self.diagonal_matrix_widget.set_matrix(diagonal_matrix)
            self.eigenvectors_matrix_widget.set_matrix(eigenvectors)

            self.summary_label.setText(f"""
                <span style="font-size:18px; color:white; font-weight:600;">
                    Последнее значение: S = {table.iloc[-1]['S']:.6f}. 
                    Заданная точность: ε = {eps:.6f}. 
                    Так как S &lt; ε, процесс остановлен.
                </span>
            """)

            self.iteration_table.fill_from_dataframe(table)
            self.canvas.plot_s(table)

        except Exception as error:
            self.summary_label.setText(f"""
                <span style="color:#ff4d4d; font-size:17px; font-weight:900;">
                    Ошибка: {error}
                </span>
            """)