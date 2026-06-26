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


DEFAULT_X = [0.110, 0.116, 0.122, 0.128, 0.134, 0.140]
DEFAULT_Y = [8.65729, 8.29329, 7.95829, 7.64893, 7.36235, 7.09613]


def build_cyclic_spline(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    h = np.diff(x)
    n = len(x)

    if np.any(h <= 0):
        raise ValueError("Узлы x_i должны идти строго по возрастанию.")

    matrix = np.zeros((n, n))
    rhs = np.zeros(n)

    row = 0

    # Внутренние условия гладкости
    for i in range(1, n - 1):
        matrix[row][i - 1] = h[i - 1]
        matrix[row][i] = 2 * (h[i - 1] + h[i])
        matrix[row][i + 1] = h[i]

        rhs[row] = 6 * (
            (y[i + 1] - y[i]) / h[i]
            - (y[i] - y[i - 1]) / h[i - 1]
        )

        row += 1

    # Условие из задания: S''(x1) = S''(x6)
    matrix[row][0] = 1
    matrix[row][n - 1] = -1
    rhs[row] = 0
    row += 1

    # Дополнительное условие замкнутого сплайна: S'(x1) = S'(x6)
    delta = np.diff(y) / h

    matrix[row][0] = -2 * h[0] / 6
    matrix[row][1] = -h[0] / 6
    matrix[row][n - 2] = -h[-1] / 6
    matrix[row][n - 1] = -2 * h[-1] / 6
    rhs[row] = delta[-1] - delta[0]

    M = np.linalg.solve(matrix, rhs)

    rows = []

    for i in range(n - 1):
        a = y[i]
        b = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * M[i] + M[i + 1]) / 6
        c = M[i] / 2
        d = (M[i + 1] - M[i]) / (6 * h[i])

        rows.append({
            "i": i + 1,
            "interval": f"[{x[i]:.3f}; {x[i + 1]:.3f}]",
            "a_i": a,
            "b_i": b,
            "c_i": c,
            "d_i": d
        })

    coefficient_table = pd.DataFrame(rows)

    return M, coefficient_table


def spline_value(x_value, x_nodes, coefficient_table):
    x_nodes = np.array(x_nodes, dtype=float)

    for i in range(len(x_nodes) - 1):
        if x_nodes[i] <= x_value <= x_nodes[i + 1]:
            dx = x_value - x_nodes[i]

            a = coefficient_table.loc[i, "a_i"]
            b = coefficient_table.loc[i, "b_i"]
            c = coefficient_table.loc[i, "c_i"]
            d = coefficient_table.loc[i, "d_i"]

            return a + b * dx + c * dx ** 2 + d * dx ** 3

    if abs(x_value - x_nodes[-1]) < 1e-12:
        i = len(x_nodes) - 2
        dx = x_value - x_nodes[i]

        a = coefficient_table.loc[i, "a_i"]
        b = coefficient_table.loc[i, "b_i"]
        c = coefficient_table.loc[i, "c_i"]
        d = coefficient_table.loc[i, "d_i"]

        return a + b * dx + c * dx ** 2 + d * dx ** 3

    raise ValueError("x вне области построения сплайна.")


def create_m_table(M):
    rows = []

    for i, value in enumerate(M, start=1):
        rows.append({
            "i": i,
            "M_i": value
        })

    return pd.DataFrame(rows)


def spline_formulas_text(x_nodes, coefficient_table):
    lines = []

    for i in range(len(coefficient_table)):
        x0 = x_nodes[i]

        a = coefficient_table.loc[i, "a_i"]
        b = coefficient_table.loc[i, "b_i"]
        c = coefficient_table.loc[i, "c_i"]
        d = coefficient_table.loc[i, "d_i"]

        line = (
            f"S{i + 1}(x) = {a:.6f} "
            f"{b:+.6f}(x - {x0:.3f}) "
            f"{c:+.6f}(x - {x0:.3f})² "
            f"{d:+.6f}(x - {x0:.3f})³"
        )

        lines.append(line)

    return "<br><br>".join(lines)


class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(7, 3.4), dpi=100)
        self.axes = self.figure.add_subplot(111)

        super().__init__(self.figure)

        self.figure.patch.set_facecolor(PANEL)
        self.axes.set_facecolor(PANEL)

    def plot_spline(self, x_nodes, y_nodes, coefficient_table):
        self.axes.clear()

        x_min = min(x_nodes)
        x_max = max(x_nodes)

        x_plot = np.linspace(x_min, x_max, 500)
        y_plot = [spline_value(x, x_nodes, coefficient_table) for x in x_plot]

        self.axes.plot(
            x_plot,
            y_plot,
            color=ACCENT_SOFT,
            linewidth=2.5,
            label="Кубический сплайн"
        )

        self.axes.scatter(
            x_nodes,
            y_nodes,
            s=70,
            color="white",
            label="Узлы таблицы",
            zorder=3
        )

        self.axes.set_title("Кубический сплайн дефекта 1", color="white", fontsize=13, pad=12)
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


class Task6Page(QWidget):
    def __init__(self):
        super().__init__()

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(18)

        title = QLabel("Задание 6 — Кубические сплайны")
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
                Для таблицы задания №5 выписать кубические сплайны дефекта 1
                на каждом отрезке [x<sub>i−1</sub>; x<sub>i</sub>].
                Условие: S″(x<sub>1</sub>) = S″(x<sub>6</sub>).
            </span>
        """)

        formulas_text = QLabel()
        formulas_text.setWordWrap(True)
        formulas_text.setTextFormat(Qt.RichText)
        formulas_text.setText("""
            <span style="font-size:18px; color:white; font-weight:600;">
                Сплайн на каждом отрезке ищется в виде:
                S<sub>i</sub>(x) = a<sub>i</sub> + b<sub>i</sub>(x − x<sub>i</sub>)
                + c<sub>i</sub>(x − x<sub>i</sub>)² + d<sub>i</sub>(x − x<sub>i</sub>)³.
                Для однозначного построения используется замкнутый сплайн.
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

        self.m1_card = ResultCard("M₁")
        self.m6_card = ResultCard("M₆")
        self.check_card = ResultCard("M₁ − M₆")

        cards_layout.addWidget(self.m1_card)
        cards_layout.addWidget(self.m6_card)
        cards_layout.addWidget(self.check_card)

        layout.addLayout(cards_layout)

        m_title = QLabel("Вторые производные")
        m_title.setObjectName("BlockTitle")
        layout.addWidget(m_title)

        m_panel = SectionPanel()
        m_layout = QHBoxLayout(m_panel)
        m_layout.setContentsMargins(18, 18, 18, 18)
        m_layout.addStretch()

        self.m_table = AutoTableWidget()

        m_layout.addWidget(self.m_table)
        m_layout.addStretch()

        layout.addWidget(m_panel)

        coeff_title = QLabel("Таблица коэффициентов сплайнов")
        coeff_title.setObjectName("BlockTitle")
        layout.addWidget(coeff_title)

        coeff_panel = SectionPanel()
        coeff_layout = QHBoxLayout(coeff_panel)
        coeff_layout.setContentsMargins(18, 18, 18, 18)
        coeff_layout.addStretch()

        self.coeff_table = AutoTableWidget()

        coeff_layout.addWidget(self.coeff_table)
        coeff_layout.addStretch()

        layout.addWidget(coeff_panel)

        formulas_panel = SectionPanel()
        formulas_layout = QVBoxLayout(formulas_panel)
        formulas_layout.setContentsMargins(18, 18, 18, 18)

        formulas_title = QLabel("Кубические сплайны")
        formulas_title.setObjectName("BlockTitle")

        self.spline_formulas_label = QLabel()
        self.spline_formulas_label.setWordWrap(True)
        self.spline_formulas_label.setTextFormat(Qt.RichText)

        formulas_layout.addWidget(formulas_title)
        formulas_layout.addWidget(self.spline_formulas_label)

        layout.addWidget(formulas_panel)

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

            M, coefficient_table = build_cyclic_spline(x_nodes, y_nodes)
            m_table = create_m_table(M)

            self.m1_card.set_value(M[0])
            self.m6_card.set_value(M[-1])
            self.check_card.set_value(M[0] - M[-1])

            self.m_table.fill_from_dataframe(m_table)
            self.coeff_table.fill_from_dataframe(coefficient_table)

            text = spline_formulas_text(x_nodes, coefficient_table)

            self.spline_formulas_label.setText(f"""
                <span style="font-size:18px; color:white; font-weight:600;">
                    {text}
                </span>
            """)

            self.canvas.plot_spline(x_nodes, y_nodes, coefficient_table)

        except Exception as error:
            self.spline_formulas_label.setText(f"""
                <span style="font-size:18px; color:white; font-weight:700;">
                    Ошибка: {error}
                </span>
            """)