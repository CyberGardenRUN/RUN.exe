# -*- coding: utf-8 -*-
"""
RTLS BLE (ESP32 beacons) — robust real-time trilateration + calibration GUI
Author: you

Requirements:
  pip install pyqt5 pyserial matplotlib numpy

Expected beacon line (preferred):
  ANCHOR:A1 RSSI:-67

Fallback accepted (not recommended on anchors):
  Distance: 2.345 meters  RSSI: -67
"""

import sys, os, json, re, time
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, QThread

import serial
from collections import deque

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# --------------------- CONST / RE ---------------------

BAUDRATE = 115200
SER_TIMEOUT = 0.01        # 10 ms — минимальная задержка на порт
SKIP_FIRST_N = 20         # пропуск первых строк на порту (прогрев BLE)
TTL_SEC = 0.50            # "время жизни" измерения маяка (сек)
UPDATE_MS = 40            # период отрисовки/оценки (мс) ~25 Гц

# фильтрация расстояния
MEDIAN_WIN = 5
EMA_ALPHA = 0.35
MAX_JUMP = 1.0            # м, максимально допустимый скачок/кадр
D_RANGE = (0.1, 30.0)     # допустимый диапазон дистанций (м)

# сглаживание позиции (alpha-beta; можно выключить beta=0)
POS_ALPHA = 0.35
POS_BETA  = 0.10

# сцена (рамки)
SCENE_MIN = np.array([-20.0, -20.0, 0.0], float)
SCENE_MAX = np.array([ 20.0,  20.0, 20.0], float)

# regexp для парсинга портовых строк
RE_RSSI_LINE = re.compile(r"ANCHOR:([A-Za-z0-9_]+)\s+RSSI:\s*(-?\d+)", re.I)
RE_DISTANCE  = re.compile(r"Distance(?:\s*\(.*?\))?:\s*([0-9]+(?:[.,][0-9]+)?)\s*meters?", re.I)
RE_RSSI      = re.compile(r"RSSI:\s*(-?\d+)", re.I)


# --------------------- МОДЕЛЬ RSSI→D ---------------------

def rssi_to_distance(rssi, tx1m, n):
    """
    Log-distance path loss model:
      d = 10 ** ((Tx@1m - RSSI) / (10*n))
    """
    return float(10 ** ((float(tx1m) - float(rssi)) / (10.0 * float(n))))


def fit_tx_n_from_pairs(pairs):
    """
    Оценка (tx1m, n) по списку пар (rssi, d_true).
    Из модели: RSSI = tx - 10*n*log10(d) => линейная регрессия:
      RSSI = a + b * log10(d), где a=tx, b=-10*n
    """
    if not pairs or len(pairs) < 2:
        return None
    rssi = np.array([p[0] for p in pairs], float)
    d    = np.array([max(1e-6, p[1]) for p in pairs], float)
    X = np.column_stack([np.ones_like(d), np.log10(d)])
    # rssi ~ X @ coeff, coeff = [a, b]
    coeff, *_ = np.linalg.lstsq(X, rssi, rcond=None)
    tx = float(coeff[0])
    n  = float(-coeff[1] / 10.0)
    if not np.isfinite(tx) or not np.isfinite(n):
        return None
    # разумные рамки
    if not (-100.0 <= tx <= -20.0):  # Tx@1m обычно -70..-40
        return None
    if not (1.2 <= n <= 4.0):
        return None
    return tx, n


# --------------------- ФИЛЬТРЫ ---------------------

class DistanceFilter:
    """
    Пер-маячный фильтр: медиана-окно + EMA; отсев по диапазону и скачку.
    """
    def __init__(self, median_win=5, alpha=0.35, d_range=(0.1,30.0), max_jump=1.0):
        self.buf = deque(maxlen=max(1, int(median_win)))
        self.alpha = float(alpha)
        self.ema = None
        self.dmin, self.dmax = float(d_range[0]), float(d_range[1])
        self.max_jump = float(max_jump)
        self.prev_accepted = None

    def push_raw(self, d):
        # диапазон
        if not (self.dmin <= d <= self.dmax):
            return None
        # отсев по скачку (мягкий)
        if self.prev_accepted is not None:
            jump = d - self.prev_accepted
            if abs(jump) > self.max_jump:
                d = self.prev_accepted + np.clip(jump, -self.max_jump, self.max_jump)
        self.prev_accepted = d
        # медиана
        self.buf.append(d)
        med = float(np.median(self.buf))
        # EMA
        self.ema = med if (self.ema is None) else (self.alpha * med + (1.0 - self.alpha) * self.ema)
        return self.ema


class PosAlphaBeta:
    """
    3D alpha-beta (g-h) фильтр для позиции/скорости.
    Очень лёгкий и эффективный на «неровной» оценке.
    """
    def __init__(self, alpha=0.35, beta=0.10):
        self.alpha = float(alpha)
        self.beta  = float(beta)
        self.x = None  # позиция (3,)
        self.v = np.zeros(3, float)
        self.t_prev = None

    def reset(self):
        self.x = None
        self.v = np.zeros(3, float)
        self.t_prev = None

    def update(self, z, t=None):
        """
        z — измеренная позиция (3,), t — текущее время (сек).
        """
        now = time.time() if t is None else t
        if self.x is None:
            self.x = np.array(z, float)
            self.v = np.zeros(3, float)
            self.t_prev = now
            return self.x
        dt = max(1e-3, now - self.t_prev)
        self.t_prev = now

        # прогноз
        x_pred = self.x + self.v * dt
        # инновация
        r = np.array(z, float) - x_pred
        # коррекция
        self.x = x_pred + self.alpha * r
        self.v = self.v + (self.beta * r / dt)
        return self.x


# --------------------- ТРИЛАТЕРАЦИЯ ---------------------

def trilaterate_robust(anchors, dists, x0=None, iters=12, damp=0.6, huber=0.4):
    """
    Робастный Гаусс–Ньютон для min Σ ρ(||x - ai|| - di), ρ — Huber.
    anchors: Nx3, dists: N
    """
    A = np.asarray(anchors, float)
    d = np.asarray(dists, float)
    N = len(d)
    if N < 3:
        return None

    x = np.array(A.mean(axis=0) if x0 is None else x0, float)
    for _ in range(iters):
        vec = A - x
        r   = np.linalg.norm(vec, axis=1) - d  # residuals
        # веса Huber
        w = np.ones(N)
        big = np.abs(r) > huber
        w[big] = huber / (np.abs(r[big]) + 1e-9)
        # Якобиан dr/dx = (x - ai)/||x-ai||
        J = (-vec) / (np.linalg.norm(vec, axis=1)[:, None] + 1e-9)
        # взвешенная система
        JT_W = (J.T * w)
        H = JT_W @ J + 1e-6 * np.eye(3)
        g = JT_W @ r
        try:
            dx = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        x = x + damp * dx
    return x


# --------------------- СЕРИЙНЫЕ ПОТОКИ ---------------------

class SerialReader(QThread):
    newRSSI = pyqtSignal(int, object, float, str)   # (beacon_id, rssi_or_None, timestamp, raw_line)
    portStatus = pyqtSignal(str)

    def __init__(self, beacon_id, port, skip_first=SKIP_FIRST_N):
        super().__init__()
        self.beacon_id = beacon_id
        self.port_name = port
        self.skipN = int(skip_first)
        self.ser = None
        self._run = True
        self._skipped = 0

    def run(self):
        try:
            self.ser = serial.Serial(self.port_name, BAUDRATE, timeout=SER_TIMEOUT)
            self.portStatus.emit(f"Открыт {self.port_name} для маяка {self.beacon_id}")
        except Exception as e:
            self.portStatus.emit(f"Ошибка открытия {self.port_name}: {e}")
            return

        while self._run:
            try:
                line = self.ser.readline().decode('latin-1', errors='ignore').strip()
                if not line:
                    continue
                if self._skipped < self.skipN:
                    self._skipped += 1
                    continue

                ts = time.time()

                # Предпочтительный формат
                mR = RE_RSSI_LINE.search(line)
                if mR:
                    try:
                        rssi = int(mR.group(2))
                    except:
                        rssi = None
                    self.newRSSI.emit(self.beacon_id, rssi, ts, line)
                    continue

                # Фоллбек: Distance ... meters
                mD = RE_DISTANCE.search(line)
                if mD:
                    # RSSI опционально
                    m2 = RE_RSSI.search(line)
                    rssi = int(m2.group(1)) if m2 else None
                    self.newRSSI.emit(self.beacon_id, rssi, ts, line)
                    continue

                # прочее игнорируем
            except Exception as e:
                self.portStatus.emit(f"[{self.port_name}] ошибка чтения: {e}")
                time.sleep(0.03)

        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
                self.portStatus.emit(f"Закрыт {self.port_name}")
        except:
            pass

    def stop(self):
        self._run = False


# --------------------- MPL CANVAS ---------------------

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, show_spheres=False):
        self.fig = Figure(constrained_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.show_spheres = show_spheres
        self._last_sphere_time = 0.0

    def draw_scene(self, anchors, tracker_pos, dists=None, autoscale=True):
        self.ax.cla()

        # маяки
        if anchors:
            arr = np.array(anchors)[:, :3]
            self.ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=80, marker='^', label='Beacons')

        # сферы радиусов (дорогая операция — рисуем не чаще 5 Гц)
        if self.show_spheres and anchors and dists:
            now = time.time()
            if now - self._last_sphere_time > 0.2:
                self._last_sphere_time = now
                try:
                    for (x, y, z, _bid), d in zip(anchors, dists):
                        # редуцированный "сферический" вид — трёх орт. окружностей
                        u = np.linspace(0, 2 * np.pi, 60)
                        self.ax.plot(x + d * np.cos(u), y + d * np.sin(u), z, linewidth=0.5)
                        self.ax.plot(x + d * np.cos(u), y, z + d * np.sin(u), linewidth=0.5)
                        self.ax.plot(x, y + d * np.cos(u), z + d * np.sin(u), linewidth=0.5)
                except Exception:
                    pass

        # трекер
        if tracker_pos is not None:
            self.ax.scatter([tracker_pos[0]], [tracker_pos[1]], [tracker_pos[2]], s=90, label='Tracker')

        # оси
        if autoscale:
            pts = []
            if anchors:
                pts.extend([a[:3] for a in anchors])
            if tracker_pos is not None:
                pts.append(tracker_pos)
            if pts:
                P = np.array(pts)
                mn = np.min(P, axis=0) - 0.5
                mx = np.max(P, axis=0) + 0.5
                self.ax.set_xlim(mn[0], mx[0])
                self.ax.set_ylim(mn[1], mx[1])
                self.ax.set_zlim(max(0.0, mn[2]), mx[2])  # z не уводим ниже 0
        else:
            self.ax.set_xlim(SCENE_MIN[0], SCENE_MAX[0])
            self.ax.set_ylim(SCENE_MIN[1], SCENE_MAX[1])
            self.ax.set_zlim(SCENE_MIN[2], SCENE_MAX[2])

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend(loc='upper right')
        self.draw()


# --------------------- MAIN WINDOW ---------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ESP32 RTLS — Robust Trilateration + Calibration")
        self.resize(1400, 820)

        # ==== Таблица маяков ====
        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["ON", "ID", "COM", "X", "Y", "Z", "Tx@1m", "n"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)

        # ==== Кнопки таблицы ====
        self.btnAdd = QtWidgets.QPushButton("Добавить маяк")
        self.btnDel = QtWidgets.QPushButton("Удалить выбранный")

        # ==== Управление ====
        self.btnStart = QtWidgets.QPushButton("Старт")
        self.btnStop  = QtWidgets.QPushButton("Стоп"); self.btnStop.setEnabled(False)
        self.btnSave  = QtWidgets.QPushButton("Сохранить конфиг")
        self.btnLoad  = QtWidgets.QPushButton("Загрузить конфиг")

        # ==== Фильтр ====
        filtBox = QtWidgets.QGroupBox("Фильтрация расстояний")
        self.spMedian = QtWidgets.QSpinBox(); self.spMedian.setRange(1, 99); self.spMedian.setValue(MEDIAN_WIN)
        self.spAlpha  = QtWidgets.QDoubleSpinBox(); self.spAlpha.setRange(0.0, 1.0); self.spAlpha.setDecimals(2); self.spAlpha.setSingleStep(0.05); self.spAlpha.setValue(EMA_ALPHA)
        self.edDmin   = QtWidgets.QDoubleSpinBox(); self.edDmin.setRange(0.0, 1000.0); self.edDmin.setValue(D_RANGE[0]); self.edDmin.setDecimals(2)
        self.edDmax   = QtWidgets.QDoubleSpinBox(); self.edDmax.setRange(0.1, 1000.0); self.edDmax.setValue(D_RANGE[1]); self.edDmax.setDecimals(2)
        self.edJump   = QtWidgets.QDoubleSpinBox(); self.edJump.setRange(0.0, 10.0); self.edJump.setValue(MAX_JUMP); self.edJump.setDecimals(2)
        self.spSkip   = QtWidgets.QSpinBox(); self.spSkip.setRange(0, 200); self.spSkip.setValue(SKIP_FIRST_N)
        self.spTTL    = QtWidgets.QDoubleSpinBox(); self.spTTL.setRange(0.05, 5.0); self.spTTL.setValue(TTL_SEC); self.spTTL.setSingleStep(0.05)
        formF = QtWidgets.QFormLayout(filtBox)
        formF.addRow("Окно медианы:", self.spMedian)
        formF.addRow("EMA α:", self.spAlpha)
        formF.addRow("Диапазон d min/max (м):", self._h(self.edDmin, self.edDmax))
        formF.addRow("Скачок max (м):", self.edJump)
        formF.addRow("Пропустить первые N строк:", self.spSkip)
        formF.addRow("TTL маяка (с):", self.spTTL)

        # ==== Позиционный фильтр ====
        posBox = QtWidgets.QGroupBox("Сглаживание позиции (alpha-beta)")
        self.spPosAlpha = QtWidgets.QDoubleSpinBox(); self.spPosAlpha.setRange(0.0, 1.0); self.spPosAlpha.setValue(POS_ALPHA); self.spPosAlpha.setDecimals(2)
        self.spPosBeta  = QtWidgets.QDoubleSpinBox(); self.spPosBeta.setRange(0.0, 1.0); self.spPosBeta.setValue(POS_BETA); self.spPosBeta.setDecimals(2)
        formP = QtWidgets.QFormLayout(posBox)
        formP.addRow("α:", self.spPosAlpha)
        formP.addRow("β:", self.spPosBeta)

        # ==== Калибровка ====
        calBox = QtWidgets.QGroupBox("Калибровка (RSSI → distance)")
        self.cbApplyCal = QtWidgets.QCheckBox("Применять калибровку"); self.cbApplyCal.setChecked(True)

        # точки
        self.edPX = QtWidgets.QDoubleSpinBox(); self.edPY = QtWidgets.QDoubleSpinBox(); self.edPZ = QtWidgets.QDoubleSpinBox()
        for w in (self.edPX, self.edPY, self.edPZ):
            w.setRange(-1e4, 1e4); w.setDecimals(3)
        self.btnSnap = QtWidgets.QPushButton("Снять срез (точка)")
        self.btnSolveTxN = QtWidgets.QPushButton("Посчитать tx@1м, n по точкам")
        self.btnClearCal = QtWidgets.QPushButton("Сбросить набор точек")

        # A→B
        self.edAx = QtWidgets.QDoubleSpinBox(); self.edAy = QtWidgets.QDoubleSpinBox(); self.edAz = QtWidgets.QDoubleSpinBox()
        self.edBx = QtWidgets.QDoubleSpinBox(); self.edBy = QtWidgets.QDoubleSpinBox(); self.edBz = QtWidgets.QDoubleSpinBox()
        for w in (self.edAx, self.edAy, self.edAz, self.edBx, self.edBy, self.edBz):
            w.setRange(-1e4, 1e4); w.setDecimals(3)
        self.btnPathStart = QtWidgets.QPushButton("Старт запись A→B")
        self.btnPathStop  = QtWidgets.QPushButton("Стоп запись"); self.btnPathStop.setEnabled(False)
        self.btnSolvePath = QtWidgets.QPushButton("Посчитать tx@1м, n по A→B")

        layC = QtWidgets.QFormLayout(calBox)
        layC.addRow(self.cbApplyCal)
        layC.addRow("Точка (X,Y,Z):", self._h(self.edPX, self.edPY, self.edPZ, self.btnSnap))
        layC.addRow("", self._h(self.btnSolveTxN, self.btnClearCal))
        layC.addRow("A (x,y,z):", self._h(self.edAx, self.edAy, self.edAz))
        layC.addRow("B (x,y,z):", self._h(self.edBx, self.edBy, self.edBz))
        layC.addRow("", self._h(self.btnPathStart, self.btnPathStop, self.btnSolvePath))

        # ==== Графика + опции ====
        self.cbSpheres = QtWidgets.QCheckBox("Показывать сферы радиусов"); self.cbSpheres.setChecked(False)
        self.cbAutoscale = QtWidgets.QCheckBox("Авто-масштаб сцены"); self.cbAutoscale.setChecked(True)
        self.canvas = MplCanvas(self, show_spheres=self.cbSpheres.isChecked())
        self.cbSpheres.stateChanged.connect(lambda _: setattr(self.canvas, "show_spheres", self.cbSpheres.isChecked()))

        # ==== Лог ====
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(4000)

        # ==== Компоновка ====
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.table)
        rowBtns = QtWidgets.QHBoxLayout(); rowBtns.addWidget(self.btnAdd); rowBtns.addWidget(self.btnDel)
        left.addLayout(rowBtns)
        left.addWidget(filtBox)
        left.addWidget(posBox)
        left.addWidget(calBox)
        left.addWidget(self._h(self.btnStart, self.btnStop, self.btnSave, self.btnLoad))
        left.addWidget(self.cbSpheres)
        left.addWidget(self.cbAutoscale)
        left.addWidget(self.log, 1)
        leftW = QtWidgets.QWidget(); leftW.setLayout(left)

        splitter = QtWidgets.QSplitter(); splitter.addWidget(leftW); splitter.addWidget(self.canvas); splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        # ==== Состояние ====
        self.threads = []                # SerialReader[]
        self.filters = {}                # bid -> DistanceFilter
        self.latest = {}                 # bid -> (dist_filtered, ts_last, rssi_last)
        self.anchors = {}                # bid -> (x,y,z, bid)
        self.txn = {}                    # bid -> (tx1m, n)
        self.calPairs = {}               # bid -> list[(rssi, d_true)]
        self.pathRecording = False
        self.pathT0 = None
        self.pathSamples = []            # list of (t, {bid: rssi})  (именно RSSI!)
        self.posFilter = PosAlphaBeta(self.spPosAlpha.value(), self.spPosBeta.value())
        self._last_pos = None
        self._last_update_time = 0.0

        # ==== Сигналы ====
        self.btnAdd.clicked.connect(self.add_row)
        self.btnDel.clicked.connect(self.delete_selected)
        self.btnStart.clicked.connect(self.start_readers)
        self.btnStop.clicked.connect(self.stop_readers)
        self.btnSave.clicked.connect(self.save_config)
        self.btnLoad.clicked.connect(self.load_config)

        self.btnSnap.clicked.connect(self.capture_point)
        self.btnSolveTxN.clicked.connect(self.solve_txn_from_points)
        self.btnClearCal.clicked.connect(self.clear_pairs)

        self.btnPathStart.clicked.connect(self.start_path_record)
        self.btnPathStop.clicked.connect(self.stop_path_record)
        self.btnSolvePath.clicked.connect(self.solve_txn_from_path)

        # таймер обновления
        self.tmr = QtCore.QTimer(self); self.tmr.setInterval(UPDATE_MS); self.tmr.timeout.connect(self.update_solution)

        # пример (как у вас)
        self.prefill_example()

        # статусбар
        self.status = self.statusBar()
        self._fps_t = time.time()
        self._fps_n = 0

    # ---------- Утилиты UI ----------
    def _h(self, *widgets):
        box = QtWidgets.QHBoxLayout()
        for w in widgets:
            box.addWidget(w if isinstance(w, QtWidgets.QWidget) else QtWidgets.QLabel(str(w)))
        cont = QtWidgets.QWidget(); cont.setLayout(box)
        return cont

    def log_line(self, s): self.log.appendPlainText(s)

    def prefill_example(self):
        # Пример ваших координат
        #   (0,10,0), (10,0,0), (-10,0,0), (0,0,5)
        self.add_row(True,  1, "COM8",  0, 10, 0, -59, 2.0)
        self.add_row(True,  2, "COM14", 10,  0, 0, -59, 2.0)
        self.add_row(True,  3, "COM15",-10,  0, 0, -59, 2.0)
        self.add_row(True,  4, "COM16",  0,  0, 5, -59, 2.2)

    def add_row(self, on=True, bid=None, port="COM6", x=0.0, y=0.0, z=0.0, tx=-59.0, n=2.0):
        r = self.table.rowCount()
        self.table.insertRow(r)
        # ON checkbox
        chk = QtWidgets.QTableWidgetItem()
        chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        chk.setCheckState(Qt.Checked if on else Qt.Unchecked)
        self.table.setItem(r, 0, chk)
        # ID
        it_id = QtWidgets.QTableWidgetItem(str(bid if bid is not None else r+1)); it_id.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(r, 1, it_id)
        # COM
        it_port = QtWidgets.QTableWidgetItem(port); it_port.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(r, 2, it_port)
        # X Y Z
        for c, val in enumerate([x, y, z], start=3):
            it = QtWidgets.QTableWidgetItem(str(val)); it.setTextAlignment(Qt.AlignCenter); self.table.setItem(r, c, it)
        # tx1m, n
        it_tx = QtWidgets.QTableWidgetItem(str(tx)); it_tx.setTextAlignment(Qt.AlignCenter); self.table.setItem(r, 6, it_tx)
        it_n  = QtWidgets.QTableWidgetItem(str(n));  it_n.setTextAlignment(Qt.AlignCenter); self.table.setItem(r, 7, it_n)

    def delete_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def gather_beacons(self):
        res = []
        for r in range(self.table.rowCount()):
            try:
                on = self.table.item(r, 0).checkState() == Qt.Checked
                bid = int(self.table.item(r, 1).text())
                port = self.table.item(r, 2).text().strip()
                x = float(self.table.item(r, 3).text())
                y = float(self.table.item(r, 4).text())
                z = float(self.table.item(r, 5).text())
                tx = float(self.table.item(r, 6).text())
                n  = float(self.table.item(r, 7).text())
            except Exception as e:
                self.log_line(f"Ошибка строки {r+1}: {e}")
                continue
            res.append((on, bid, port, (x, y, z), tx, n))
        return res

    # ---------- Старт/Стоп ----------
    def start_readers(self):
        self.stop_readers()

        self.filters.clear()
        self.latest.clear()
        self.anchors.clear()
        self.txn.clear()
        self.calPairs.clear()
        self.pathSamples.clear()
        self.posFilter = PosAlphaBeta(self.spPosAlpha.value(), self.spPosBeta.value())
        self._last_pos = None

        ids = set()
        for on, bid, port, pos, tx, n in self.gather_beacons():
            if not on: continue
            if bid in ids:
                self.log_line(f"Дублируется ID {bid} — пропуск")
                continue
            ids.add(bid)
            self.filters[bid] = DistanceFilter(
                self.spMedian.value(),
                self.spAlpha.value(),
                (self.edDmin.value(), self.edDmax.value()),
                self.edJump.value()
            )
            self.anchors[bid] = (float(pos[0]), float(pos[1]), float(pos[2]), bid)
            self.txn[bid] = (tx, n)
            th = SerialReader(bid, port, self.spSkip.value())
            th.newRSSI.connect(self.on_new_rssi)
            th.portStatus.connect(self.log_line)
            th.start()
            self.threads.append(th)

        if len(self.anchors) < 3:
            self.log_line("Нужно ≥3 включённых маяков.")
            return

        self.btnStart.setEnabled(False); self.btnStop.setEnabled(True)
        self.tmr.start()
        self.log_line("Старт опроса…")

    def stop_readers(self):
        self.tmr.stop()
        for th in self.threads:
            th.stop()
            th.wait(500)
        self.threads.clear()
        self.btnStart.setEnabled(True); self.btnStop.setEnabled(False)

    # ---------- Приём данных ----------
    @QtCore.pyqtSlot(int, object, float, str)
    def on_new_rssi(self, bid, rssi, ts, raw):
        """
        Преобразуем RSSI→distance (либо извлекаем дистанцию из фоллбека), фильтруем, копим.
        """
        tx, n = self.txn.get(bid, (-59.0, 2.0))
        d_meas = None

        if rssi is not None:
            # основной путь: RSSI→distance
            d_meas = rssi_to_distance(rssi, tx, n)
        else:
            # фоллбек: попробуем "Distance: X meters"
            mD = RE_DISTANCE.search(raw)
            if mD:
                try:
                    d_meas = float(mD.group(1).replace(',', '.'))
                except:
                    d_meas = None

        if d_meas is None:
            return

        filt = self.filters.get(bid)
        if not filt:
            return
        d_f = filt.push_raw(d_meas)
        if d_f is None:
            return

        self.latest[bid] = (float(d_f), float(ts), int(rssi) if rssi is not None else None)

        # запись для A→B — храним RSSI (если есть)
        if self.pathRecording and (rssi is not None):
            if not self.pathSamples:
                self.pathT0 = time.time()
            t = time.time() - self.pathT0
            # агрегируем по времени ~10мс
            if not self.pathSamples or abs(self.pathSamples[-1][0] - t) > 0.01:
                self.pathSamples.append((t, {bid: int(rssi)}))
            else:
                self.pathSamples[-1][1][bid] = int(rssi)

    # ---------- Оценка позиции + отрисовка ----------
    def update_solution(self):
        now = time.time()
        ttl = self.spTTL.value()

        # актуальные маяки
        valid = []
        for bid, (d, ts, _rssi) in self.latest.items():
            if now - ts <= ttl:
                valid.append((bid, d))
        if len(valid) < 3:
            # мало маяков — только маяки рисуем
            self.canvas.draw_scene(list(self.anchors.values()), None)
            return

        valid.sort(key=lambda x: x[0])  # по ID
        bids = [b for b, _ in valid]
        dists = [d for _, d in valid]
        anchors = [self.anchors[b] for b in bids]
        A_xyz = [a[:3] for a in anchors]

        # робастная триангуляция
        x0 = self._last_pos
        pos = trilaterate_robust(A_xyz, dists, x0=x0, iters=12, damp=0.6, huber=0.4)

        tracker = None
        if pos is not None and np.all(np.isfinite(pos)):
            # ограничиваем сценой
            pos = np.clip(pos, SCENE_MIN, SCENE_MAX)
            # альфа-бета сглаживание
            self.posFilter.alpha = self.spPosAlpha.value()
            self.posFilter.beta  = self.spPosBeta.value()
            pos_sm = self.posFilter.update(pos)
            tracker = tuple(map(float, pos_sm))
            self._last_pos = pos_sm

            # метрики качества (RMS резидуал)
            res = np.linalg.norm(np.array(A_xyz) - pos, axis=1) - np.array(dists)
            rms = float(np.sqrt(np.mean(res**2)))
            self.status.showMessage(f"Beacons used: {len(valid)} | RMS: {rms:.3f} m")

        self.canvas.draw_scene(
            [(a[0], a[1], a[2], a[3]) for a in anchors],
            tracker,
            dists=dists,
            autoscale=self.cbAutoscale.isChecked()
        )

        # fps
        self._fps_n += 1
        if now - self._fps_t >= 1.0:
            self.setWindowTitle(f"ESP32 RTLS — {self._fps_n} FPS")
            self._fps_t = now; self._fps_n = 0

    # ---------- Калибровка: точки ----------
    def capture_point(self):
        if not self.anchors:
            self.log_line("Нет маяков.")
            return
        x, y, z = self.edPX.value(), self.edPY.value(), self.edPZ.value()
        p = np.array([x, y, z], float)
        cnt = 0
        for bid, anc in self.anchors.items():
            anc_xyz = np.array(anc[:3], float)
            d_true = float(np.linalg.norm(p - anc_xyz))
            # берём последний RSSI (а не distance!)
            triple = self.latest.get(bid, None)
            if not triple:
                continue
            _d_f, ts, rssi = triple
            if rssi is None:
                continue
            self.calPairs.setdefault(bid, []).append((int(rssi), d_true))
            cnt += 1
        self.log_line(f"Срез точки ({x:.3f},{y:.3f},{z:.3f}) — пар добавлено: {cnt}")

    def solve_txn_from_points(self):
        if not self.calPairs:
            self.log_line("Нет наборов точек.")
            return
        updated = 0
        for bid, pairs in self.calPairs.items():
            est = fit_tx_n_from_pairs(pairs)
            if not est:
                continue
            tx, n = est
            self.txn[bid] = (tx, n)
            self._update_row_txn(bid, tx, n)
            updated += 1
        self.log_line(f"Калибровка по точкам: обновлено {updated} маяков.")
        if updated == 0:
            self.log_line("Добавьте больше точек на разных расстояниях (1м, 2м, 5м…).")

    def clear_pairs(self):
        self.calPairs.clear()
        self.log_line("Наборы калибровочных точек очищены.")

    # ---------- Калибровка: A→B ----------
    def start_path_record(self):
        self.pathSamples.clear()
        self.pathRecording = True
        self.pathT0 = None
        self.btnPathStart.setEnabled(False)
        self.btnPathStop.setEnabled(True)
        self.log_line("Запись A→B начата — ведите метку ровно и равномерно от A к B.")

    def stop_path_record(self):
        self.pathRecording = False
        self.btnPathStart.setEnabled(True)
        self.btnPathStop.setEnabled(False)
        self.log_line(f"Запись A→B остановлена. Кадров: {len(self.pathSamples)}")

    def solve_txn_from_path(self):
        if not self.pathSamples:
            self.log_line("Нет данных A→B.")
            return
        A = np.array([self.edAx.value(), self.edAy.value(), self.edAz.value()], float)
        B = np.array([self.edBx.value(), self.edBy.value(), self.edBz.value()], float)
        if np.allclose(A, B):
            self.log_line("Точки A и B совпадают.")
            return
        T = self.pathSamples[-1][0]
        if T <= 0:
            self.log_line("Слишком короткая запись.")
            return

        # для каждого кадра и маяка формируем пары (rssi, d_true)
        by_id = {}
        for t, dmap in self.pathSamples:
            p = A + (t / T) * (B - A)
            for bid, rssi in dmap.items():
                d_true = float(np.linalg.norm(p - np.array(self.anchors[bid][:3])))
                by_id.setdefault(bid, []).append((rssi, d_true))

        updated = 0
        for bid, pairs in by_id.items():
            if len(pairs) < 12:   # хотя бы 12 точек на маяк
                continue
            est = fit_tx_n_from_pairs(pairs)
            if not est:
                continue
            tx, n = est
            self.txn[bid] = (tx, n)
            self._update_row_txn(bid, tx, n)
            updated += 1

        self.log_line(f"Калибровка по A→B: обновлено {updated} маяков.")
        if updated == 0:
            self.log_line("Проведите дольше/ровнее, или используйте калибровку по точкам.")

    # ---------- Помощники ----------
    def _update_row_txn(self, bid, tx, n):
        # Отобразить новые tx/n в таблице
        for r in range(self.table.rowCount()):
            try:
                rid = int(self.table.item(r, 1).text())
            except:
                continue
            if rid == bid:
                self.table.item(r, 6).setText(f"{tx:.2f}")
                self.table.item(r, 7).setText(f"{n:.3f}")
                break

    # ---------- Сохранение/загрузка ----------
    def save_config(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить конфиг", "rtls_config.json", "JSON (*.json)")
        if not path:
            return
        data = {
            "beacons": [],
            "filters": {
                "median": self.spMedian.value(),
                "alpha": self.spAlpha.value(),
                "dmin": self.edDmin.value(),
                "dmax": self.edDmax.value(),
                "max_jump": self.edJump.value(),
                "skipN": self.spSkip.value(),
                "ttl": self.spTTL.value(),
            },
            "pos_filter": {"alpha": self.spPosAlpha.value(), "beta": self.spPosBeta.value()},
        }
        for on, bid, port, pos, tx, n in self.gather_beacons():
            data["beacons"].append({
                "on": on, "id": bid, "port": port,
                "x": pos[0], "y": pos[1], "z": pos[2],
                "tx1m": tx, "n": n
            })
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.log_line(f"Конфиг сохранён: {path}")
        except Exception as e:
            self.log_line(f"Ошибка сохранения: {e}")

    def load_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Загрузить конфиг", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.log_line(f"Ошибка загрузки: {e}")
            return

        self.table.setRowCount(0)
        for b in data.get("beacons", []):
            self.add_row(
                bool(b.get("on", True)),
                int(b.get("id", 0)),
                str(b.get("port", "COM6")),
                float(b.get("x", 0.0)), float(b.get("y", 0.0)), float(b.get("z", 0.0)),
                float(b.get("tx1m", -59.0)), float(b.get("n", 2.0))
            )
        f = data.get("filters", {})
        self.spMedian.setValue(int(f.get("median", MEDIAN_WIN)))
        self.spAlpha.setValue(float(f.get("alpha", EMA_ALPHA)))
        self.edDmin.setValue(float(f.get("dmin", D_RANGE[0])))
        self.edDmax.setValue(float(f.get("dmax", D_RANGE[1])))
        self.edJump.setValue(float(f.get("max_jump", MAX_JUMP)))
        self.spSkip.setValue(int(f.get("skipN", SKIP_FIRST_N)))
        self.spTTL.setValue(float(f.get("ttl", TTL_SEC)))

        pf = data.get("pos_filter", {})
        self.spPosAlpha.setValue(float(pf.get("alpha", POS_ALPHA)))
        self.spPosBeta.setValue(float(pf.get("beta", POS_BETA)))

        self.log_line(f"Конфиг загружен: {path}")

    # ----------
    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        self.stop_readers()
        super().closeEvent(e)


# --------------------- MAIN ---------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
