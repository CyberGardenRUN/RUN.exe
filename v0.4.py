import sys, re, time, math
from collections import deque, defaultdict
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, QThread

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (activates 3D)

import serial

# ------------------------- ПАРАМЕТРЫ ПО УМОЛЧАНИЮ -------------------------

BAUDRATE = 115200
SER_TIMEOUT = 0.01       # 10 ms для минимальной задержки
SKIP_FIRST_N = 20        # пропустить первые N строк с порта (прогрев)
MEDIAN_WIN = 5           # окно медианы по расстоянию
EMA_ALPHA = 0.35         # EMA по результату медианы
TARGET_NAME = "TAG_123"  # просто для лога; парсер вытаскивает Distance

# Регекс вытаскивает "Distance: 1.234 meters" и, если есть, RSSI.
RE_DISTANCE = re.compile(r"Distance:\s*([0-9]+(?:\.[0-9]+)?)\s*meters", re.IGNORECASE)
RE_RSSI     = re.compile(r"RSSI:\s*(-?\d+)", re.IGNORECASE)

# ------------------------- ПОМОЩНИКИ -------------------------

def trilaterate_lstsq(anchors, distances):
    """
    anchors: list of (x, y, z)
    distances: list of d
    Возвращает (x, y, z) решения в смысле НК (3D).
    Требует >= 3 точек. С 4 — устойчивей.
    """
    n = len(anchors)
    if n < 3:
        return None

    # Базовую берем первую
    x0, y0, z0 = anchors[0]
    d0 = distances[0]

    A = []
    b = []
    for i in range(1, n):
        xi, yi, zi = anchors[i]
        di = distances[i]
        A.append([2*(xi - x0), 2*(yi - y0), 2*(zi - z0)])
        rhs = (d0**2 - di**2) - (x0**2 + y0**2 + z0**2) + (xi**2 + yi**2 + zi**2)
        b.append(rhs)

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # МНК
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        return sol  # np.array([x, y, z])
    except np.linalg.LinAlgError:
        return None


class DistFilter:
    """ Медиана+EMA по расстоянию (простая и быстрая) """
    def __init__(self, median_win=5, alpha=0.35):
        self.buf = deque(maxlen=max(1, median_win))
        self.alpha = alpha
        self.ema = None

    def update(self, value):
        self.buf.append(float(value))
        med = np.median(self.buf)
        if self.ema is None:
            self.ema = med
        else:
            self.ema = self.alpha * med + (1 - self.alpha) * self.ema
        return self.ema


# ------------------------- ПОТОК ЧТЕНИЯ СЕРИЙНИКА -------------------------

class SerialReader(QThread):
    """
    Отдельный поток на каждый COM.
    Сигналит (beacon_id, distance, rssi, raw_line) при новой валидной строке.
    """
    newDistance = pyqtSignal(int, float, object, str)
    portStatus  = pyqtSignal(str)  # для статуса/логов

    def __init__(self, beacon_id, port, parent=None):
        super().__init__(parent)
        self.beacon_id = beacon_id
        self.port_name = port
        self.ser = None
        self._running = True
        self.skip_count = 0

    def run(self):
        try:
            self.ser = serial.Serial(self.port_name, BAUDRATE, timeout=SER_TIMEOUT)
            self.portStatus.emit(f"Открыт {self.port_name} для маяка {self.beacon_id}")
        except Exception as e:
            self.portStatus.emit(f"Ошибка открытия {self.port_name}: {e}")
            return

        while self._running:
            try:
                line = self.ser.readline().decode('latin-1', errors='ignore').strip()
                if not line:
                    continue

                # Пропускаем первые N сообщений после старта
                if self.skip_count < SKIP_FIRST_N:
                    self.skip_count += 1
                    continue

                m = RE_DISTANCE.search(line)
                if not m:
                    # Часто попадаются строки "ets Jul..." — игнор
                    continue
                distance = float(m.group(1))

                rssi_val = None
                m2 = RE_RSSI.search(line)
                if m2:
                    try:
                        rssi_val = int(m2.group(1))
                    except:
                        rssi_val = None

                self.newDistance.emit(self.beacon_id, distance, rssi_val, line)

            except Exception as e:
                self.portStatus.emit(f"[{self.port_name}] ошибка чтения: {e}")
                time.sleep(0.05)

        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
                self.portStatus.emit(f"Закрыт {self.port_name}")
        except:
            pass

    def stop(self):
        self._running = False


# ------------------------- КАНВАС MPL -------------------------

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(constrained_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')

    def draw_scene(self, anchors, tracker_pos):
        self.ax.cla()
        # Маяки
        if anchors:
            ax_pts = np.array(anchors)
            self.ax.scatter(ax_pts[:,0], ax_pts[:,1], ax_pts[:,2], s=60, marker='^', label='Beacons')
        # Трекер
        if tracker_pos is not None:
            self.ax.scatter([tracker_pos[0]], [tracker_pos[1]], [tracker_pos[2]], s=70, label='Tracker')

        # автоматические границы
        all_pts = anchors.copy()
        if tracker_pos is not None:
            all_pts.append(tracker_pos)
        if all_pts:
            all_pts = np.array(all_pts)
            mins = all_pts.min(axis=0) - 0.5
            maxs = all_pts.max(axis=0) + 0.5
            self.ax.set_xlim(mins[0], maxs[0])
            self.ax.set_ylim(mins[1], maxs[1])
            self.ax.set_zlim(mins[2], maxs[2])

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend(loc='upper right')
        self.draw()


# ------------------------- ГЛАВНОЕ ОКНО -------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ESP32 RTLS (BLE RSSI → Distance) — Real-Time Trilateration")
        self.resize(1200, 700)

        # Таблица маяков
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["ID", "COM порт", "X", "Y", "Z"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked | QtWidgets.QAbstractItemView.SelectedClicked)

        # Кнопки
        self.btnAdd = QtWidgets.QPushButton("Добавить маяк")
        self.btnDel = QtWidgets.QPushButton("Удалить выбранный")
        self.btnStart = QtWidgets.QPushButton("Старт")
        self.btnStop = QtWidgets.QPushButton("Стоп")
        self.btnStop.setEnabled(False)

        # Статус/лог
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)

        # Параметры фильтра
        self.medianSpin = QtWidgets.QSpinBox()
        self.medianSpin.setRange(1, 99)
        self.medianSpin.setValue(MEDIAN_WIN)
        self.alphaSpin = QtWidgets.QDoubleSpinBox()
        self.alphaSpin.setDecimals(2)
        self.alphaSpin.setRange(0.0, 1.0)
        self.alphaSpin.setSingleStep(0.05)
        self.alphaSpin.setValue(EMA_ALPHA)

        filtBox = QtWidgets.QGroupBox("Фильтр расстояний")
        filtLay = QtWidgets.QFormLayout(filtBox)
        filtLay.addRow("Окно медианы:", self.medianSpin)
        filtLay.addRow("EMA α:", self.alphaSpin)

        # Платно/графика
        self.canvas = MplCanvas(self)

        # Компоновка слева (управление)
        leftLay = QtWidgets.QVBoxLayout()
        leftLay.addWidget(self.table)
        btnRow = QtWidgets.QHBoxLayout()
        btnRow.addWidget(self.btnAdd)
        btnRow.addWidget(self.btnDel)
        leftLay.addLayout(btnRow)
        leftLay.addWidget(filtBox)
        leftLay.addWidget(self.btnStart)
        leftLay.addWidget(self.btnStop)
        leftLay.addWidget(self.log, 1)

        left = QtWidgets.QWidget()
        left.setLayout(leftLay)

        # Сплиттер
        splitter = QtWidgets.QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

        # Данные/потоки
        self.threads = []
        self.filters = {}        # beacon_id -> DistFilter
        self.latest_dist = {}    # beacon_id -> float
        self.beacons_def = {}    # beacon_id -> (x,y,z)

        # Сигналы
        self.btnAdd.clicked.connect(self.add_row)
        self.btnDel.clicked.connect(self.delete_selected)
        self.btnStart.clicked.connect(self.start_readers)
        self.btnStop.clicked.connect(self.stop_readers)

        # Предзаполним 4 маяка как пример
        self.prefill_example()

        # Таймер для перерисовки/триангуляции (минимальная задержка)
        self.tmr = QtCore.QTimer(self)
        self.tmr.setInterval(40)  # 25 FPS ~ 40ms
        self.tmr.timeout.connect(self.update_solution)

    def log_line(self, s):
        self.log.appendPlainText(s)

    def prefill_example(self):
        data = [
            (1, "COM8", 0.0, 0.0, 0.0),
            (2, "COM14", 5.0, 0.0, 0.0),
            (3, "COM15", 0.0, 5.0, 0.0),
            (4, "COM16", 0.0, 0.0, 3.0),
        ]
        for row in data:
            self.add_row(*row)

    def add_row(self, id_val=None, port=None, x=0.0, y=0.0, z=0.0):
        r = self.table.rowCount()
        self.table.insertRow(r)

        id_item = QtWidgets.QTableWidgetItem(str(id_val if id_val is not None else r + 1))
        port_item = QtWidgets.QTableWidgetItem(port or "COM6")
        x_item = QtWidgets.QTableWidgetItem(str(x))
        y_item = QtWidgets.QTableWidgetItem(str(y))
        z_item = QtWidgets.QTableWidgetItem(str(z))

        for it in (id_item, port_item, x_item, y_item, z_item):
            it.setTextAlignment(Qt.AlignCenter)

        self.table.setItem(r, 0, id_item)
        self.table.setItem(r, 1, port_item)
        self.table.setItem(r, 2, x_item)
        self.table.setItem(r, 3, y_item)
        self.table.setItem(r, 4, z_item)

    def delete_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def gather_beacons(self):
        """ Прочитать конфиг из таблицы. """
        beacons = []
        ids_seen = set()
        for r in range(self.table.rowCount()):
            try:
                bid = int(self.table.item(r, 0).text())
                port = self.table.item(r, 1).text().strip()
                x = float(self.table.item(r, 2).text())
                y = float(self.table.item(r, 3).text())
                z = float(self.table.item(r, 4).text())
                if bid in ids_seen:
                    self.log_line(f"Дублируется ID {bid} — пропущено")
                    continue
                ids_seen.add(bid)
                beacons.append((bid, port, (x, y, z)))
            except Exception as e:
                self.log_line(f"Ошибка строки {r+1}: {e}")
        return beacons

    def start_readers(self):
        # Очистка старых
        self.stop_readers()

        self.latest_dist.clear()
        self.filters.clear()
        self.beacons_def.clear()

        beacons = self.gather_beacons()
        if len(beacons) < 3:
            self.log_line("Нужно минимум 3 маяка.")
            return

        # Подготовка фильтров и словарей
        for bid, port, pos in beacons:
            self.filters[bid] = DistFilter(self.medianSpin.value(), self.alphaSpin.value())
            self.beacons_def[bid] = tuple(pos)

        # Старт потоков
        for bid, port, _pos in beacons:
            th = SerialReader(bid, port)
            th.newDistance.connect(self.on_new_distance)
            th.portStatus.connect(self.log_line)
            th.start()
            self.threads.append(th)

        self.btnStart.setEnabled(False)
        self.btnStop.setEnabled(True)
        self.log_line("Старт опроса портов…")

        # Запускаем периодический пересчёт/отрисовку
        self.tmr.start()

    def stop_readers(self):
        # Останавливаем таймер
        self.tmr.stop()
        # Останавливаем потоки
        for th in self.threads:
            th.stop()
            th.wait(500)
        self.threads.clear()
        self.btnStart.setEnabled(True)
        self.btnStop.setEnabled(False)

    @QtCore.pyqtSlot(int, float, object, str)
    def on_new_distance(self, beacon_id, distance, rssi, raw):
        # фильтруем и запоминаем
        filt = self.filters.get(beacon_id)
        if filt is None:
            return
        dist_f = float(distance)
        smooth = filt.update(dist_f)
        self.latest_dist[beacon_id] = smooth

    def update_solution(self):
        # Готовим входы для трилатерации
        if len(self.latest_dist) < 3:
            # Обновляем сцену только с маяками
            anchors = [self.beacons_def[bid] for bid in sorted(self.beacons_def.keys())]
            self.canvas.draw_scene(anchors, None)
            return

        bids = sorted(self.latest_dist.keys())
        anchors = []
        dists = []
        for bid in bids:
            anchors.append(self.beacons_def[bid])
            dists.append(self.latest_dist[bid])

        pos = trilaterate_lstsq(anchors, dists)
        if pos is not None and np.all(np.isfinite(pos)):
            tracker = tuple(map(float, pos))
        else:
            tracker = None

        self.canvas.draw_scene(anchors, tracker)

    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        self.stop_readers()
        return super().closeEvent(e)


# ------------------------- MAIN -------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
