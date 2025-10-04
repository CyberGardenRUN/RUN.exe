import sys, re, time
from collections import deque
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, QThread

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import serial

# ------------------------- ПАРАМЕТРЫ -------------------------

BAUDRATE = 115200
SER_TIMEOUT = 0.01     # 10ms — минимум задержки
SKIP_FIRST_N = 20
MEDIAN_WIN = 5
EMA_ALPHA = 0.35

RE_DISTANCE = re.compile(
    r"Distance(?:\s*\(.*?\))?:\s*([0-9]+(?:[.,][0-9]+)?)\s*meters?",
    re.IGNORECASE
)
# ------------------------- МАТЕМАТИКА -------------------------

def trilaterate_lstsq(anchors, distances):
    n = len(anchors)
    if n < 3:
        return None
    x0, y0, z0 = anchors[0]
    d0 = distances[0]
    A, b = [], []
    for i in range(1, n):
        xi, yi, zi = anchors[i]
        di = distances[i]
        A.append([2*(xi - x0), 2*(yi - y0), 2*(zi - z0)])
        b.append((d0**2 - di**2) - (x0**2 + y0**2 + z0**2) + (xi**2 + yi**2 + zi**2))
    A = np.array(A, float)
    b = np.array(b, float)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        return sol
    except np.linalg.LinAlgError:
        return None

class DistFilter:
    def __init__(self, median_win=5, alpha=0.35):
        from collections import deque
        self.buf = deque(maxlen=max(1, median_win))
        self.alpha = alpha
        self.ema = None
    def update(self, v):
        v = float(v)
        self.buf.append(v)
        med = np.median(self.buf)
        self.ema = med if self.ema is None else self.alpha*med + (1-self.alpha)*self.ema
        return self.ema

# ------------------------- ПОТОК СЕРПОРТА -------------------------

class SerialReader(QThread):
    newDistance = pyqtSignal(int, float, object, str)  # (beacon_id, dist, rssi, raw)
    portStatus  = pyqtSignal(str)
    def __init__(self, beacon_id, port):
        super().__init__()
        self.beacon_id = beacon_id
        self.port_name = port
        self.ser = None
        self._run = True
        self.skip = 0
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
                if self.skip < SKIP_FIRST_N:
                    self.skip += 1
                    continue
                m = RE_DISTANCE.search(line)
                if not m:
                    continue
                dist = float(m.group(1))
                rssi = None
                m2 = RE_RSSI.search(line)
                if m2:
                    try: rssi = int(m2.group(1))
                    except: rssi = None
                self.newDistance.emit(self.beacon_id, dist, rssi, line)
            except Exception as e:
                self.portStatus.emit(f"[{self.port_name}] ошибка чтения: {e}")
                time.sleep(0.03)
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
                self.portStatus.emit(f"Закрыт {self.port_name}")
        except: pass
    def stop(self):
        self._run = False

# ------------------------- MPL КАНВАС -------------------------

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(constrained_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
    def draw_scene(self, anchors, tracker_pos):
        self.ax.cla()
        if anchors:
            arr = np.array(anchors)
            self.ax.scatter(arr[:,0], arr[:,1], arr[:,2], s=70, marker='^', label='Beacons')
        if tracker_pos is not None:
            self.ax.scatter([tracker_pos[0]], [tracker_pos[1]], [tracker_pos[2]], s=80, label='Tracker')
        all_pts = anchors.copy()
        if tracker_pos is not None: all_pts.append(tracker_pos)
        if all_pts:
            pts = np.array(all_pts)
            mn = pts.min(axis=0)-0.5; mx = pts.max(axis=0)+0.5
            self.ax.set_xlim(mn[0], mx[0]); self.ax.set_ylim(mn[1], mx[1]); self.ax.set_zlim(mn[2], mx[2])
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
        self.ax.legend(loc='upper right'); self.draw()

# ------------------------- ГЛАВНОЕ ОКНО -------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ESP32 RTLS — Real-Time Trilateration + Calibration")
        self.resize(1300, 740)

        # ---- Таблица маяков
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["ID","COM","X","Y","Z"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        # ---- Кнопки
        self.btnAdd = QtWidgets.QPushButton("Добавить маяк")
        self.btnDel = QtWidgets.QPushButton("Удалить выбранный")
        self.btnStart = QtWidgets.QPushButton("Старт")
        self.btnStop = QtWidgets.QPushButton("Стоп"); self.btnStop.setEnabled(False)

        # ---- Фильтр
        self.medianSpin = QtWidgets.QSpinBox(); self.medianSpin.setRange(1,99); self.medianSpin.setValue(MEDIAN_WIN)
        self.alphaSpin  = QtWidgets.QDoubleSpinBox(); self.alphaSpin.setRange(0.0,1.0); self.alphaSpin.setDecimals(2); self.alphaSpin.setSingleStep(0.05); self.alphaSpin.setValue(EMA_ALPHA)
        filtBox = QtWidgets.QGroupBox("Фильтр расстояний")
        fl = QtWidgets.QFormLayout(filtBox)
        fl.addRow("Окно медианы:", self.medianSpin)
        fl.addRow("EMA α:", self.alphaSpin)

        # ---- Калибровка
        self.applyCalibChk = QtWidgets.QCheckBox("Применять калибровку"); self.applyCalibChk.setChecked(True)

        self.calPointX = QtWidgets.QDoubleSpinBox(); self.calPointX.setDecimals(3); self.calPointX.setRange(-1e4,1e4)
        self.calPointY = QtWidgets.QDoubleSpinBox(); self.calPointY.setDecimals(3); self.calPointY.setRange(-1e4,1e4)
        self.calPointZ = QtWidgets.QDoubleSpinBox(); self.calPointZ.setDecimals(3); self.calPointZ.setRange(-1e4,1e4)
        self.btnSnapPoint = QtWidgets.QPushButton("Снять срез в точке")
        self.btnSolveCal  = QtWidgets.QPushButton("Посчитать калибровку (точки)")
        self.btnClearCal  = QtWidgets.QPushButton("Сбросить калибровку")

        # Траектория A→B
        self.axA = QtWidgets.QDoubleSpinBox(); self.axA.setDecimals(3); self.axA.setRange(-1e4,1e4)
        self.ayA = QtWidgets.QDoubleSpinBox(); self.ayA.setDecimals(3); self.ayA.setRange(-1e4,1e4)
        self.azA = QtWidgets.QDoubleSpinBox(); self.azA.setDecimals(3); self.azA.setRange(-1e4,1e4)
        self.axB = QtWidgets.QDoubleSpinBox(); self.axB.setDecimals(3); self.axB.setRange(-1e4,1e4)
        self.ayB = QtWidgets.QDoubleSpinBox(); self.ayB.setDecimals(3); self.ayB.setRange(-1e4,1e4)
        self.azB = QtWidgets.QDoubleSpinBox(); self.azB.setDecimals(3); self.azB.setRange(-1e4,1e4)
        self.btnPathStart = QtWidgets.QPushButton("Старт (A→B) запись")
        self.btnPathStop  = QtWidgets.QPushButton("Стоп (A→B) запись"); self.btnPathStop.setEnabled(False)
        self.btnSolvePath = QtWidgets.QPushButton("Посчитать калибровку (A→B)")

        calBox = QtWidgets.QGroupBox("Калибровка")
        cl = QtWidgets.QFormLayout(calBox)
        cl.addRow(self.applyCalibChk)
        ptLay = QtWidgets.QHBoxLayout()
        for w in (QtWidgets.QLabel("X:"), self.calPointX, QtWidgets.QLabel("Y:"), self.calPointY, QtWidgets.QLabel("Z:"), self.calPointZ, self.btnSnapPoint, self.btnSolveCal, self.btnClearCal):
            ptLay.addWidget(w if isinstance(w, QtWidgets.QWidget) else QtWidgets.QLabel(str(w)))
        cl.addRow(QtWidgets.QLabel("Точки:"), QtWidgets.QWidget())
        cl.itemAt(cl.rowCount()-1, QtWidgets.QFormLayout.FieldRole).widget().setLayout(ptLay)

        pathGrid = QtWidgets.QGridLayout()
        pathGrid.addWidget(QtWidgets.QLabel("A.x"),0,0); pathGrid.addWidget(self.axA,0,1)
        pathGrid.addWidget(QtWidgets.QLabel("A.y"),0,2); pathGrid.addWidget(self.ayA,0,3)
        pathGrid.addWidget(QtWidgets.QLabel("A.z"),0,4); pathGrid.addWidget(self.azA,0,5)
        pathGrid.addWidget(QtWidgets.QLabel("B.x"),1,0); pathGrid.addWidget(self.axB,1,1)
        pathGrid.addWidget(QtWidgets.QLabel("B.y"),1,2); pathGrid.addWidget(self.ayB,1,3)
        pathGrid.addWidget(QtWidgets.QLabel("B.z"),1,4); pathGrid.addWidget(self.azB,1,5)
        pathGrid.addWidget(self.btnPathStart,2,0,1,3); pathGrid.addWidget(self.btnPathStop,2,3,1,3)
        pathGrid.addWidget(self.btnSolvePath,3,0,1,6)
        cl.addRow(QtWidgets.QLabel("Траектория A→B:"), QtWidgets.QWidget())
        cl.itemAt(cl.rowCount()-1, QtWidgets.QFormLayout.FieldRole).widget().setLayout(pathGrid)

        # ---- Лог и графика
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(2000)
        self.canvas = MplCanvas(self)

        # ---- Компоновка
        leftLay = QtWidgets.QVBoxLayout()
        leftLay.addWidget(self.table)
        rowBtns = QtWidgets.QHBoxLayout(); rowBtns.addWidget(self.btnAdd); rowBtns.addWidget(self.btnDel)
        leftLay.addLayout(rowBtns)
        leftLay.addWidget(filtBox)
        leftLay.addWidget(calBox)
        leftLay.addWidget(self.btnStart); leftLay.addWidget(self.btnStop)
        leftLay.addWidget(self.log,1)
        left = QtWidgets.QWidget(); left.setLayout(leftLay)

        splitter = QtWidgets.QSplitter(); splitter.addWidget(left); splitter.addWidget(self.canvas); splitter.setStretchFactor(1,1)
        self.setCentralWidget(splitter)

        # ---- Данные/состояние
        self.threads = []
        self.filters = {}         # id -> DistFilter
        self.latest_dist = {}     # id -> float (filtered)
        self.anchors = {}         # id -> (x,y,z)
        self.calibAB = {}         # id -> (a,b)  per-beacon
        # калибровочные пары по маякам: id -> list of (d_meas, d_true)
        self.calPairs = {}
        # запись A→B
        self.pathRecording = False
        self.pathSamples = []     # list of (timestamp, {id: d_meas})
        self.pathT0 = None

        # ---- Сигналы
        self.btnAdd.clicked.connect(self.add_row)
        self.btnDel.clicked.connect(self.delete_selected)
        self.btnStart.clicked.connect(self.start_readers)
        self.btnStop.clicked.connect(self.stop_readers)

        self.btnSnapPoint.clicked.connect(self.capture_calib_point)
        self.btnSolveCal.clicked.connect(self.solve_calibration_from_points)
        self.btnClearCal.clicked.connect(self.clear_calibration)

        self.btnPathStart.clicked.connect(self.start_path_record)
        self.btnPathStop.clicked.connect(self.stop_path_record)
        self.btnSolvePath.clicked.connect(self.solve_calibration_from_path)

        self.prefill_example()

        # ---- Таймер рендера
        self.tmr = QtCore.QTimer(self); self.tmr.setInterval(40); self.tmr.timeout.connect(self.update_solution)

    # ---------- Утилиты UI
    def log_line(self, s): self.log.appendPlainText(s)
    def prefill_example(self):
        for row in [(1,"COM8",0,0,0),(2,"COM14",5,0,0),(3,"COM15",0,5,0),(4,"COM16",0,0,3)]:
            self.add_row(*row)
    def add_row(self, id_val=None, port=None, x=0.0, y=0.0, z=0.0):
        r = self.table.rowCount(); self.table.insertRow(r)
        def cell(v): it=QtWidgets.QTableWidgetItem(str(v)); it.setTextAlignment(Qt.AlignCenter); return it
        self.table.setItem(r,0,cell(id_val if id_val is not None else r+1))
        self.table.setItem(r,1,cell(port or "COM6"))
        self.table.setItem(r,2,cell(x)); self.table.setItem(r,3,cell(y)); self.table.setItem(r,4,cell(z))
    def delete_selected(self):
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)
    def gather_beacons(self):
        res=[]; ids=set()
        for r in range(self.table.rowCount()):
            try:
                bid=int(self.table.item(r,0).text()); port=self.table.item(r,1).text().strip()
                x=float(self.table.item(r,2).text()); y=float(self.table.item(r,3).text()); z=float(self.table.item(r,4).text())
            except Exception as e:
                self.log_line(f"Ошибка строки {r+1}: {e}"); continue
            if bid in ids: self.log_line(f"Дублируется ID {bid} — пропуск"); continue
            ids.add(bid); res.append((bid,port,(x,y,z)))
        return res

    # ---------- Старт/стоп
    def start_readers(self):
        self.stop_readers()
        self.filters.clear(); self.latest_dist.clear(); self.anchors.clear()
        for bid,port,pos in self.gather_beacons():
            self.filters[bid]=DistFilter(self.medianSpin.value(), self.alphaSpin.value())
            self.anchors[bid]=tuple(pos)
        if len(self.anchors)<3: self.log_line("Нужно ≥3 маяка"); return
        for bid,port,_ in self.gather_beacons():
            th=SerialReader(bid,port); th.newDistance.connect(self.on_new_distance); th.portStatus.connect(self.log_line); th.start(); self.threads.append(th)
        self.btnStart.setEnabled(False); self.btnStop.setEnabled(True)
        self.tmr.start(); self.log_line("Старт опроса…")
    def stop_readers(self):
        self.tmr.stop()
        for th in self.threads: th.stop(); th.wait(500)
        self.threads.clear()
        self.btnStart.setEnabled(True); self.btnStop.setEnabled(False)

    # ---------- Приход данных
    @QtCore.pyqtSlot(int, float, object, str)
    def on_new_distance(self, bid, dist, rssi, raw):
        # фильтрация
        filt=self.filters.get(bid); 
        if not filt: return
        sm=filt.update(dist)
        # калибровка (если включена)
        if self.applyCalibChk.isChecked():
            a,b=self.calibAB.get(bid,(1.0,0.0))
            sm=max(0.0, a*sm + b)
        self.latest_dist[bid]=sm
        # запись для траектории
        if self.pathRecording:
            if not self.pathSamples: self.pathT0=time.time()
            t=time.time()-self.pathT0
            if not self.pathSamples or abs(self.pathSamples[-1][0]-t)>0.01:
                self.pathSamples.append((t,{bid:sm}))
            else:
                self.pathSamples[-1][1][bid]=sm

    # ---------- Отрисовка + позиция
    def update_solution(self):
        if len(self.latest_dist)<3:
            self.canvas.draw_scene(list(self.anchors.values()), None); return
        bids=sorted(self.latest_dist.keys())
        anchors=[self.anchors[b] for b in bids]
        dists=[self.latest_dist[b] for b in bids]
        pos=trilaterate_lstsq(anchors,dists)
        tracker = tuple(map(float,pos)) if pos is not None and np.all(np.isfinite(pos)) else None
        self.canvas.draw_scene(list(self.anchors.values()), tracker)

    # ---------- КАЛИБРОВКА: ТОЧКИ
    def capture_calib_point(self):
        if len(self.latest_dist)<3:
            self.log_line("Недостаточно данных для среза (нужно ≥1 маяк, но лучше все).")
        x=self.calPointX.value(); y=self.calPointY.value(); z=self.calPointZ.value()
        if not self.anchors:
            self.log_line("Нет маяков."); return
        for bid, anc in self.anchors.items():
            d_true = float(np.linalg.norm(np.array([x,y,z]) - np.array(anc)))
            d_meas = self.latest_dist.get(bid, None)
            if d_meas is None: continue
            self.calPairs.setdefault(bid, []).append((float(d_meas), d_true))
        self.log_line(f"Снят срез в точке ({x:.3f},{y:.3f},{z:.3f})")

    def solve_calibration_from_points(self):
        if not self.calPairs:
            self.log_line("Нет срезов для калибровки."); return
        cnt=0
        for bid,pairs in self.calPairs.items():
            if len(pairs)<2: continue
            X=np.array([p[0] for p in pairs],float).reshape(-1,1)
            X=np.hstack([X, np.ones((len(X),1))]) # [d_meas, 1]
            y=np.array([p[1] for p in pairs],float)
            try:
                coeff, *_ = np.linalg.lstsq(X,y,rcond=None)  # a,b
                a=float(coeff[0]); b=float(coeff[1])
                # немного ограничим сумасшедшие значения
                if not np.isfinite(a) or abs(a)>10: continue
                if not np.isfinite(b) or abs(b)>20: continue
                self.calibAB[bid]=(a,b); cnt+=1
            except Exception as e:
                self.log_line(f"ID {bid}: ошибка МНК — {e}")
        self.log_line(f"Калибровка по точкам: обновлено {cnt} маяков.")
        if cnt==0: self.log_line("Добавь больше точек (разные расстояния).")

    def clear_calibration(self):
        self.calibAB.clear(); self.calPairs.clear()
        self.log_line("Калибровка сброшена.")

    # ---------- КАЛИБРОВКА: ТРАЕКТОРИЯ A→B
    def start_path_record(self):
        self.pathSamples.clear()
        self.pathRecording=True
        self.btnPathStart.setEnabled(False); self.btnPathStop.setEnabled(True)
        self.log_line("Запись A→B начата. Плавно веди трекер от A к B примерно равномерно…")
    def stop_path_record(self):
        self.pathRecording=False
        self.btnPathStart.setEnabled(True); self.btnPathStop.setEnabled(False)
        self.log_line(f"Запись A→B остановлена. Кадров: {len(self.pathSamples)}")
    def solve_calibration_from_path(self):
        if not self.pathSamples:
            self.log_line("Нет записанных данных A→B."); return
        A=np.array([self.axA.value(), self.ayA.value(), self.azA.value()],float)
        B=np.array([self.axB.value(), self.ayB.value(), self.azB.value()],float)
        if np.allclose(A,B):
            self.log_line("A и B совпадают."); return
        T = self.pathSamples[-1][0] if self.pathSamples else 0.0
        if T<=0: self.log_line("Длина записи слишком мала."); return

        # Собираем пары для каждого маяка
        pairs_by_id = {}
        for t, dmap in self.pathSamples:
            p = A + (t/T)*(B-A)  # истинная позиция при предположении равномерности
            for bid, dist_meas in dmap.items():
                d_true = float(np.linalg.norm(p - np.array(self.anchors[bid])))
                pairs_by_id.setdefault(bid, []).append((dist_meas, d_true))

        cnt=0
        for bid, pairs in pairs_by_id.items():
            if len(pairs)<5:  # желательно побольше
                continue
            X=np.array([p[0] for p in pairs],float).reshape(-1,1)
            X=np.hstack([X, np.ones((len(X),1))])
            y=np.array([p[1] for p in pairs],float)
            try:
                coeff,*_ = np.linalg.lstsq(X,y,rcond=None)
                a=float(coeff[0]); b=float(coeff[1])
                if np.isfinite(a) and abs(a)<10 and np.isfinite(b) and abs(b)<20:
                    self.calibAB[bid]=(a,b); cnt+=1
            except Exception as e:
                self.log_line(f"ID {bid}: ошибка МНК — {e}")

        self.log_line(f"Калибровка по траектории: обновлено {cnt} маяков.")
        if cnt==0: self.log_line("Попробуй дольше и ровнее провести от A к B, или используй калибровку точками.")

    # ----------
    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        self.stop_readers()
        return super().closeEvent(e)

# ------------------------- MAIN -------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
