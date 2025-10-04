# rtls_calib_tracker.py
import sys, re, time, json, os
from collections import deque, defaultdict
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, QThread

import serial
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# ------------------- ПАРАМЕТРЫ -------------------
BAUDRATE        = 115200
SER_TIMEOUT     = 0.01
SKIP_FIRST_N    = 20

AGG_WINDOW_SEC  = 0.20   # окно агрегации пакетов (≈实时 + устойчивость)
RESID_THRESH    = 0.80   # м — отсев выбросов по резидуалу
TRUST_STEP_MAX  = 0.40   # м — max шаг итерации
V_MAX           = 2.0    # м/с — мягкое ограничение скорости

CALIB_HOLD_SEC  = 3.0    # сек — длительность сбора на каждой точке
CALIB_R_LIST    = [1,2,3,4,5]  # радиусы метки от маяка
CALIB_PTS_PER_R = 3      # «угловых» точки на каждом радиусе

RE_DISTANCE = re.compile(r"Distance(?:\s*\(.*?\))?:\s*([0-9]+(?:[.,][0-9]+)?)\s*meters?", re.I)
RE_RSSI     = re.compile(r"RSSI:\s*(-?\d+)", re.I)

SCENE_MIN = np.array([-15,-15,-2],float)
SCENE_MAX = np.array([ 15, 15,10],float)

# --------------- МАТЕМАТИКА / ФИЛЬТРЫ ---------------
def trilaterate_wls_robust(anchors, dists, x0=None, iters=12, huber=0.4, step_max=0.4, vel_soft=None):
    A = np.asarray(anchors, float); d = np.asarray(dists, float)
    N = len(d)
    if N < 3: return None
    x = np.array(A.mean(axis=0) if x0 is None else x0, float)

    for _ in range(iters):
        vec = A - x
        rng = np.linalg.norm(vec, axis=1) + 1e-9
        r   = rng - d
        w = np.ones(N)
        big = np.abs(r) > huber
        w[big] = huber/(np.abs(r[big])+1e-9)
        J = (-vec)/rng[:,None]

        if vel_soft is not None and vel_soft[0] is not None and vel_soft[2] > 0:
            x_prev, vmax, dt = vel_soft
            lam = 1.0 / max(1e-6, (vmax*dt))
            J = np.vstack([J, lam*np.eye(3)])
            r = np.hstack([r, lam*(x - x_prev)])
            w = np.hstack([w, np.ones(3)])

        JT_W = J.T * w
        H = JT_W @ J + 1e-6*np.eye(3)
        g = JT_W @ r
        try: dx = -np.linalg.solve(H,g)
        except np.linalg.LinAlgError: break

        step = np.linalg.norm(dx)
        if step > step_max: dx *= (step_max/step)
        x = x + dx
    return x

class AlphaBetaFilter:
    def __init__(self, alpha=0.25, beta=0.05):
        self.alpha=alpha; self.beta=beta
        self.x=None; self.v=np.zeros(3); self.t_prev=None
    def update(self, x_meas):
        t=time.time()
        if self.x is None:
            self.x=np.array(x_meas, float); self.v=np.zeros(3); self.t_prev=t; return self.x
        dt=max(1e-3, t - self.t_prev)
        x_pred = self.x + self.v*dt
        r = np.array(x_meas)-x_pred
        self.x = x_pred + self.alpha*r
        self.v = self.v + (self.beta/dt)*r
        self.t_prev = t
        return self.x

# --------------- СЕРИЙНЫЙ ЧТЕНИЕ ----------------
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
                if not line: continue
                if self.skip < SKIP_FIRST_N:
                    self.skip += 1; continue
                m = RE_DISTANCE.search(line)
                if not m: continue
                dist = float(m.group(1).replace(',','.'))
                rssi = None
                m2 = RE_RSSI.search(line)
                if m2:
                    try: rssi = int(m2.group(1))
                    except: rssi=None
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

# ---------------- МАТПЛОТ КАНВАС ----------------
class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        fig = Figure(constrained_layout=True)
        super().__init__(fig)
        self.ax = fig.add_subplot(111, projection='3d')
    def draw_scene(self, anchors, tracker_pos, dists=None, autoscale=True):
        self.ax.cla()
        if anchors:
            A=np.array(anchors)[:, :3]
            ids=[a[3] for a in anchors]
            self.ax.scatter(A[:,0],A[:,1],A[:,2],s=70,marker='^',label='Beacons')
            for i,(x,y,z) in enumerate(A):
                self.ax.text(x,y,z,f"{ids[i]}",fontsize=8)
            if dists is not None and len(dists)==len(anchors):
                for (x,y,z),R in zip(A,dists):
                    u,v=np.mgrid[0:2*np.pi:12j, 0:np.pi:6j]
                    xs=x+R*np.cos(u)*np.sin(v); ys=y+R*np.sin(u)*np.sin(v); zs=z+R*np.cos(v)
                    self.ax.plot_wireframe(xs,ys,zs,alpha=0.08,linewidth=0.5)
        if tracker_pos is not None:
            self.ax.scatter([tracker_pos[0]],[tracker_pos[1]],[tracker_pos[2]],s=80,label='Tracker')
        if autoscale and (anchors or tracker_pos is not None):
            pts=[]
            if anchors: pts+= [a[:3] for a in anchors]
            if tracker_pos is not None: pts.append(tracker_pos)
            P=np.array(pts); mn=P.min(axis=0)-0.5; mx=P.max(axis=0)+0.5
            self.ax.set_xlim(mn[0],mx[0]); self.ax.set_ylim(mn[1],mx[1]); self.ax.set_zlim(mn[2],mx[2])
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
        self.ax.legend(loc='upper right'); self.draw()

# -------------------- ГЛАВНОЕ ОКНО --------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ESP32 RTLS — Calibration + Real-Time Tracking")
        self.resize(1400, 780)

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # ======== TAB: TRACKING ========
        self.trk_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.trk_tab,"Трекинг")

        self.table = QtWidgets.QTableWidget(0,5)
        self.table.setHorizontalHeaderLabels(["ID","COM","X","Y","Z"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        self.btnAdd = QtWidgets.QPushButton("Добавить маяк")
        self.btnDel = QtWidgets.QPushButton("Удалить выбранный")
        self.btnStart = QtWidgets.QPushButton("Старт")
        self.btnStop  = QtWidgets.QPushButton("Стоп"); self.btnStop.setEnabled(False)

        self.cb2D = QtWidgets.QCheckBox("2D режим (фикс. Z)")
        self.edZfix = QtWidgets.QDoubleSpinBox(); self.edZfix.setRange(-1e4,1e4); self.edZfix.setDecimals(3); self.edZfix.setValue(0.0)
        self.cbAutoscale = QtWidgets.QCheckBox("Авто-масштаб"); self.cbAutoscale.setChecked(True)
        self.spPosAlpha = QtWidgets.QDoubleSpinBox(); self.spPosAlpha.setRange(0,1); self.spPosAlpha.setSingleStep(0.05); self.spPosAlpha.setValue(0.25)
        self.spPosBeta  = QtWidgets.QDoubleSpinBox(); self.spPosBeta.setRange(0,1); self.spPosBeta.setSingleStep(0.05); self.spPosBeta.setValue(0.05)
        self.spTTL      = QtWidgets.QDoubleSpinBox(); self.spTTL.setRange(0.05,5.0); self.spTTL.setSingleStep(0.05); self.spTTL.setValue(1.0)
        self.btnLoadCal = QtWidgets.QPushButton("Загрузить калибровку")
        self.btnSaveCal = QtWidgets.QPushButton("Сохранить калибровку")

        ctrl1 = QtWidgets.QHBoxLayout()
        for w in (self.btnAdd,self.btnDel,self.btnStart,self.btnStop, self.cbAutoscale):
            ctrl1.addWidget(w)
        ctrl2 = QtWidgets.QHBoxLayout()
        ctrl2.addWidget(self.cb2D); ctrl2.addWidget(QtWidgets.QLabel("Zfix")); ctrl2.addWidget(self.edZfix)
        ctrl2.addSpacing(20)
        ctrl2.addWidget(QtWidgets.QLabel("α:")); ctrl2.addWidget(self.spPosAlpha)
        ctrl2.addWidget(QtWidgets.QLabel("β:")); ctrl2.addWidget(self.spPosBeta)
        ctrl2.addWidget(QtWidgets.QLabel("TTL (с):")); ctrl2.addWidget(self.spTTL)
        ctrl2.addStretch(1)
        ctrl2.addWidget(self.btnLoadCal); ctrl2.addWidget(self.btnSaveCal)

        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(2000)
        self.canvas = MplCanvas(self)
        leftLay = QtWidgets.QVBoxLayout()
        leftLay.addWidget(self.table)
        leftLay.addLayout(ctrl1); leftLay.addLayout(ctrl2)
        leftLay.addWidget(self.log,1)
        left = QtWidgets.QWidget(); left.setLayout(leftLay)

        split = QtWidgets.QSplitter(); split.addWidget(left); split.addWidget(self.canvas); split.setStretchFactor(1,1)
        lay = QtWidgets.QVBoxLayout(self.trk_tab); lay.addWidget(split)

        # ======== TAB: CALIBRATION ========
        self.cal_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.cal_tab,"Калибровка")

        self.cal_table = QtWidgets.QTableWidget(0,2)
        self.cal_table.setHorizontalHeaderLabels(["ID","COM"])
        self.cal_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.btnCalAdd = QtWidgets.QPushButton("Добавить")
        self.btnCalDel = QtWidgets.QPushButton("Удалить выбранный")
        self.btnCalStart = QtWidgets.QPushButton("Старт автокалибровки")
        self.btnCalStop  = QtWidgets.QPushButton("Стоп"); self.btnCalStop.setEnabled(False)
        self.calLog = QtWidgets.QPlainTextEdit(); self.calLog.setReadOnly(True)

        calLay = QtWidgets.QVBoxLayout(self.cal_tab)
        rowCal = QtWidgets.QHBoxLayout(); rowCal.addWidget(self.btnCalAdd); rowCal.addWidget(self.btnCalDel)
        calLay.addWidget(self.cal_table); calLay.addLayout(rowCal)
        calLay.addWidget(self.btnCalStart); calLay.addWidget(self.btnCalStop)
        calLay.addWidget(self.calLog,1)

        # ---- состояние
        self.threads=[]
        self.filters={}
        self.latest={}    # id -> (d_filt, ts, rssi)
        self.hist={}      # id -> deque(ts, d_filt)
        self.anchors={}   # id -> (x,y,z)
        self.calibAB={}   # id -> ('linear',a,b) or ('quad',c2,a,b)
        self.posFilter = AlphaBetaFilter()
        self._last_pos = None

        self.status = QtWidgets.QStatusBar(); self.setStatusBar(self.status)

        # ---- сигналы
        self.btnAdd.clicked.connect(self.add_row)
        self.btnDel.clicked.connect(self.delete_selected)
        self.btnStart.clicked.connect(self.start_readers)
        self.btnStop.clicked.connect(self.stop_readers)
        self.btnLoadCal.clicked.connect(self.load_calibration)
        self.btnSaveCal.clicked.connect(self.save_calibration)

        self.btnCalAdd.clicked.connect(self.cal_add_row)
        self.btnCalDel.clicked.connect(self.cal_delete_selected)
        self.btnCalStart.clicked.connect(self.start_calibration)
        self.btnCalStop.clicked.connect(self.stop_calibration)

        self.prefill_example()

        self.tmr = QtCore.QTimer(self); self.tmr.setInterval(40); self.tmr.timeout.connect(self.update_solution)

    # ---------- утилиты
    def log_line(self, s): self.log.appendPlainText(s)
    def cal_log(self, s): self.calLog.appendPlainText(s)

    def prefill_example(self):
        # пример: 4 маяка
        for row in [(1,"COM8",0,10,0),(2,"COM14",10,0,0),(3,"COM15",-10,0,0),(4,"COM16",0,0,5)]:
            self.add_row(*row)
        # в таблице калибровки — только ID/COM
        for r in [(1,"COM8"),(2,"COM14"),(3,"COM15"),(4,"COM16")]:
            self.cal_add_row(*r)

    def add_row(self, id_val=None, port=None, x=0.0, y=0.0, z=0.0):
        r=self.table.rowCount(); self.table.insertRow(r)
        def cell(v): it=QtWidgets.QTableWidgetItem(str(v)); it.setTextAlignment(Qt.AlignCenter); return it
        self.table.setItem(r,0,cell(id_val if id_val is not None else r+1))
        self.table.setItem(r,1,cell(port or "COM6"))
        self.table.setItem(r,2,cell(x)); self.table.setItem(r,3,cell(y)); self.table.setItem(r,4,cell(z))
    def delete_selected(self):
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True): self.table.removeRow(r)
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

    # ---- старт/стоп трекинга
    def start_readers(self):
        self.stop_readers()
        self.filters.clear(); self.latest.clear(); self.hist.clear(); self.anchors.clear()
        for bid,port,pos in self.gather_beacons():
            self.filters[bid]=deque(maxlen=5)   # медиана по 5 последним «агрегатам»
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

    # ---- входящие расстояния
    @QtCore.pyqtSlot(int, float, object, str)
    def on_new_distance(self, bid, d_raw, rssi, rawline):
        # применяем калибровку маяка (если есть)
        d_corr = self.apply_calibration(bid, d_raw)

        # агрегируем в окно AGG_WINDOW_SEC
        ts = time.time()
        dq = self.hist.get(bid)
        if dq is None: dq = deque(); self.hist[bid]=dq
        dq.append((ts, d_corr))
        tmin = ts - AGG_WINDOW_SEC
        while dq and dq[0][0] < tmin: dq.popleft()

        # на каждый кадр будем использовать среднее по окну
        vals = [v for _,v in dq]
        if not vals: return
        d_mean = float(np.mean(vals))
        # медианное сглаживание «агрегатов»
        self.filters[bid].append(d_mean)
        d_f = float(np.median(self.filters[bid]))

        self.latest[bid] = (d_f, ts, rssi)

    # ---- апдейт решения
    def update_solution(self):
        now = time.time()
        ttl = self.spTTL.value()

        bids,A_xyz,dists,sigmas=[],[],[],[]
        for bid, anc in self.anchors.items():
            last = self.latest.get(bid)
            if not last: continue
            d,ts,_ = last
            if now - ts > ttl: continue
            # σ по текущему окну:
            dq = self.hist.get(bid)
            if dq:
                vs = [v for t,v in dq if now - t <= AGG_WINDOW_SEC]
                if vs:
                    std = float(np.std(vs))
                else:
                    std = 0.05
            else:
                std = 0.05
            std = max(0.05, std)

            bids.append(bid); A_xyz.append(anc); dists.append(d); sigmas.append(std)

        if len(bids)<3:
            self.canvas.draw_scene([(x,y,z,i) for i,(x,y,z) in self.anchors.items()], None)
            return

        A_xyz=np.array(A_xyz,float); dists=np.array(dists,float)
        used=np.arange(len(bids)); x0=self._last_pos
        # итеративный отсев выбросов
        for _ in range(2):
            pos_tmp = trilaterate_wls_robust(A_xyz[used], dists[used],
                                             x0=x0, iters=12, huber=0.4,
                                             step_max=TRUST_STEP_MAX,
                                             vel_soft=(self.posFilter.x, V_MAX, max(1e-3, now-(self.posFilter.t_prev or now))))
            if pos_tmp is None or not np.all(np.isfinite(pos_tmp)): break
            r = np.linalg.norm(A_xyz[used]-pos_tmp,axis=1) - dists[used]
            bad = np.where(np.abs(r)>RESID_THRESH)[0]
            if len(bad)==0 or len(used)-len(bad)<3:
                x0=pos_tmp; break
            used=np.delete(used,bad); x0=pos_tmp

        pos = trilaterate_wls_robust(A_xyz[used], dists[used], x0=x0, iters=12, huber=0.4,
                                     step_max=TRUST_STEP_MAX,
                                     vel_soft=(self.posFilter.x, V_MAX, max(1e-3, now-(self.posFilter.t_prev or now))))
        tracker=None
        if pos is not None and np.all(np.isfinite(pos)):
            if self.cb2D.isChecked(): pos[2]=float(self.edZfix.value())
            pos=np.clip(pos, SCENE_MIN, SCENE_MAX)
            self.posFilter.alpha=self.spPosAlpha.value(); self.posFilter.beta=self.spPosBeta.value()
            pos_sm=self.posFilter.update(pos)
            tracker=tuple(map(float,pos_sm)); self._last_pos=pos_sm
            res=np.linalg.norm(A_xyz[used]-pos,axis=1)-dists[used]; rms=float(np.sqrt(np.mean(res**2)))
            self.status.showMessage(f"Used {len(used)}/{len(bids)} | RMS {rms:.3f} m")
        self.canvas.draw_scene([(x,y,z,i) for i,(x,y,z) in self.anchors.items()], tracker, dists=dists.tolist(), autoscale=self.cbAutoscale.isChecked())

    # ---- калибровки: load/save/apply
    def load_calibration(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,"Загрузить калибровку","calibration.json","JSON (*.json)")
        if not path: return
        try:
            with open(path,'r',encoding='utf-8') as f:
                data=json.load(f)
            self.calibAB={}
            for k,v in data.items():
                if v["type"]=="linear":
                    self.calibAB[int(k)]=('linear', float(v["a"]), float(v["b"]))
                elif v["type"]=="quad":
                    self.calibAB[int(k)]=('quad', float(v["c2"]), float(v["a"]), float(v["b"]))
            self.log_line(f"Загружено калибровок: {len(self.calibAB)}")
        except Exception as e:
            self.log_line(f"Ошибка загрузки: {e}")
    def save_calibration(self):
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,"Сохранить калибровку","calibration.json","JSON (*.json)")
        if not path: return
        try:
            out={}
            for bid, model in self.calibAB.items():
                if model[0]=='linear':
                    _,a,b=model; out[str(bid)]={"type":"linear","a":a,"b":b}
                else:
                    _,c2,a,b=model; out[str(bid)]={"type":"quad","c2":c2,"a":a,"b":b}
            with open(path,'w',encoding='utf-8') as f:
                json.dump(out,f,ensure_ascii=False,indent=2)
            self.log_line(f"Сохранено: {path}")
        except Exception as e:
            self.log_line(f"Ошибка сохранения: {e}")
    def apply_calibration(self, bid, d_meas):
        m=self.calibAB.get(bid)
        if not m: return float(d_meas)
        if m[0]=='linear':
            _,a,b=m; return max(0.0, a*float(d_meas)+b)
        else:
            _,c2,a,b=m; return max(0.0, c2*float(d_meas)**2 + a*float(d_meas)+b)

    # ----------------- КАЛИБРОВКА -----------------
    def cal_add_row(self, id_val=None, port=None):
        r=self.cal_table.rowCount(); self.cal_table.insertRow(r)
        def cell(v): it=QtWidgets.QTableWidgetItem(str(v)); it.setTextAlignment(Qt.AlignCenter); return it
        self.cal_table.setItem(r,0,cell(id_val if id_val is not None else r+1))
        self.cal_table.setItem(r,1,cell(port or "COM6"))
    def cal_delete_selected(self):
        for r in sorted({i.row() for i in self.cal_table.selectedIndexes()}, reverse=True): self.cal_table.removeRow(r)
    def gather_cal_beacons(self):
        res=[]
        for r in range(self.cal_table.rowCount()):
            try:
                bid=int(self.cal_table.item(r,0).text()); port=self.cal_table.item(r,1).text().strip()
            except Exception as e:
                self.cal_log(f"Ошибка строки {r+1}: {e}"); continue
            res.append((bid,port))
        return res

    def start_calibration(self):
        self.stop_calibration()
        # читатели только для калибруемых портов
        self.cal_data = defaultdict(list)  # bid -> list of (d_meas_median, d_true)
        self.cal_hist = defaultdict(lambda: deque())
        self.cal_threads=[]
        for bid,port in self.gather_cal_beacons():
            th=SerialReader(bid,port); th.newDistance.connect(self.on_cal_distance); th.portStatus.connect(self.cal_log); th.start(); self.cal_threads.append(th)
        self.btnCalStart.setEnabled(False); self.btnCalStop.setEnabled(True)
        self.cal_log("Старт автокалибровки.\nИнструкция: для каждого маяка ставьте метку на расстояниях 1,2,3,4,5 метров и в ТРЁХ разных точках (углах) на каждом расстоянии. Держим метку неподвижно ~3 сек до звукового/текстового сигнала завершения шага.")
        QtCore.QTimer.singleShot(100, self._run_calibration_protocol)

    def stop_calibration(self):
        for th in getattr(self,'cal_threads',[]): th.stop(); th.wait(500)
        self.cal_threads=[]
        self.btnCalStart.setEnabled(True); self.btnCalStop.setEnabled(False)
        self.cal_log("Калибровка остановлена.")

    @QtCore.pyqtSlot(int, float, object, str)
    def on_cal_distance(self, bid, d_raw, rssi, raw):
        # в калибровке берём «как есть» (без модели)
        ts=time.time()
        dq=self.cal_hist[bid]
        dq.append((ts,float(d_raw)))
        # держим скользящее окно для медианы и σ
        while dq and ts - dq[0][0] > CALIB_HOLD_SEC: dq.popleft()

    def _run_calibration_protocol(self):
        # Пошаговый сценарий: для каждого маяка — R=1..5 ; для каждого R — три точки.
        for bid,_ in self.gather_cal_beacons():
            self.cal_log(f"\n=== Маяк ID {bid}: ===")
            for R in CALIB_R_LIST:
                for k in range(CALIB_PTS_PER_R):
                    self.cal_log(f"Поставьте метку на расстояние ~{R} м от маяка {bid}, точка {k+1}/{CALIB_PTS_PER_R}. Держите неподвижно {CALIB_HOLD_SEC:.0f} с…")
                    t0=time.time()
                    while time.time()-t0 < CALIB_HOLD_SEC:
                        QtWidgets.QApplication.processEvents()
                        time.sleep(0.02)
                    # берём медиану из окна
                    dq=self.cal_hist[bid]
                    vals=[v for _,v in dq]
                    if not vals:
                        self.cal_log("Нет данных — повтор шага.")
                        # повтор этой же точки
                        t0=time.time(); continue
                    d_med=float(np.median(vals)); d_std=float(np.std(vals)) if len(vals)>1 else 0.0
                    self.cal_data[bid].append((d_med, float(R)))
                    self.cal_log(f"Снято: d_meas_med={d_med:.3f} м (σ≈{d_std:.3f}) → d_true={R:.3f} м")

            # подгонка модели для маяка
            pairs=self.cal_data[bid]
            Dm=np.array([p[0] for p in pairs],float)
            Dt=np.array([p[1] for p in pairs],float)
            # сначала линейная
            X=np.vstack([Dm, np.ones_like(Dm)]).T
            a,b = np.linalg.lstsq(X, Dt, rcond=None)[0]
            Dt_hat = a*Dm + b
            lin_err = float(np.sqrt(np.mean((Dt_hat - Dt)**2)))
            # если нелинейность заметна — квадратичная
            if lin_err > 0.35:  # ~35 см RMSE — эмпирический порог
                X2=np.vstack([Dm**2, Dm, np.ones_like(Dm)]).T
                c2,a2,b2 = np.linalg.lstsq(X2, Dt, rcond=None)[0]
                Dt_hat2 = c2*Dm**2 + a2*Dm + b2
                quad_err = float(np.sqrt(np.mean((Dt_hat2 - Dt)**2)))
                if quad_err < lin_err*0.9:
                    self.calibAB[bid]=('quad', float(c2), float(a2), float(b2))
                    self.cal_log(f"Модель: QUAD (RMSE={quad_err:.3f} м)  => d_true = {c2:.4f} d^2 + {a2:.4f} d + {b2:.4f}")
                else:
                    self.calibAB[bid]=('linear', float(a), float(b))
                    self.cal_log(f"Модель: LINEAR (RMSE={lin_err:.3f} м) => d_true = {a:.4f} d + {b:.4f}")
            else:
                self.calibAB[bid]=('linear', float(a), float(b))
                self.cal_log(f"Модель: LINEAR (RMSE={lin_err:.3f} м) => d_true = {a:.4f} d + {b:.4f}")

        # автосейв
        try:
            out={}
            for bid, m in self.calibAB.items():
                if m[0]=='linear': _,a,b=m; out[str(bid)]={"type":"linear","a":a,"b":b}
                else: _,c2,a,b=m; out[str(bid)]={"type":"quad","c2":c2,"a":a,"b":b}
            with open("calibration.json","w",encoding="utf-8") as f: json.dump(out,f,ensure_ascii=False,indent=2)
            self.cal_log("\nКалибровка завершена. Файл сохранён: calibration.json")
        except Exception as e:
            self.cal_log(f"Ошибка сохранения calibration.json: {e}")
        self.btnCalStart.setEnabled(True); self.btnCalStop.setEnabled(False)

# -------------------- MAIN --------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
