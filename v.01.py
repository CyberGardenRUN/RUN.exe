# Let's write an updated Python GUI script with anchor status panel.
# This is a self-contained file. It extends the previous tdoa_view_calib.py capabilities
# by adding per-anchor status tracking based on JSON lines emitted by A0 that include
# RX metadata, e.g. {"rx":{"seq":123,"anchor":1,"rssi":-61}}.
# If such lines are not available, the panel still shows A0 USB link and marks A1..A3 as "unknown".

import json, sys, os, time, threading, csv, math, traceback
from collections import deque, defaultdict

from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
from pyqtgraph.opengl import GLViewWidget, GLGridItem, GLScatterPlotItem
import serial, serial.tools.list_ports

APP_TITLE = "TDoA Viewer & Calibrator (A0) — with Anchor Status"
DEFAULT_BAUD = 921600
ANCHOR_COUNT = 4  # A0..A3
ANCHOR_NAMES = ["A0(Server)", "A1", "A2", "A3"]
STATUS_TIMEOUT_SEC = 2.5     # if no RX from anchor in this window => offline
RATE_WINDOW_SEC = 10.0       # window for Hz estimation
CSV_HEADER = ["seq","ts","x","y","z","q","mode"]

pg.setConfigOptions(antialias=True)

# ---------- Utilities ----------
def list_serial_ports():
    return [p.device for p in serial.tools.list_ports.comports()]

def pretty_ts():
    return time.strftime("%H:%M:%S")

# ---------- Anchor status tracking ----------
class AnchorStatus:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_seen = [0.0]*ANCHOR_COUNT
        self.last_rssi = [None]*ANCHOR_COUNT
        self.rx_deque  = [deque() for _ in range(ANCHOR_COUNT)]  # store timestamps to compute Hz
        self.total_cnt = [0]*ANCHOR_COUNT
        self.connected_usb = False  # for A0 USB link state

    def mark_usb(self, ok: bool):
        with self.lock:
            self.connected_usb = ok
            self.last_seen[0] = time.time() if ok else self.last_seen[0]

    def update_anchor_rx(self, anchor_idx: int, rssi: int):
        t = time.time()
        with self.lock:
            if 0 <= anchor_idx < ANCHOR_COUNT:
                self.last_seen[anchor_idx] = t
                self.last_rssi[anchor_idx] = rssi
                dq = self.rx_deque[anchor_idx]
                dq.append(t)
                self.total_cnt[anchor_idx] += 1
                # purge old timestamps
                while dq and t - dq[0] > RATE_WINDOW_SEC:
                    dq.popleft()

    def snapshot(self):
        # return a dict with computed status per anchor
        t = time.time()
        out = []
        with self.lock:
            for i in range(ANCHOR_COUNT):
                online = (t - self.last_seen[i] < STATUS_TIMEOUT_SEC) if i>0 else self.connected_usb
                hz = 0.0
                dq = self.rx_deque[i]
                if len(dq) >= 2:
                    dt = dq[-1] - dq[0]
                    hz = (len(dq)-1)/dt if dt>1e-6 else 0.0
                out.append({
                    "name": ANCHOR_NAMES[i],
                    "online": online,
                    "last_rssi": self.last_rssi[i],
                    "rate_hz": hz,
                    "last_seen_age": (t - self.last_seen[i]) if self.last_seen[i]>0 else None,
                    "total": self.total_cnt[i],
                })
        return out

# ---------- Serial worker ----------
class SerialWorker(QtCore.QObject):
    line_parsed = QtCore.Signal(dict)        # emits parsed JSON dict
    log_text    = QtCore.Signal(str)         # text to append to log
    port_changed= QtCore.Signal(str)         # when connected / disconnected
    rx_meta     = QtCore.Signal(int, int)    # (anchor_idx, rssi)

    def __init__(self, status: AnchorStatus, parent=None):
        super().__init__(parent)
        self.status = status
        self._stop = threading.Event()
        self._ser = None
        self.port = None
        self.baud = DEFAULT_BAUD

    def stop(self):
        self._stop.set()
        try:
            if self._ser:
                self._ser.close()
        except:
            pass

    @QtCore.Slot(str, int)
    def open_port(self, port: str, baud: int):
        # close if different
        self.close_port()
        try:
            self._ser = serial.Serial(port=port, baudrate=baud, timeout=0.1)
            self.port = port
            self.baud = baud
            self.port_changed.emit(f"Connected: {port} @ {baud}")
            self.status.mark_usb(True)
            self.log_text.emit(f"[{pretty_ts()}] Serial connected: {port} @ {baud}\n")
        except Exception as e:
            self.port_changed.emit(f"Connect failed: {e}")
            self.log_text.emit(f"[{pretty_ts()}] ERROR open {port}: {e}\n")
            self._ser = None
            self.status.mark_usb(False)

    @QtCore.Slot()
    def close_port(self):
        if self._ser:
            try:
                self._ser.close()
            except:
                pass
        self._ser = None
        self.port = None
        self.port_changed.emit("Disconnected")
        self.status.mark_usb(False)

    @QtCore.Slot()
    def auto_connect(self):
        # try to connect to first available port
        ports = list_serial_ports()
        if not ports:
            self.log_text.emit(f"[{pretty_ts()}] No serial ports.\n")
            return
        for p in ports:
            try:
                self.open_port(p, self.baud)
                if self._ser:
                    return
            except:
                continue

    @QtCore.Slot(bytes)
    def send_bytes(self, data: bytes):
        if self._ser:
            try:
                self._ser.write(data)
            except Exception as e:
                self.log_text.emit(f"[{pretty_ts()}] TX error: {e}\n")

    @QtCore.Slot(str)
    def send_line(self, line: str):
        self.send_bytes((line+"\n").encode("ascii", errors="ignore"))

    @QtCore.Slot()
    def read_loop(self):
        buf = bytearray()
        while not self._stop.is_set():
            if not self._ser:
                time.sleep(0.2)
                continue
            try:
                data = self._ser.read(512)
                if data:
                    buf.extend(data)
                    while b'\n' in buf:
                        line, _, buf = buf.partition(b'\n')
                        s = line.decode('utf-8', errors='ignore').strip()
                        if not s:
                            continue
                        # parse JSON, tolerate trailing characters
                        obj = None
                        try:
                            obj = json.loads(s)
                        except Exception:
                            # try truncating after last '}' or ']'
                            r = s.rfind('}')
                            if r != -1:
                                try:
                                    obj = json.loads(s[:r+1])
                                except:
                                    pass
                        if obj is not None:
                            self.line_parsed.emit(obj)
                            # Optional: if A0 prints raw RX meta lines:
                            # e.g. {"rx":{"seq":123,"anchor":1,"rssi":-63}}
                            if isinstance(obj, dict) and "rx" in obj and isinstance(obj["rx"], dict):
                                a = obj["rx"].get("anchor", None)
                                rssi = obj["rx"].get("rssi", None)
                                if isinstance(a, int) and isinstance(rssi, int):
                                    self.rx_meta.emit(a, rssi)
                        else:
                            self.log_text.emit(s + "\n")
                else:
                    time.sleep(0.01)
            except (serial.SerialException, OSError) as e:
                self.log_text.emit(f"[{pretty_ts()}] Serial error: {e}\n")
                self.close_port()
                time.sleep(0.5)
            except Exception as e:
                self.log_text.emit(f"[{pretty_ts()}] Reader exception: {e}\n")
                self.log_text.emit(traceback.format_exc() + "\n")
                time.sleep(0.1)

# ---------- Main Window ----------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1200, 780)

        self.status_data = AnchorStatus()

        # Serial worker thread
        self.thread = QtCore.QThread(self)
        self.worker = SerialWorker(self.status_data)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.read_loop)
        self.worker.line_parsed.connect(self.on_line_parsed)
        self.worker.log_text.connect(self.append_log)
        self.worker.port_changed.connect(self.on_port_changed)
        self.worker.rx_meta.connect(self.on_rx_meta)
        self.thread.start()

        # --- UI ---
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        v = QtWidgets.QVBoxLayout(cw)

        # Top controls
        top = QtWidgets.QHBoxLayout()
        v.addLayout(top)

        self.port_cb = QtWidgets.QComboBox()
        self.refresh_ports()
        top.addWidget(QtWidgets.QLabel("Port:"))
        top.addWidget(self.port_cb)

        self.baud_cb = QtWidgets.QComboBox()
        for b in [921600, 460800, 230400, 115200]:
            self.baud_cb.addItem(str(b))
        self.baud_cb.setCurrentText(str(DEFAULT_BAUD))
        top.addWidget(QtWidgets.QLabel("Baud:"))
        top.addWidget(self.baud_cb)

        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_disconnect = QtWidgets.QPushButton("Disconnect")
        self.btn_auto = QtWidgets.QPushButton("Auto")
        top.addWidget(self.btn_connect)
        top.addWidget(self.btn_disconnect)
        top.addWidget(self.btn_auto)

        top.addStretch(1)

        # Bias controls
        self.bias_edits = [QtWidgets.QDoubleSpinBox() for _ in range(ANCHOR_COUNT)]
        for i,ed in enumerate(self.bias_edits):
            ed.setDecimals(1); ed.setRange(-1e6, 1e6); ed.setSingleStep(10.0)
            ed.setSuffix(" us")
        bias_box = QtWidgets.QGroupBox("Bias (us) A0..A3")
        hb = QtWidgets.QHBoxLayout(bias_box)
        for i,ed in enumerate(self.bias_edits):
            hb.addWidget(QtWidgets.QLabel(f"b{i}:"))
            hb.addWidget(ed)
        self.btn_get = QtWidgets.QPushButton("Get from A0 (G)")
        self.btn_set = QtWidgets.QPushButton("Set & Save (B i val)")
        self.btn_zero = QtWidgets.QPushButton("Zero all")
        hb.addWidget(self.btn_get); hb.addWidget(self.btn_set); hb.addWidget(self.btn_zero)
        v.addWidget(bias_box)

        # Anchor Status Panel
        self.status_table = QtWidgets.QTableWidget(ANCHOR_COUNT, 6)
        self.status_table.setHorizontalHeaderLabels(["Anchor","Online","Hz (10s)","RSSI","Last seen","Total"])
        self.status_table.verticalHeader().setVisible(False)
        self.status_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.status_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        v.addWidget(self.status_table)
        for i in range(ANCHOR_COUNT):
            self.status_table.setItem(i, 0, QtWidgets.QTableWidgetItem(ANCHOR_NAMES[i]))
            for c in range(1,6):
                self.status_table.setItem(i, c, QtWidgets.QTableWidgetItem("-"))
        self.status_timer = QtCore.QTimer(self)
        self.status_timer.timeout.connect(self.refresh_status_table)
        self.status_timer.start(500)

        # Split: left plots / right log
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        v.addWidget(split, 1)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)

        # 3D view
        self.gl = GLViewWidget()
        self.gl.setCameraPosition(distance=6)
        grid = GLGridItem()
        grid.scale(1,1,1)
        self.gl.addItem(grid)
        self.scatter = GLScatterPlotItem(pos=pg.np.array([[0,0,0]]), size=10.0)
        self.gl.addItem(self.scatter)
        lv.addWidget(self.gl, 3)

        # XY plots
        self.plot_x = pg.PlotWidget(title="X (m)")
        self.plot_y = pg.PlotWidget(title="Y (m)")
        self.plot_z = pg.PlotWidget(title="Z (m)")
        for pw in (self.plot_x, self.plot_y, self.plot_z):
            pw.showGrid(x=True, y=True, alpha=0.3)
        lv.addWidget(self.plot_x,1); lv.addWidget(self.plot_y,1); lv.addWidget(self.plot_z,1)

        split.addWidget(left)

        # Right panel: log + controls
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)

        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True)
        rv.addWidget(self.log, 1)

        # CSV logging
        log_line = QtWidgets.QHBoxLayout()
        self.btn_csv = QtWidgets.QPushButton("Start CSV Log")
        self.csv_active = False
        self.csv_file = None; self.csv_writer = None
        log_line.addWidget(self.btn_csv)
        log_line.addStretch(1)
        rv.addLayout(log_line)

        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        # Data buffers for plots
        self.buf_len = 1000
        self.ts_buf = deque(maxlen=self.buf_len)
        self.x_buf = deque(maxlen=self.buf_len)
        self.y_buf = deque(maxlen=self.buf_len)
        self.z_buf = deque(maxlen=self.buf_len)

        self.curves = {
            "x": self.plot_x.plot(pen='y'),
            "y": self.plot_y.plot(pen='c'),
            "z": self.plot_z.plot(pen='m'),
        }

        # signals
        self.btn_connect.clicked.connect(self.on_connect_clicked)
        self.btn_disconnect.clicked.connect(lambda: self.worker.close_port())
        self.btn_auto.clicked.connect(lambda: self.worker.auto_connect())
        self.btn_get.clicked.connect(lambda: self.worker.send_line("G"))
        self.btn_zero.clicked.connect(self.on_zero_clicked)
        self.btn_set.clicked.connect(self.on_set_clicked)
        self.btn_csv.clicked.connect(self.on_csv_clicked)

        # Start with auto-connect trial
        QtCore.QTimer.singleShot(200, lambda: self.worker.auto_connect())

    # ----- UI handlers -----
    def refresh_ports(self):
        self.port_cb.clear()
        self.port_cb.addItems(list_serial_ports())

    def on_connect_clicked(self):
        p = self.port_cb.currentText()
        try:
            b = int(self.baud_cb.currentText())
        except:
            b = DEFAULT_BAUD
        self.worker.open_port(p, b)

    def on_zero_clicked(self):
        for ed in self.bias_edits:
            ed.setValue(0.0)
        self.on_set_clicked()

    def on_set_clicked(self):
        # send B i val for each bias
        for i,ed in enumerate(self.bias_edits):
            v = int(round(ed.value()))
            self.worker.send_line(f"B {i} {v}")
            time.sleep(0.01)  # small spacing

    def on_csv_clicked(self):
        if not self.csv_active:
            fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", "tdoa_log.csv", "CSV Files (*.csv)")
            if not fn: return
            try:
                self.csv_file = open(fn, "w", newline='', encoding='utf-8')
                self.csv_writer = csv.writer(self.csv_file)
                self.csv_writer.writerow(CSV_HEADER)
                self.csv_active = True
                self.btn_csv.setText("Stop CSV Log")
                self.append_log(f"[{pretty_ts()}] CSV logging to {fn}\n")
            except Exception as e:
                self.append_log(f"[{pretty_ts()}] CSV open error: {e}\n")
        else:
            try:
                if self.csv_file:
                    self.csv_file.close()
            except:
                pass
            self.csv_file = None; self.csv_writer=None; self.csv_active=False
            self.btn_csv.setText("Start CSV Log")
            self.append_log(f"[{pretty_ts()}] CSV logging stopped\n")

    def on_port_changed(self, s: str):
        self.statusBar().showMessage(s)

    def append_log(self, s: str):
        self.log.appendPlainText(s.rstrip())

    @QtCore.Slot(int, int)
    def on_rx_meta(self, anchor_idx: int, rssi: int):
        self.status_data.update_anchor_rx(anchor_idx, rssi)

    def refresh_status_table(self):
        snap = self.status_data.snapshot()
        for i, st in enumerate(snap):
            # Online column: colored dot + text
            online = st["online"]
            item1 = self.status_table.item(i,1)
            text = "ONLINE" if online else ("OFF" if st["last_seen_age"] is not None else "—")
            item1.setText(text)
            color = QtGui.QColor(0,180,0) if online else (QtGui.QColor(200,80,0) if st["last_seen_age"] is not None else QtGui.QColor(120,120,120))
            item1.setForeground(QtGui.QBrush(color))

            # Hz
            hz = st["rate_hz"]
            self.status_table.item(i,2).setText(f"{hz:,.1f}" if hz>0 else "-")
            # RSSI
            rssi = st["last_rssi"]
            self.status_table.item(i,3).setText(str(rssi) if rssi is not None else "-")
            # Last seen age
            age = st["last_seen_age"]
            self.status_table.item(i,4).setText(f"{age:.1f}s" if age is not None else "-")
            # Total
            self.status_table.item(i,5).setText(str(st["total"]))

    # ----- Data path -----
    @QtCore.Slot(dict)
    def on_line_parsed(self, obj: dict):
        # recognize bias get response
        if isinstance(obj, dict) and obj.get("cal") == "bias_get":
            vals = [obj.get(f"b{i}", 0.0) for i in range(ANCHOR_COUNT)]
            for i,v in enumerate(vals):
                try:
                    self.bias_edits[i].setValue(float(v))
                except:
                    pass
            self.append_log(f"[{pretty_ts()}] Bias from A0: {vals}\n")
            return

        # Standard solution JSON
        # {"seq":123, "x":1.23,"y":..., "z":..., "q":0.001, "mode":"RUN"}
        if all(k in obj for k in ("seq","x","y","z","q")):
            seq = int(obj.get("seq", 0))
            x = float(obj.get("x", 0.0))
            y = float(obj.get("y", 0.0))
            z = float(obj.get("z", 0.0))
            q = float(obj.get("q", 0.0))
            mode = str(obj.get("mode", ""))

            t = time.time()
            self.ts_buf.append(t)
            self.x_buf.append(x); self.y_buf.append(y); self.z_buf.append(z)

            # update plots
            self.curves["x"].setData(list(self.ts_buf), list(self.x_buf))
            self.curves["y"].setData(list(self.ts_buf), list(self.y_buf))
            self.curves["z"].setData(list(self.ts_buf), list(self.z_buf))

            # update 3D scatter
            import numpy as np
            self.scatter.setData(pos=np.array([[x,y,z]]))

            # CSV
            if self.csv_active and self.csv_writer:
                try:
                    self.csv_writer.writerow([seq, t, x, y, z, q, mode])
                except:
                    pass
            return

        # If A0 outputs custom status lines, pass-through to log
        self.append_log(json.dumps(obj, ensure_ascii=False))

    def closeEvent(self, e: QtGui.QCloseEvent):
        try:
            self.worker.stop()
        except:
            pass
        try:
            self.thread.quit()
            self.thread.wait(1000)
        except:
            pass
        super().closeEvent(e)

# ---------- Entry ----------
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
