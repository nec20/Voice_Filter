
import sys
import numpy as np
import sounddevice as sd
from scipy import signal
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from pyrnnoise.rnnoise import FRAME_SIZE, create, destroy, process_mono_frame

# --- SİSTEM PARAMETRELERİ ---
SAMPLE_RATE = 48000
CHUNK_SIZE = FRAME_SIZE  # 480 örnek


class ProAudioStudio(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Riley AI Studio - Pro Logic Controller")
        self.resize(1300, 950)

        # 1. PARAMETRELER (Arayüzden Kontrol Edilenler)
        self.params = {
            'rnn_en': True,
            'rnn_mix': 1.0,  # AI Temizlik Şiddeti (0.0 - 1.0)
            'gate_en': True,
            'gate_thr': 0.020,  # Kapı Eşiği
            'gate_speed': 0.2,  # Kapı Açılış/Kapanış Hızı (Attack/Release)
            'hpf_en': True,
            'hpf_cutoff': 100.0,
            'master_gain': 1.5
        }

        # 2. MOTORLAR VE DURUMLAR
        self.rnn_state = create()
        self.gate_current_gain = 1.0
        self.display_raw = np.zeros(CHUNK_SIZE)
        self.display_clean = np.zeros(CHUNK_SIZE)

        self.init_ui()
        self.start_audio()

    def init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # --- SOL PANEL: GELİŞMİŞ KONTROLLER ---
        ctrl_layout = QtWidgets.QVBoxLayout()
        ctrl_widget = QtWidgets.QWidget()
        ctrl_widget.setFixedWidth(400)
        ctrl_widget.setLayout(ctrl_layout)
        main_layout.addWidget(ctrl_widget)

        # Stil
        header_style = "font-weight: bold font-size: 13px color: #00FF00 margin-top: 10px"

        # A. RNNoise (AI) Kontrol Grubu
        rnn_group = QtWidgets.QGroupBox("RNNoise (Deep Learning)")
        rnn_v = QtWidgets.QVBoxLayout()
        self.rnn_cb = QtWidgets.QCheckBox("AI Temizleme Aktif")
        self.rnn_cb.setChecked(True)
        self.rnn_cb.toggled.connect(lambda v: self.set_p('rnn_en', v))
        rnn_v.addWidget(self.rnn_cb)
        self.add_pro_slider(rnn_v, "Temizlik Şiddeti (Mix)", 'rnn_mix', 0, 100, 100, 100, "%")
        rnn_group.setLayout(rnn_v)
        ctrl_layout.addWidget(rnn_group)

        # B. Noise Gate Kontrol Grubu
        gate_group = QtWidgets.QGroupBox("Noise Gate (Gürültü Kapısı)")
        gate_v = QtWidgets.QVBoxLayout()
        self.gate_cb = QtWidgets.QCheckBox("Gate Aktif")
        self.gate_cb.setChecked(True)
        self.gate_cb.toggled.connect(lambda v: self.set_p('gate_en', v))
        gate_v.addWidget(self.gate_cb)
        self.add_pro_slider(gate_v, "Threshold (Eşik)", 'gate_thr', 0, 200, 20, 1000, "")
        self.add_pro_slider(gate_v, "Tepki Hızı (Smooth)", 'gate_speed', 1, 100, 20, 100, "")
        gate_group.setLayout(gate_v)
        ctrl_layout.addWidget(gate_group)

        # C. Filtre ve Master
        self.add_pro_slider(ctrl_layout, "HPF Cutoff", 'hpf_cutoff', 20, 1000, 100, 1, "Hz")
        self.add_pro_slider(ctrl_layout, "Master Gain", 'master_gain', 1, 100, 15, 10, "x")

        ctrl_layout.addStretch()

        # --- SAĞ PANEL: GRAFİKLER ---
        self.win = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.win)

        self.p1 = self.win.addPlot(title="Zaman Analizi (Gri: Ham, Yeşil: İşlenmiş)")
        self.p1.setYRange(-0.6, 0.6)
        self.p1.showGrid(x=True, y=True)
        self.curve_raw = self.p1.plot(pen=pg.mkPen('#555555', width=1, style=QtCore.Qt.DotLine))
        self.curve_clean = self.p1.plot(pen=pg.mkPen('#00FF00', width=2))

        self.win.nextRow()
        self.p2 = self.win.addPlot(title="Frekans Spektrumu (dB Analizi)")
        self.p2.setLogMode(x=True, y=False)
        self.p2.setYRange(-70, 10)
        self.p2.showGrid(x=True, y=True)
        self.curve_fft = self.p2.plot(pen=pg.mkPen('#00FFFF', width=2))

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(30)

    def add_pro_slider(self, layout, label, key, min_v, max_v, def_v, div, unit):
        vbox = QtWidgets.QVBoxLayout()
        val_lbl = QtWidgets.QLabel(f"{label}: {def_v / div}{unit}")
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(def_v)
        slider.valueChanged.connect(
            lambda v: (self.set_p(key, v / div), val_lbl.setText(f"{label}: {v / div:.3f}{unit}")))
        vbox.addWidget(val_lbl)
        vbox.addWidget(slider)
        layout.addLayout(vbox)

    def set_p(self, key, val):
        self.params[key] = val

    def audio_callback(self, indata, outdata, frames, time, status):
        raw_in = indata.flatten().astype(np.float32)
        x = raw_in.copy()

        # 1. RNNoise + Mix (Intensity)
        if self.params['rnn_en']:
            x_int16 = (x * 32767).astype(np.int16)
            denoised_int16, _ = process_mono_frame(self.rnn_state, x_int16)
            denoised_float = denoised_int16.astype(np.float32) / 32767.0

            # Dry/Wet Mix: Orijinal ses ile AI sesini karıştır
            mix = self.params['rnn_mix']
            x = (denoised_float * mix) + (x * (1.0 - mix))

        # 2. HPF
        sos = signal.butter(4, self.params['hpf_cutoff'], 'hp', fs=SAMPLE_RATE, output='sos')
        x = signal.sosfilt(sos, x)

        # 3. Noise Gate (Gelişmiş)
        if self.params['gate_en']:
            rms = np.sqrt(np.mean(x ** 2))
            target = 1.0 if rms > self.params['gate_thr'] else 0.0

            # Tepki Hızı (Gate Speed)
            speed = self.params['gate_speed']
            self.gate_current_gain = ((1.0 - speed) * self.gate_current_gain) + (speed * target)
            x *= self.gate_current_gain

        # 4. Master Gain
        x = np.clip(x * self.params['master_gain'], -1.0, 1.0)

        self.display_raw = raw_in
        self.display_clean = x
        outdata[:] = x.reshape(-1, 1)

    def update_plots(self):
        self.curve_raw.setData(self.display_raw)
        self.curve_clean.setData(self.display_clean)
        if np.max(np.abs(self.display_clean)) > 0.0001:
            fft_mag = np.abs(np.fft.rfft(self.display_clean * np.hanning(CHUNK_SIZE)))
            freqs = np.fft.rfftfreq(CHUNK_SIZE, 1 / SAMPLE_RATE)
            self.curve_fft.setData(freqs[1:], 20 * np.log10(fft_mag[1:] + 1e-7))

    def start_audio(self):
        self.stream = sd.Stream(samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE,
                                channels=1, callback=self.audio_callback)
        self.stream.start()

    def closeEvent(self, event):
        destroy(self.rnn_state)
        self.stream.stop()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    lab = ProAudioStudio()
    lab.show()
    sys.exit(app.exec_())