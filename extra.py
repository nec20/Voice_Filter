import sys
import numpy as np
import sounddevice as sd
from scipy import signal
from scipy.io import wavfile # WAV dosyası okumak için eklendi
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
            'rnn_mix': 1.0,
            'gate_en': True,
            'gate_thr': 0.020,
            'gate_speed': 0.2,
            'hpf_en': True,
            'hpf_cutoff': 100.0,
            'master_gain': 1.5
        }

        # --- YENİ: WAV Dosyası Durum Değişkenleri ---
        self.source_mode = "mic"  # "mic" veya "wav"
        self.wav_data = None
        self.wav_index = 0

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

        # --- YENİ: Ses Kaynağı Seçimi UI ---
        source_group = QtWidgets.QGroupBox("Ses Kaynağı Seçimi")
        source_v = QtWidgets.QVBoxLayout()
        
        self.radio_mic = QtWidgets.QRadioButton("Mikrofon (Canlı Akış)")
        self.radio_mic.setChecked(True)
        self.radio_mic.toggled.connect(self.change_source)
        
        self.radio_wav = QtWidgets.QRadioButton("Kayıtlı .WAV Dosyası")
        self.radio_wav.toggled.connect(self.change_source)
        
        self.btn_load_wav = QtWidgets.QPushButton("WAV Dosyası Seç ve Başlat")
        self.btn_load_wav.setEnabled(False) # Başlangıçta pasif
        self.btn_load_wav.clicked.connect(self.load_wav_file)
        
        source_v.addWidget(self.radio_mic)
        source_v.addWidget(self.radio_wav)
        source_v.addWidget(self.btn_load_wav)
        source_group.setLayout(source_v)
        ctrl_layout.addWidget(source_group)
        # -------------------------------------

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

    # --- YENİ: WAV ve Kaynak Kontrol Fonksiyonları ---
    def change_source(self):
        if self.radio_wav.isChecked():
            self.source_mode = "wav"
            self.btn_load_wav.setEnabled(True)
        else:
            self.source_mode = "mic"
            self.btn_load_wav.setEnabled(False)

    def load_wav_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Analiz İçin WAV Dosyası Seç', '', 'WAV Files (*.wav)')
        if fname:
            try:
                fs, data = wavfile.read(fname)
                
                # RNNoise 48kHz bekler, farklıysa uyar
                if fs != SAMPLE_RATE:
                    QtWidgets.QMessageBox.warning(self, "Uyarı", f"Dosya {fs}Hz. AI algoritması {SAMPLE_RATE}Hz ile çalışacak şekilde optimize edilmiştir. Ses hızı/perdesi bozuk çıkabilir.")
                
                # Stereo ise tek kanala (mono) düşür
                if len(data.shape) > 1:
                    data = data[:, 0]
                
                # Veriyi float32 ve -1.0 ile 1.0 arasına normalize et
                if data.dtype == np.int16:
                    self.wav_data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.float32:
                    self.wav_data = data
                else:
                    self.wav_data = data.astype(np.float32) / np.max(np.abs(data))
                
                self.wav_index = 0 # Dosyayı başa sar
                self.setWindowTitle(f"Riley AI Studio - Çalıyor: {fname.split('/')[-1]}")
                
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Hata", f"Dosya okunamadı:\n{str(e)}")
    # ------------------------------------------------

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
        # --- DEĞİŞEN: Veri Kaynağını Belirleme ---
        if self.source_mode == "mic":
            # Mikrofondan gelen ham veriyi al
            raw_in = indata.flatten().astype(np.float32)
        else:
            # WAV dosyasından CHUNK_SIZE kadar veri kopar
            if self.wav_data is not None and self.wav_index < len(self.wav_data):
                end_idx = min(self.wav_index + CHUNK_SIZE, len(self.wav_data))
                chunk = self.wav_data[self.wav_index : end_idx]
                
                # Dosya sonuna gelindiğinde eksik kalan kısmı sıfır (sessizlik) ile doldur
                if len(chunk) < CHUNK_SIZE:
                    raw_in = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)), 'constant')
                else:
                    raw_in = chunk
                
                self.wav_index += CHUNK_SIZE # Bir sonraki tur için indeksi ilerlet
            else:
                # Dosya yüklenmediyse veya bittiyse sessizlik ver
                raw_in = np.zeros(CHUNK_SIZE, dtype=np.float32)

        x = raw_in.copy()
        # -----------------------------------------

        # 1. RNNoise + Mix (Intensity)
        if self.params['rnn_en']:
            x_int16 = (x * 32767).astype(np.int16)
            denoised_int16, _ = process_mono_frame(self.rnn_state, x_int16)
            denoised_float = denoised_int16.astype(np.float32) / 32767.0

            mix = self.params['rnn_mix']
            x = (denoised_float * mix) + (x * (1.0 - mix))

        # 2. HPF
        if self.params['hpf_en']:
            sos = signal.butter(4, self.params['hpf_cutoff'], 'hp', fs=SAMPLE_RATE, output='sos')
            x = signal.sosfilt(sos, x)

        # 3. Noise Gate (Gelişmiş)
        if self.params['gate_en']:
            rms = np.sqrt(np.mean(x ** 2))
            target = 1.0 if rms > self.params['gate_thr'] else 0.0

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