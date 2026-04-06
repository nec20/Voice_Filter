import sys
import queue
import threading
import numpy as np
import sounddevice as sd
from scipy import signal
from scipy.io import wavfile
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from pyrnnoise.rnnoise import FRAME_SIZE, create, destroy, process_mono_frame

# --- SİSTEM PARAMETRELERİ ---
SYSTEM_SAMPLE_RATE = 16000
INTERNAL_SAMPLE_RATE = 48000
RESAMPLE_RATIO = 3
CHUNK_SIZE_16K = 160
CHUNK_SIZE_48K = FRAME_SIZE

class ProAudioStudio(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Riley AI Studio - Timeline & Playback Controller")
        self.resize(1300, 950)

        # 1. PARAMETRELER
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

        # 2. WAV & OYNATMA DURUMLARI (YENİ EKLENENLER)
        self.source_mode = "mic"
        self.wav_data = None
        self.wav_index = 0
        self.is_playing = True # Oynatma durumu
        self.is_dragging = False # Kullanıcı çizgiyi sürüklüyor mu?

        # 3. YAYIN (STREAMING) KUYRUĞU
        self.stream_queue = queue.Queue()
        self.is_streaming = True

        # 4. MOTORLAR VE DURUMLAR
        self.rnn_state = create()
        self.gate_current_gain = 1.0
        
        self.display_raw = np.zeros(CHUNK_SIZE_16K)
        self.display_clean = np.zeros(CHUNK_SIZE_16K)

        self.init_ui()
        self.start_audio()
        
        # STT Streaming Thread
        self.stt_thread = threading.Thread(target=self.stt_stream_worker, daemon=True)
        self.stt_thread.start()

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

        # Ses Kaynağı Seçimi
        source_group = QtWidgets.QGroupBox("Ses Kaynağı & Çalar (Player)")
        source_v = QtWidgets.QVBoxLayout()
        
        self.radio_mic = QtWidgets.QRadioButton("Mikrofon (Canlı Akış)")
        self.radio_mic.setChecked(True)
        self.radio_mic.toggled.connect(self.change_source)
        
        self.radio_wav = QtWidgets.QRadioButton("Kayıtlı .WAV Dosyası")
        self.radio_wav.toggled.connect(self.change_source)
        
        self.btn_load_wav = QtWidgets.QPushButton("📂 WAV Dosyası Yükle")
        self.btn_load_wav.setEnabled(False)
        self.btn_load_wav.clicked.connect(self.load_wav_file)
        
        # --- YENİ: OYNATMA KONTROLLERİ ---
        play_layout = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("▶ Oynat")
        self.btn_pause = QtWidgets.QPushButton("⏸ Duraklat")
        self.btn_stop = QtWidgets.QPushButton("⏹ Başa Sar")
        
        self.btn_play.clicked.connect(self.play_wav)
        self.btn_pause.clicked.connect(self.pause_wav)
        self.btn_stop.clicked.connect(self.stop_wav)
        
        # Başlangıçta inaktif olsunlar
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        
        play_layout.addWidget(self.btn_play)
        play_layout.addWidget(self.btn_pause)
        play_layout.addWidget(self.btn_stop)
        
        source_v.addWidget(self.radio_mic)
        source_v.addWidget(self.radio_wav)
        source_v.addWidget(self.btn_load_wav)
        source_v.addLayout(play_layout) # Kontrolleri ekle
        source_group.setLayout(source_v)
        ctrl_layout.addWidget(source_group)

        # RNNoise (AI) Kontrol Grubu
        rnn_group = QtWidgets.QGroupBox("RNNoise (Deep Learning)")
        rnn_v = QtWidgets.QVBoxLayout()
        self.rnn_cb = QtWidgets.QCheckBox("AI Temizleme Aktif")
        self.rnn_cb.setChecked(True)
        self.rnn_cb.toggled.connect(lambda v: self.set_p('rnn_en', v))
        rnn_v.addWidget(self.rnn_cb)
        self.add_pro_slider(rnn_v, "Temizlik Şiddeti (Mix)", 'rnn_mix', 0, 100, 100, 100, "%")
        rnn_group.setLayout(rnn_v)
        ctrl_layout.addWidget(rnn_group)

        # Noise Gate Kontrol Grubu
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

        # Filtre ve Master
        self.add_pro_slider(ctrl_layout, "HPF Cutoff", 'hpf_cutoff', 20, 1000, 100, 1, "Hz")
        self.add_pro_slider(ctrl_layout, "Master Gain", 'master_gain', 1, 100, 15, 10, "x")

        ctrl_layout.addStretch()

        # --- SAĞ PANEL: GRAFİKLER ---
        self.win = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.win)

        self.p1 = self.win.addPlot(title="Anlık Zaman Analizi (10ms)")
        self.p1.setYRange(-0.6, 0.6)
        self.p1.showGrid(x=True, y=True)
        self.curve_raw = self.p1.plot(pen=pg.mkPen('#555555', width=1, style=QtCore.Qt.DotLine))
        self.curve_clean = self.p1.plot(pen=pg.mkPen('#00FF00', width=2))

        self.win.nextRow()
        self.p2 = self.win.addPlot(title="Frekans Spektrumu")
        self.p2.setLogMode(x=True, y=False)
        self.p2.setYRange(-70, 10)
        self.p2.showGrid(x=True, y=True)
        self.curve_fft = self.p2.plot(pen=pg.mkPen('#00FFFF', width=2))

        # --- YENİ: TÜM DOSYA TIMELINE GRAFİĞİ ---
        self.win.nextRow()
        self.p3 = self.win.addPlot(title="Tüm Dosya (Zaman Çizelgesi) - Tıklayarak veya Çizgiyi Sürükleyerek İleri/Geri Sarın")
        self.p3.showGrid(x=True, y=False)
        self.curve_full_wav = self.p3.plot(pen=pg.mkPen('#666666', width=1))
        
        # Oynatma Başlığı (Kırmızı dikey çizgi)
        self.playhead = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('r', width=2))
        self.p3.addItem(self.playhead)
        
        # Kullanıcı çizgiyi sürüklediğinde indeks güncellensin
        self.playhead.sigDragged.connect(self.on_playhead_dragged)
        self.playhead.sigPositionChangeFinished.connect(self.on_playhead_released)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(30)

    # --- OYNATMA FONKSİYONLARI ---
    def play_wav(self):
        self.is_playing = True
    
    def pause_wav(self):
        self.is_playing = False
        
    def stop_wav(self):
        self.is_playing = False
        self.wav_index = 0
        self.playhead.setValue(0) # Çizgiyi başa al

    def on_playhead_dragged(self):
        self.is_dragging = True # Sürüklerken timer'ın çizgiyi bozmasını engelle
        if self.wav_data is not None:
            new_pos = int(self.playhead.value())
            # Sınırların dışına çıkmasını engelle
            self.wav_index = max(0, min(new_pos, len(self.wav_data) - 1))

    def on_playhead_released(self):
        self.is_dragging = False

    def change_source(self):
        if self.radio_wav.isChecked():
            self.source_mode = "wav"
            self.btn_load_wav.setEnabled(True)
            self.btn_play.setEnabled(True)
            self.btn_pause.setEnabled(True)
            self.btn_stop.setEnabled(True)
            self.is_playing = False # Varsayılan olarak duraklatılmış başlasın
        else:
            self.source_mode = "mic"
            self.btn_load_wav.setEnabled(False)
            self.btn_play.setEnabled(False)
            self.btn_pause.setEnabled(False)
            self.btn_stop.setEnabled(False)
            self.is_playing = True # Mikrofon her zaman akar

    def load_wav_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Analiz İçin WAV Dosyası Seç', '', 'WAV Files (*.wav)')
        if fname:
            try:
                fs, data = wavfile.read(fname)
                if fs != SYSTEM_SAMPLE_RATE:
                    QtWidgets.QMessageBox.warning(self, "Uyarı", f"Dosya {fs}Hz. Sistem {SYSTEM_SAMPLE_RATE}Hz bekliyor.")
                
                if len(data.shape) > 1:
                    data = data[:, 0]
                
                if data.dtype == np.int16:
                    self.wav_data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.float32:
                    self.wav_data = data
                else:
                    self.wav_data = data.astype(np.float32) / np.max(np.abs(data))
                
                self.wav_index = 0
                self.setWindowTitle(f"Riley AI Studio - Yüklendi: {fname.split('/')[-1]}")
                
                # --- YENİ: Dalga Formunu Çiz ---
                # Arayüzü dondurmamak için veriyi seyrekleştirerek (downsample) çiziyoruz
                step = max(1, len(self.wav_data) // 5000) 
                x_axis = np.arange(0, len(self.wav_data), step)
                y_axis = self.wav_data[::step]
                
                self.curve_full_wav.setData(x_axis, y_axis)
                self.p3.setXRange(0, len(self.wav_data))
                self.playhead.setBounds([0, len(self.wav_data)])
                self.playhead.setValue(0)
                
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Hata", f"Dosya okunamadı:\n{str(e)}")

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
        # 1. VERİ KAYNAĞI SEÇİMİ
        if self.source_mode == "mic":
            raw_in = indata.flatten().astype(np.float32)
        else:
            # WAV modunda, eğer oynatılıyorsa ve dosya varsa veriyi al
            if self.is_playing and self.wav_data is not None and self.wav_index < len(self.wav_data):
                end_idx = min(self.wav_index + CHUNK_SIZE_16K, len(self.wav_data))
                chunk = self.wav_data[self.wav_index : end_idx]
                
                if len(chunk) < CHUNK_SIZE_16K:
                    raw_in = np.pad(chunk, (0, CHUNK_SIZE_16K - len(chunk)), 'constant')
                else:
                    raw_in = chunk
                
                self.wav_index += CHUNK_SIZE_16K
            else:
                # Oynatılmıyorsa (Pause) veya dosya bittiyse sessizlik ver
                raw_in = np.zeros(CHUNK_SIZE_16K, dtype=np.float32)

        x_16k = raw_in.copy()

        # 2. YAPAY ZEKA ARA KATMANI (16kHz -> 48kHz -> 16kHz)
        if self.params['rnn_en'] and np.any(x_16k): # Sadece ses varsa AI çalışsın (CPU tasarrufu)
            x_48k = signal.resample_poly(x_16k, RESAMPLE_RATIO, 1)
            
            x_int16 = (x_48k * 32767).astype(np.int16)
            denoised_int16, _ = process_mono_frame(self.rnn_state, x_int16)
            denoised_float_48k = denoised_int16.astype(np.float32) / 32767.0

            mix = self.params['rnn_mix']
            mixed_48k = (denoised_float_48k * mix) + (x_48k * (1.0 - mix))
            x_16k = signal.resample_poly(mixed_48k, 1, RESAMPLE_RATIO)

        # 3. YÜKSEK GEÇİREN FİLTRE
        if self.params['hpf_en']:
            sos = signal.butter(4, self.params['hpf_cutoff'], 'hp', fs=SYSTEM_SAMPLE_RATE, output='sos')
            x_16k = signal.sosfilt(sos, x_16k)

        # 4. NOISE GATE
        if self.params['gate_en']:
            rms = np.sqrt(np.mean(x_16k ** 2))
            target = 1.0 if rms > self.params['gate_thr'] else 0.0
            speed = self.params['gate_speed']
            self.gate_current_gain = ((1.0 - speed) * self.gate_current_gain) + (speed * target)
            x_16k *= self.gate_current_gain

        # 5. MASTER GAIN & KLİPLEME
        x_16k = np.clip(x_16k * self.params['master_gain'], -1.0, 1.0)

        # 6. YAYIN KUYRUĞUNA GÖNDER
        if self.is_playing:
            self.stream_queue.put(x_16k.copy())

        # 7. ÇIKIŞ
        self.display_raw = raw_in
        self.display_clean = x_16k
        outdata[:] = x_16k.reshape(-1, 1)

    def stt_stream_worker(self):
        while self.is_streaming:
            try:
                ses_paketi = self.stream_queue.get(timeout=1)
                pass
            except queue.Empty:
                continue

    def update_plots(self):
        # 1. Anlık grafikleri güncelle
        self.curve_raw.setData(self.display_raw)
        self.curve_clean.setData(self.display_clean)
        
        # FFT güncellemesi
        if np.max(np.abs(self.display_clean)) > 0.0001:
            fft_mag = np.abs(np.fft.rfft(self.display_clean * np.hanning(CHUNK_SIZE_16K)))
            freqs = np.fft.rfftfreq(CHUNK_SIZE_16K, 1 / SYSTEM_SAMPLE_RATE)
            self.curve_fft.setData(freqs[1:], 20 * np.log10(fft_mag[1:] + 1e-7))
            
        # 2. Timeline Çizgisini (Playhead) Güncelle
        # Eğer kullanıcı mouse ile çizgiyi sürüklemiyorsa ve WAV modundaysak çizgiyi otomatik ilerlet
        if not self.is_dragging and self.source_mode == 'wav' and self.wav_data is not None:
            self.playhead.setValue(self.wav_index)

    def start_audio(self):
        self.stream = sd.Stream(samplerate=SYSTEM_SAMPLE_RATE, blocksize=CHUNK_SIZE_16K,
                                channels=1, callback=self.audio_callback)
        self.stream.start()

    def closeEvent(self, event):
        self.is_streaming = False
        destroy(self.rnn_state)
        self.stream.stop()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    lab = ProAudioStudio()
    lab.show()
    sys.exit(app.exec_())