import numpy as np
import librosa
import time
import pyaudio
import wave

class VoiceIdentificationModule:
    def __init__(self, sr=16000, record_seconds=3):
        """
        :param sr: Частота дискретизации для записи и обработки (Гц).
        :param record_seconds: сколько секунд записывать при идентификации/энролле.
        """
        self.sr = sr
        self.record_seconds = record_seconds
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1

        self.known_voices = {}  # Словарь {player_id: вектор_признаков (MFCC среднее)}

    def _record_audio(self, duration=None):
        """
        Записывает звук с микрофона заданное число секунд (duration), 
        возвращает путь к wav-файлу.
        """
        if duration is None:
            duration = self.record_seconds

        p = pyaudio.PyAudio()

        stream = p.open(format=self.format,
                        channels=self.channels,
                        rate=self.sr,
                        input=True,
                        frames_per_buffer=self.chunk)

        frames = []
        print(f"Начало записи на {duration} секунд...")
        start_time = time.time()

        while True:
            data = stream.read(self.chunk)
            frames.append(data)
            current_time = time.time()
            if (current_time - start_time) >= duration:
                break

        print("Запись завершена.")

        stream.stop_stream()
        stream.close()
        p.terminate()

        # Сохраним во временный файл (можно и в BytesIO, если хотите)
        filename = f"temp_record_{int(time.time())}.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(p.get_sample_size(self.format))
        wf.setframerate(self.sr)
        wf.writeframes(b''.join(frames))
        wf.close()

        return filename

    def _extract_features(self, wav_file):
        """
        Извлекает MFCC-признаки из переданного .wav-файла, 
        возвращает усреднённый вектор (наивно).
        """
        y, sr = librosa.load(wav_file, sr=self.sr)
        # mfcc: (n_mfcc, time) – например, (20, N)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        # Возьмём среднее по временной оси => получим (20,)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean

    def enroll_voice(self, player_id):
        """
        Запрашивает у пользователя запись эталонного голоса (self.record_seconds секунд),
        извлекает признаки и сохраняет их в known_voices.
        """
        print(f"Запись эталонного голоса для игрока {player_id}...")
        wav_path = self._record_audio()
        features = self._extract_features(wav_path)
        self.known_voices[player_id] = features
        print(f"Игрок {player_id} добавлен в known_voices.")

    def identify_voice(self):
        """
        Пишет новый фрагмент голоса, извлекает признаки и сравнивает 
        с эталонами (наивно по Евклидову расстоянию).
        Возвращает player_id или None.
        """
        print("Запись голоса для идентификации...")
        wav_path = self._record_audio()
        features = self._extract_features(wav_path)

        min_dist = float('inf')
        identified_player = None

        for pid, known_feat in self.known_voices.items():
            dist = np.linalg.norm(features - known_feat)
            if dist < min_dist:
                min_dist = dist
                identified_player = pid

        # Можно настроить порог: если расстояние слишком большое – не опознавать
        # Но в демо допустим, что минимальное расстояние всегда означает совпадение

        return identified_player