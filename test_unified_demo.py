# test_unified_demo_deepface.py

import cv2
import numpy as np
import os
import time
import sys
import threading
import queue
import mediapipe as mp
import math

import pyaudio
import wave
import librosa
import speech_recognition as sr

from deepface import DeepFace

#######################################################
# 1. КЛАСС РАСПОЗНАВАНИЯ ЖЕСТОВ (Mediapipe)
#######################################################
class GestureInterpreter:
    """
    Расширенная логика жестов:
      - Прямая ладошка
      - Жест OK
      - Палец вверх (мирный/да)
      - Палец вниз (мафия/нет)
      - Готов к выстрелу (указательный и средний вытянуты, безымянный и мизинец согнуты)
      - Выстрел (фиксируется при смене с "готов к выстрелу" на "согнутые палцы")
      - Неопределённый жест
    """
    def __init__(self, mode=False, max_hands=4, detection_confidence=0.5, tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.shoot_in_progress = {}  # Для отслеживания "готов к выстрелу"
        self.prev_gesture = {}       # Запоминаем предыдущие жесты (при желании)

    def _is_hand_open(self, hand_landmarks):
        lm = hand_landmarks.landmark
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        open_fingers = 0
        for tip, pip in zip(finger_tips, finger_pips):
            if lm[tip].y < lm[pip].y:
                open_fingers += 1
        return open_fingers == 4

    def _is_ok_gesture(self, hand_landmarks):
        lm = hand_landmarks.landmark
        dx = lm[4].x - lm[8].x
        dy = lm[4].y - lm[8].y
        distance = math.hypot(dx, dy)

        # "Размер" руки
        hand_size = math.hypot(lm[0].x - lm[12].x, lm[0].y - lm[12].y)
        ratio = distance / hand_size if hand_size > 0 else 1

        bent_index = lm[8].y > lm[6].y
        extended_middle = lm[12].y < lm[10].y
        extended_ring   = lm[16].y < lm[14].y
        extended_pinky  = lm[20].y < lm[18].y

        return (ratio < 0.4) and bent_index and extended_middle and extended_ring and extended_pinky

    def _is_thumbs_up(self, hand_landmarks):
        lm = hand_landmarks.landmark
        wrist_y = lm[0].y
        thumb_tip_y = lm[4].y
        # большой палец выше запястья
        thumb_up = thumb_tip_y < wrist_y

        tips = [8,12,16,20]
        pips = [6,10,14,18]
        folded = 0
        for tip, pip in zip(tips, pips):
            if lm[tip].y > lm[pip].y:
                folded += 1
        return thumb_up and (folded == 4)

    def _is_thumbs_down(self, hand_landmarks):
        lm = hand_landmarks.landmark
        wrist_y = lm[0].y
        thumb_tip_y = lm[4].y
        thumb_down = thumb_tip_y > wrist_y

        tips = [8,12,16,20]
        pips = [6,10,14,18]
        folded = 0
        for tip, pip in zip(tips, pips):
            if lm[tip].y > lm[pip].y:
                folded += 1
        return thumb_down and (folded == 4)

    def _is_shoot_gesture(self, hand_landmarks):
        lm = hand_landmarks.landmark
        index_extended = lm[8].y < lm[6].y
        middle_extended = lm[12].y < lm[10].y
        ring_folded = lm[16].y > lm[14].y
        pinky_folded = lm[20].y > lm[18].y
        return index_extended and middle_extended and ring_folded and pinky_folded

    def _is_shoot_release(self, hand_landmarks):
        # отпускаем "выстрел": указательный и средний теперь согнуты
        lm = hand_landmarks.landmark
        index_folded = lm[8].y > lm[6].y
        middle_folded = lm[12].y > lm[10].y
        return index_folded and middle_folded

    def interpret_gestures(self, frame):
        """
        Возвращает список (hand_index, gesture_name).
        Если нет рук, возвращаем [(-1, "Нет рук")].
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return [(-1, "Нет рук")]

        # Синхронизируем словари
        active_ids = set(range(len(results.multi_hand_landmarks)))
        for k in list(self.shoot_in_progress.keys()):
            if k not in active_ids:
                del self.shoot_in_progress[k]

        gestures_info = []
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            gesture = ""
            if self._is_thumbs_up(hand_landmarks):
                gesture = "Палец вверх"
                self.shoot_in_progress[i] = False
            elif self._is_thumbs_down(hand_landmarks):
                gesture = "Палец вниз"
                self.shoot_in_progress[i] = False
            elif self._is_ok_gesture(hand_landmarks):
                gesture = "Жест OK"
                self.shoot_in_progress[i] = False
            else:
                # Логика "выстрела"
                if i not in self.shoot_in_progress:
                    self.shoot_in_progress[i] = False

                if self.shoot_in_progress[i]:
                    if self._is_shoot_release(hand_landmarks):
                        gesture = "Выстрел!"
                        self.shoot_in_progress[i] = False
                    else:
                        gesture = "Готов к выстрелу"
                else:
                    if self._is_shoot_gesture(hand_landmarks):
                        self.shoot_in_progress[i] = True
                        gesture = "Готов к выстрелу"
                    elif self._is_hand_open(hand_landmarks):
                        gesture = "Прямая ладошка"
                    else:
                        gesture = "Неопределённый жест"

            gestures_info.append((i, gesture))
        return gestures_info

###################################################
# 2. КЛАСС НАИВНОЙ ИДЕНТИФИКАЦИИ ГОЛОСА (MFCC)
###################################################
class VoiceIdentificationModule:
    def __init__(self, sr=16000):
        self.sr = sr
        self.known_voices = {}  # {player_id: mfcc_vector}

    def _extract_features(self, wav_file):
        y, sr = librosa.load(wav_file, sr=self.sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean

    def enroll_voice(self, player_id, duration=3):
        print(f"[VOICE] Запись голоса {player_id} на {duration} сек.")
        fname = record_audio(duration=duration, sr=self.sr, channels=1)
        feats = self._extract_features(fname)
        self.known_voices[player_id] = feats
        print(f"[VOICE] Игрок {player_id} – голосовой эталон сохранён.")

    def identify_voice(self, wav_file):
        feats = self._extract_features(wav_file)
        min_dist = float('inf')
        identified = None
        for pid, known_feat in self.known_voices.items():
            dist = np.linalg.norm(feats - known_feat)
            if dist < min_dist:
                min_dist = dist
                identified = pid
        return identified

###################################################
# 3. РАСПОЗНАВАНИЕ ЛИЦ (DeepFace)
###################################################
class DeepFaceRecognitionModule:
    """
    - Используем DeepFace для энролла: делаем снимок веб-камерой, сохраняем в faces_db/<player_id>.jpg
    - Для распознавания: на каждом кадре:
        1) сохраняем кадр во временный файл
        2) вызываем extract_faces => получаем bounding box по каждому лицу
        3) вызываем find => получаем список DataFrame (по одному на лицо), чтобы определить, кто это
        4) сопоставляем bounding box и имя игрока
    """
    def __init__(self, db_path="faces_db", detector_backend="retinaface"):
        self.db_path = db_path
        self.detector_backend = detector_backend
        os.makedirs(db_path, exist_ok=True)

    def enroll_face(self, player_id):
        """
        Делает одиночный снимок с веб-камеры, сохраняет как faces_db/ИгрокX.jpg.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[FACE] Камера не открывается, пропускаем энролл.")
            return
        print(f"[FACE] {player_id}, смотрите в камеру 3 секунды...")
        time.sleep(3)

        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("[FACE] Не удалось прочитать кадр.")
            return

        out_path = os.path.join(self.db_path, f"{player_id}.jpg")
        cv2.imwrite(out_path, frame)
        print(f"[FACE] Лицо {player_id} сохранено в {out_path}")

    def recognize_faces(self, frame):
        """
        Возвращает список: [ (top, right, bottom, left, player_id_или_None), ... ]

        1) extract_faces => список dict, в каждом { "face": np.array, "facial_area": {...} }
        2) find => список DF (по одному на каждое лицо)
        """
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)

        try:
            # 1. Извлекаем bounding box каждого лица
            faces_info = DeepFace.extract_faces(
                img_path=temp_path,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )

            # 2. Ищем совпадение по базе (возвращает list из DataFrame, если несколько лиц)
            df_list = DeepFace.find(
                img_path=temp_path,
                db_path=self.db_path,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            # Если только 1 лицо, find вернёт DataFrame, а не list
            if not isinstance(df_list, list):
                df_list = [df_list]

            results = []
            # faces_info[i]["facial_area"] => {"x":..., "y":..., "w":..., "h":...}
            # df_list[i] => DataFrame (или пустой)
            for i, face_dict in enumerate(faces_info):
                area = face_dict["facial_area"]  # x,y,w,h
                x = area["x"]
                y = area["y"]
                w = area["w"]
                h = area["h"]

                top = y
                left = x
                right = x + w
                bottom = y + h

                # Смотрим, есть ли df_list[i]
                if i < len(df_list):
                    df = df_list[i]
                    if len(df) > 0:
                        # Берём первую (наиболее похожую) строку
                        best_match = df.iloc[0]
                        identity_path = best_match["identity"]  # например faces_db/Игрок1.jpg
                        # Извлечём имя файла как "Игрок1"
                        base_name = os.path.basename(identity_path)
                        player_id = os.path.splitext(base_name)[0]
                        results.append((top, right, bottom, left, player_id))
                    else:
                        # Нет совпадений
                        results.append((top, right, bottom, left, None))
                else:
                    # Нет DataFrame для этого лица
                    results.append((top, right, bottom, left, None))

            return results

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

###################################################
# 4. ФУНКЦИЯ ЗАПИСИ АУДИО (general)
###################################################
def record_audio(duration=1, sr=16000, channels=1):
    """
    Записывает duration секунд с микрофона, сохраняет в temp*.wav, возвращает имя файла.
    """
    p = pyaudio.PyAudio()
    chunk = 1024
    fmt = pyaudio.paInt16

    stream = p.open(format=fmt,
                    channels=channels,
                    rate=sr,
                    input=True,
                    frames_per_buffer=chunk)
    frames = []
    start = time.time()
    while True:
        data = stream.read(chunk)
        frames.append(data)
        if (time.time() - start) >= duration:
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    fname = f"temp_{int(time.time())}.wav"
    wf = wave.open(fname, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(fmt))
    wf.setframerate(sr)
    wf.writeframes(b''.join(frames))
    wf.close()
    return fname

###################################################
# 5. ОСНОВНОЙ КЛАСС ДЕМО
###################################################
class UnifiedDemoDeepFace:
    def __init__(self, players=2):
        self.players = players

        # Модули
        self.gesture_module = GestureInterpreter()
        self.voice_id_module = VoiceIdentificationModule(sr=16000)
        self.face_module = DeepFaceRecognitionModule(db_path="faces_db", detector_backend="retinaface")

        # SpeechRecognition
        self.sr_recognizer = sr.Recognizer()

        # Поток для записи коротких аудио-сегментов
        self.audio_queue = queue.Queue()
        self.stop_audio_thread = False
        self.audio_thread = threading.Thread(target=self._audio_capture_loop, daemon=True)

    def enroll_all(self):
        """
        1) Энроллим лицо (сохраняем снимок в faces_db/<player>.jpg)
        2) Энроллим голос (сохраняем MFCC)
        """
        for i in range(1, self.players + 1):
            pid = f"Игрок_{i}"
            print(f"=== Энролл лица для {pid} ===")
            self.face_module.enroll_face(pid)

            print(f"=== Энролл голоса для {pid} ===")
            self.voice_id_module.enroll_voice(pid, duration=3)

    def _audio_capture_loop(self):
        """
        Непрерывно записывает маленькие аудиофрагменты (1 сек) и кладёт их в очередь.
        """
        while not self.stop_audio_thread:
            wav_file = record_audio(duration=1, sr=16000, channels=1)
            self.audio_queue.put(wav_file)

    def _process_audio_queue(self):
        """
        Обрабатывает накопленные в очереди аудиофрагменты:
          - Распознаёт речь Google Speech
          - Определяет говорившего (VoiceIdentificationModule)
          - Выводит результат
        """
        try:
            while True:
                wav_file = self.audio_queue.get_nowait()
                text = ""
                with sr.AudioFile(wav_file) as source:
                    audio_data = self.sr_recognizer.record(source)
                try:
                    text = self.sr_recognizer.recognize_google(audio_data, language="ru-RU")
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    print(f"[Speech] Ошибка сервиса: {e}")

                if text.strip():
                    speaker = self.voice_id_module.identify_voice(wav_file)
                    if speaker is None:
                        speaker = "Неизвестный_голос"
                    print(f"{speaker} (голос) произнёс: {text}")

        except queue.Empty:
            pass

    def run(self):
        """
        1) Запускаем поток аудиозаписи
        2) Захватываем видео, распознаём лица (DeepFace), жесты, выводим в консоль при изменении.
        3) Одновременно распознаём речь в (почти) реальном времени.
        """
        self.audio_thread.start()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[MAIN] Не удалось открыть камеру.")
            return

        print("[MAIN] Нажмите ESC для выхода.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1) Распознаём лица DeepFace
            face_results = self.face_module.recognize_faces(frame)
            # face_results: [(top, right, bottom, left, pid), ...]

            # Рисуем bounding box + имя
            for (top, right, bottom, left, pid) in face_results:
                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
                label = pid if pid else "Неизвестный"
                cv2.putText(frame, label, (left, top-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # 2) Распознаём жесты
            gestures_info = self.gesture_module.interpret_gestures(frame)
            for hand_idx, gesture_name in gestures_info:
                if hand_idx < 0:
                    # "Нет рук"
                    pass
                else:
                    # Упрощённо считаем, что "рука i" ~ "лицо i",
                    # хотя это не всегда верно, нужно было бы сопоставлять по координатам
                    if 0 <= hand_idx < len(face_results):
                        pid = face_results[hand_idx][4]
                        pid_str = pid if pid else f"? (неизвестное лицо)"
                    else:
                        pid_str = f"Игрок_{hand_idx} (без лица)"
                    print(f"{pid_str} – жест: {gesture_name}")

            # 3) Обрабатываем очередь аудио
            self._process_audio_queue()

            # 4) Показать кадр
            cv2.imshow("DeepFace Demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

        self.stop_audio_thread = True
        cap.release()
        cv2.destroyAllWindows()
        self.audio_thread.join()


###################################################
# 6. ЗАПУСК
###################################################
if __name__ == "__main__":
    demo = UnifiedDemoDeepFace(players=2)
    demo.enroll_all()  # Сначала записываем лица и голоса
    demo.run()         # Основной цикл
