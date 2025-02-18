# test_unified_demo.py

import cv2
import mediapipe as mp
import time
import sys

from modules.gesture_interpreter import GestureInterpreter
from modules.speech_recognition_module import SpeechRecognitionModule
from modules.voice_identification import VoiceIdentificationModule


class TestUnifiedDemo:
    def __init__(self, num_players=2):
        self.num_players = num_players

        # Модули:
        self.gesture_module = GestureInterpreter(mode=False, max_hands=4)
        self.sr_module = SpeechRecognitionModule(language="ru-RU")
        self.voice_id_module = VoiceIdentificationModule(sr=16000, record_seconds=3)

        # Детектор лиц Mediapipe для примера
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)

        # Запомним текущие жесты по рукам (чтобы определить, новый он или нет)
        self.last_gestures = {}

    def enroll_players(self):
        """
        Последовательно записывает голос каждого игрока и сохраняет в known_voices.
        Здесь можно было бы добавить логику "запиши лицо игрока i", если бы
        делали полноценное распознавание лиц.
        """
        print("=== Этап записи голосов (энролл) ===")
        for i in range(1, self.num_players + 1):
            print(f"--- Игрок {i}: подготовьтесь к записи голоса ---")
            self.voice_id_module.enroll_voice(player_id=f"Игрок {i}")

    def run(self):
        print("Запуск основного цикла видео. Нажмите ESC для выхода.")
        print("Нажмите 's' чтобы записать речь и определить, кто говорит.")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Не удалось открыть веб-камеру")
            sys.exit(1)

        with self.mp_face_detection.FaceDetection(
                min_detection_confidence=0.5) as face_det:

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 1. Поиск лиц
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_det.process(frame_rgb)
                if results.detections:
                    for i, detection in enumerate(results.detections):
                        # Просто рисуем рамку и пишем "Игрок i" (упрощённо)
                        self.mp_drawing.draw_detection(frame, detection)
                        # Извлекаем bounding box, чтобы подписать текст
                        box = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        x1, y1 = int(box.xmin * w), int(box.ymin * h)

                        cv2.putText(frame, f"Игрок {i+1}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Нет лиц", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

                # 2. Распознаём жесты
                gestures_info = self.gesture_module.interpret_gesture(frame)
                for (hand_idx, gesture) in gestures_info:
                    if hand_idx < 0:
                        # "Нет рук"
                        cv2.putText(frame, "Нет рук", (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                    else:
                        # Отслеживаем "новизну" жеста
                        old_g = self.last_gestures.get(hand_idx, "")
                        if gesture != old_g:
                            # Предположим, что hand_idx ~ номер игрока, 
                            # или надо найти сопоставление с лицом...
                            # В реальном проекте нужно решать, какая рука у какого игрока.
                            print(f"Игрок ? (рука {hand_idx}) - жест: {gesture}")

                            # Запомним
                            self.last_gestures[hand_idx] = gesture

                # 3. Показываем кадр
                cv2.imshow("Test Unified Demo", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('s'):
                    # Запускаем распознавание речи (блокирующий вызов)
                    print("=== Начало записи речи. Говорите... ===")
                    text = self.sr_module.recognize_from_microphone()
                    print("Речь распознана, определяем говорившего...")

                    # Идентифицируем говорившего (по голосу)
                    speaker = self.voice_id_module.identify_voice()
                    if speaker is not None:
                        print(f"{speaker} - речь: {text}")
                    else:
                        print(f"Не удалось идентифицировать говорившего. Текст: {text}")

            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    demo = TestUnifiedDemo(num_players=2)
    demo.enroll_players()    # Сначала записываем голос каждого игрока
    demo.run()               # Запускаем основной цикл
