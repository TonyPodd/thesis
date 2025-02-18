# modules/gesture_interpreter.py

import cv2
import mediapipe as mp
import math

class GestureInterpreter:
    def __init__(self, mode=False, max_hands=4, detection_confidence=0.5, tracking_confidence=0.5):
        """
        :param mode: Если True, MediaPipe работает в статическом режиме (для фото).
        :param max_hands: Максимальное число рук в кадре.
        :param detection_confidence: порог для детекции руки.
        :param tracking_confidence: порог для трекинга ключевых точек.
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        # Словарь для отслеживания состояния "готов к выстрелу" для каждой руки по её индексу.
        self.shoot_in_progress = {}
        # Запомним предыдущий жест каждой руки, чтобы "отлавливать" появление нового
        self.prev_gesture = {}

    def is_hand_open(self, hand_landmarks):
        """
        Проверяет «прямая ладошка» (4 пальца, кроме большого, направлены вверх).
        """
        landmarks = hand_landmarks.landmark
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        open_fingers = 0
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip].y < landmarks[pip].y:
                open_fingers += 1
        return open_fingers == 4

    def is_ok_gesture(self, hand_landmarks):
        """
        Проверяет, что кончики большого (4) и указательного (8) пальцев близко,
        а указательный при этом «согнут», средний, безымянный и мизинец выпрямлены.
        """
        landmarks = hand_landmarks.landmark
        dx = landmarks[4].x - landmarks[8].x
        dy = landmarks[4].y - landmarks[8].y
        distance = math.hypot(dx, dy)

        # "Размер" руки – расстояние между запястьем (0) и кончиком среднего пальца (12)
        hand_size = math.hypot(landmarks[0].x - landmarks[12].x, landmarks[0].y - landmarks[12].y)
        ratio = distance / hand_size if hand_size > 0 else 1

        bent_index = landmarks[8].y > landmarks[6].y
        extended_middle = landmarks[12].y < landmarks[10].y
        extended_ring   = landmarks[16].y < landmarks[14].y
        extended_pinky  = landmarks[20].y < landmarks[18].y

        return (ratio < 0.4) and bent_index and extended_middle and extended_ring and extended_pinky

    def is_shoot_gesture(self, hand_landmarks):
        """
        Готов к «выстрелу»: указательный и средний выпрямлены, безымянный и мизинец согнуты.
        """
        landmarks = hand_landmarks.landmark
        index_extended = landmarks[8].y < landmarks[6].y
        middle_extended = landmarks[12].y < landmarks[10].y
        ring_folded = landmarks[16].y > landmarks[14].y
        pinky_folded = landmarks[20].y > landmarks[18].y
        return index_extended and middle_extended and ring_folded and pinky_folded

    def is_shoot_release(self, hand_landmarks):
        """
        Отпускание выстрела – указательный и средний, ранее выпрямленные, теперь согнуты.
        """
        landmarks = hand_landmarks.landmark
        index_folded = landmarks[8].y > landmarks[6].y
        middle_folded = landmarks[12].y > landmarks[10].y
        return index_folded and middle_folded

    def is_thumbs_up(self, hand_landmarks):
        """
        Простейшая проверка «палец вверх»: большой палец выше запястья,
        остальные пальцы (указательный, средний, безымянный, мизинец) согнуты.
        """
        landmarks = hand_landmarks.landmark
        # Запястье – точка 0
        wrist_y = landmarks[0].y
        # Кончик большого пальца – точка 4
        thumb_tip_y = landmarks[4].y

        # Проверяем, что большой палец «выше» (меньше y) запястья
        thumb_up = thumb_tip_y < wrist_y

        # Остальные пальцы считаем согнутыми, если tip.y > pip.y
        # Указательный (tip=8, pip=6), Средний (12,10), Безымянный (16,14), Мизинец (20,18)
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        folded_fingers = 0
        for tip, pip in zip(tips, pips):
            if landmarks[tip].y > landmarks[pip].y:
                folded_fingers += 1

        return thumb_up and (folded_fingers == 4)

    def is_thumbs_down(self, hand_landmarks):
        """
        Простейшая проверка «палец вниз»: большой палец ниже запястья,
        остальные пальцы согнуты (аналогично thumbs_up, но big thumb_tip_y > wrist_y).
        """
        landmarks = hand_landmarks.landmark
        wrist_y = landmarks[0].y
        thumb_tip_y = landmarks[4].y

        # Большой палец «ниже» (больше y) запястья
        thumb_down = thumb_tip_y > wrist_y

        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        folded_fingers = 0
        for tip, pip in zip(tips, pips):
            if landmarks[tip].y > landmarks[pip].y:
                folded_fingers += 1

        return thumb_down and (folded_fingers == 4)

    def interpret_gesture(self, frame):
        """
        Для каждой обнаруженной руки определяет жест из списка:
         - Жест OK
         - Готов к выстрелу / Выстрел выполнен!
         - Прямая ладошка
         - Палец вверх (мирный/да)
         - Палец вниз (мафия/нет)
         - Неопределённый жест

        Возвращает список (idx -> строка жеста).
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        persons_gestures = []
        if results.multi_hand_landmarks:
            # Синхронизируем shoot_in_progress с актуальным количеством рук
            active_indices = set(range(len(results.multi_hand_landmarks)))
            for key in list(self.shoot_in_progress.keys()):
                if key not in active_indices:
                    del self.shoot_in_progress[key]

            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Рисуем ключевые точки
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                gesture = ""

                # 1) Сначала проверяем новые жесты: thumbs_up / thumbs_down
                if self.is_thumbs_up(hand_landmarks):
                    gesture = "Палец вверх"
                    self.shoot_in_progress[idx] = False
                elif self.is_thumbs_down(hand_landmarks):
                    gesture = "Палец вниз"
                    self.shoot_in_progress[idx] = False
                # 2) Далее OK
                elif self.is_ok_gesture(hand_landmarks):
                    gesture = "Жест OK"
                    self.shoot_in_progress[idx] = False
                else:
                    # Инициализируем состояние
                    if idx not in self.shoot_in_progress:
                        self.shoot_in_progress[idx] = False

                    # Логика с «выстрелом»
                    if self.shoot_in_progress[idx]:
                        if self.is_shoot_release(hand_landmarks):
                            gesture = "Выстрел выполнен!"
                            self.shoot_in_progress[idx] = False
                        else:
                            gesture = "Готов к выстрелу"
                    else:
                        if self.is_shoot_gesture(hand_landmarks):
                            self.shoot_in_progress[idx] = True
                            gesture = "Готов к выстрелу"
                        elif self.is_hand_open(hand_landmarks):
                            gesture = "Прямая ладошка"
                        else:
                            gesture = "Неопределённый жест"

                persons_gestures.append((idx, gesture))
        else:
            # Если нет рук, вернём запись об этом
            persons_gestures.append((-1, "Нет рук"))
            self.shoot_in_progress.clear()

        return persons_gestures
