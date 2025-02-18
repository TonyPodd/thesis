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

    def is_hand_open(self, hand_landmarks):
        """
        Определяет, является ли рука «прямой ладошкой» (открытой),
        анализируя положение 4 пальцев (без большого).
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
        Определяет, является ли жест "ОК".
        Проверяем, что кончики большого (4) и указательного (8) пальцев находятся близко,
        а остальные пальцы вытянуты.
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

        if ratio < 0.4 and bent_index and extended_middle and extended_ring and extended_pinky:
            return True
        return False

    def is_shoot_gesture(self, hand_landmarks):
        """
        Определяет, находится ли рука в состоянии "готова к выстрелу":
        указательный и средний пальцы выпрямлены, а безымянный и мизинец согнуты.
        """
        landmarks = hand_landmarks.landmark

        index_extended = landmarks[8].y < landmarks[6].y
        middle_extended = landmarks[12].y < landmarks[10].y
        ring_folded = landmarks[16].y > landmarks[14].y
        pinky_folded = landmarks[20].y > landmarks[18].y

        if index_extended and middle_extended and ring_folded and pinky_folded:
            return True
        return False

    def is_shoot_release(self, hand_landmarks):
        """
        Определяет, что произошло "отпускание" выстрела – указательный и средний пальцы, ранее выпрямленные,
        теперь сгибаются (кончики опускаются ниже своих PIP-соединений).
        """
        landmarks = hand_landmarks.landmark

        index_folded = landmarks[8].y > landmarks[6].y
        middle_folded = landmarks[12].y > landmarks[10].y

        if index_folded and middle_folded:
            return True
        return False

    def interpret_gesture(self, frame):
        """
        Принимает кадр BGR (numpy-массив OpenCV).
        Для каждой обнаруженной руки (предположим, что каждая рука принадлежит отдельному человеку)
        определяет жест:
          - "Жест OK" – если обнаружен жест "ОК"
          - "Готов к выстрелу" – если рука находится в состоянии выстрела (ожидание отпускания)
          - "Выстрел выполнен!" – если произошло отпускание ранее выпрямленных пальцев
          - "Прямая ладошка" – если рука открыта
          - "Неопределённый жест" – если ни одно из условий не выполнено.
        Рисует ключевые точки на кадре.
        Возвращает строку с распознанными жестами для каждого человека.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        persons_gestures = []
        if results.multi_hand_landmarks:
            # Если количество рук меньше, чем в предыдущем кадре, удаляем неактуальные записи
            active_indices = set(range(len(results.multi_hand_landmarks)))
            for key in list(self.shoot_in_progress.keys()):
                if key not in active_indices:
                    del self.shoot_in_progress[key]

            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Рисуем ключевые точки на изображении
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                gesture = ""
                # Если обнаружен жест "ОК", сбрасываем состояние выстрела для этого человека
                if self.is_ok_gesture(hand_landmarks):
                    gesture = "Жест OK"
                    print(f"Человек {idx+1}: Жест OK обнаружен!")
                    self.shoot_in_progress[idx] = False
                else:
                    # Инициализируем состояние, если ранее его не было
                    if idx not in self.shoot_in_progress:
                        self.shoot_in_progress[idx] = False

                    # Если ранее был зафиксирован жест "готов к выстрелу"
                    if self.shoot_in_progress[idx]:
                        if self.is_shoot_release(hand_landmarks):
                            gesture = "Выстрел выполнен!"
                            print(f"Человек {idx+1}: Выстрел выполнен!")
                            self.shoot_in_progress[idx] = False
                        else:
                            gesture = "Готов к выстрелу"
                    else:
                        # Если сейчас рука в положении "готова к выстрелу", зафиксируем это
                        if self.is_shoot_gesture(hand_landmarks):
                            self.shoot_in_progress[idx] = True
                            gesture = "Готов к выстрелу"
                        elif self.is_hand_open(hand_landmarks):
                            gesture = "Прямая ладошка"
                        else:
                            gesture = "Неопределённый жест"
                persons_gestures.append(f"Человек {idx+1}: {gesture}")
        else:
            persons_gestures.append("Нет рук")
            self.shoot_in_progress.clear()

        # Можно разделить вывод по строкам или вывести в одной строке
        return "\n".join(persons_gestures)
