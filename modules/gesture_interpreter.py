import cv2
import mediapipe as mp

class GestureInterpreter:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        """
        :param mode: Если True, MediaPipe работает в статическом режиме (для фото).
        :param max_hands: Макс. число рук в кадре.
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

    def interpret_gesture(self, frame):
        """
        Принимает кадр BGR (numpy-массив OpenCV).
        Возвращает простое текстовое описание обнаруженной руки/жеста
        и/или рисует ключевые точки на кадре для наглядности.
        """
        # Конвертируем цвет, т.к. mediapipe работает с RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        gesture_info = "Нет рук"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Рисуем точки и соединения на исходном кадре
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
            # Можно развить логику: анализировать расположение пальцев
            # Пока выводим лишь число найденных рук
            gesture_info = f"Рук в кадре: {len(results.multi_hand_landmarks)}"

        return gesture_info