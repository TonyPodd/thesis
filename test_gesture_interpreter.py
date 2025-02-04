import cv2
from modules.gesture_interpreter import GestureInterpreter

if __name__ == "__main__":
    gest_module = GestureInterpreter()

    cap = cv2.VideoCapture(0)  # Открываем веб-камеру (индекс 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Распознаём жест
        gesture_text = gest_module.interpret_gesture(frame)

        # Добавим вывод текста на кадр
        cv2.putText(frame, gesture_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Gesture Interpreter Demo", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC для выхода
            break

    cap.release()
    cv2.destroyAllWindows()