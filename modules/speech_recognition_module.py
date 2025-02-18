import speech_recognition as sr

class SpeechRecognitionModule:
    def __init__(self, language="ru-RU"):
        """
        :param language: язык распознавания (по умолчанию русский).
        """
        self.language = language
        self.recognizer = sr.Recognizer()
        # При необходимости можно настроить порог чувствительности и т. д.
        # self.recognizer.energy_threshold = 300

    def recognize_from_microphone(self):
        """
        Записывает звук с микрофона и пытается распознать его в текст.
        Возвращает распознанную строку или пустую строку, если не удалось.
        """
        with sr.Microphone() as source:
            print("Скажите что-нибудь...")
            audio_data = self.recognizer.listen(source)

        try:
            text = self.recognizer.recognize_google(audio_data, language=self.language)
            print(f"Распознанный текст: {text}")
            return text
        except sr.UnknownValueError:
            print("Не удалось распознать речь")
            return ""
        except sr.RequestError as e:
            print(f"Ошибка сервиса распознавания: {e}")
            return ""