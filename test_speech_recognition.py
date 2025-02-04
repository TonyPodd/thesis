from modules.speech_recognition_module import SpeechRecognitionModule

if __name__ == "__main__":
    sr_module = SpeechRecognitionModule(language="ru-RU")
    text = sr_module.recognize_from_microphone()
    print("Итоговое распознанное сообщение:", text)