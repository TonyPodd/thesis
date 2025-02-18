from modules.voice_identification import VoiceIdentificationModule

if __name__ == "__main__":
    vid_module = VoiceIdentificationModule(sr=16000, record_seconds=3)

    # 1. "Обучаем" двух разных игроков
    print("=== Энролл игрока 1 ===")
    vid_module.enroll_voice(player_id="Игрок 1")

    print("=== Энролл игрока 2 ===")
    vid_module.enroll_voice(player_id="Игрок 2")

    # 2. Пытаемся идентифицировать (скажите, например, голосом ближе к одному из игроков)
    identified = vid_module.identify_voice()
    print(f"Определён как: {identified}")