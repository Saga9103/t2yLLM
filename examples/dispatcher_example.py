from t2yLLM.Chat.dispatcher import VoiceEngine


def main():
    engine = VoiceEngine()
    engine.start()
    try:
        while engine.running:
            cmd = input("\nCommand ('test', 'status', 'exit'): ").strip()
            if cmd.lower() == "test":
                engine.server.send_test_command()
            elif cmd.lower() == "status":
                status = engine.get_status()
                print(f"Running: {status['running']}")
                print(f"Recording: {status['recording']}")
                print(f"Command queue: {status['command_queue']}")
                print(f"Response queue: {status['response_queue']}")
                print(f"Whisper queue: {status['whisper_queue']}")
            elif cmd.lower() in ("exit", "quit", "stop"):
                break
    except KeyboardInterrupt:
        print("\nCtrl+C was pressed, exiting")
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
