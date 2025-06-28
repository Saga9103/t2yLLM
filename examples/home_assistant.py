from t2yLLM.llm_backend_async import AssistantEngine, WebUI, logger
from t2yLLM.localDispatcher import VoiceEngine
import uvicorn
import threading
import webbrowser
import time


def main():
    engine = AssistantEngine()
    engine.start()
    while not engine.is_running():
        time.sleep(0.5)
    time.sleep(2.0)
    dispatcher = None
    logger.info("t2yLLM started in \033[91mlocal\033[0m mode")
    dispatcher = VoiceEngine()
    dispatcher.start()
    logger.info("Voice Assistant started, ready to speak")

    webui = WebUI(engine)

    try:
        while engine.is_running():
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() == "exit":
                    break
                elif user_input.lower() == "status":
                    print(f"Engine: {engine.status()}")
                    if dispatcher:
                        print(f"Dispatcher: {dispatcher.get_status()}")
                elif user_input.lower() == "test" and dispatcher:
                    dispatcher.server.send_test_command()
                else:
                    engine(user_input)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")

    finally:
        logger.info("Shutting down...")
        if dispatcher:
            dispatcher.stop()

        if engine.is_running():
            logger.info("Stopping AssistantEngine")
            engine.stop()

        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
