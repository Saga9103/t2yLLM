from t2yLLM.llm_backend_async import AssistantEngine, WebUI, logger
import uvicorn
import threading
import webbrowser
import time


def main():
    engine = AssistantEngine()
    engine.start()
    webui = WebUI(engine)
    config = uvicorn.Config(webui.app, host="127.0.0.1", port=8765, log_level="info")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:8765")

    while engine.is_running():
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == "exit":
                engine.stop()
                break
            elif user_input.lower() == "status":
                print(engine.status())
            else:
                engine(user_input)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error in AssistantEngine : {e}")

    if engine.is_running():
        logger.info("Stopping AssistantEngine")
        engine.stop()


if __name__ == "__main__":
    main()

