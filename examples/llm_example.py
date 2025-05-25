from t2yLLM.llm_backend_async import AssistantEngine, logger
# from llm_backend_async import AssistantEngine, logger


def main():
    engine = AssistantEngine()
    engine.start()

    while engine.is_running():
        try:
            user_input = input("\nYou: ").strip()
            engine(user_input)
            if user_input.lower() == "exit":
                engine.stop()
            elif user_input.lower() == "status":
                print(engine.status())
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error in AssistantEngine : {e}")

    if engine.is_running():
        logger.info("Stopping AssistantEngine")
        engine.stop()


if __name__ == "__main__":
    main()
