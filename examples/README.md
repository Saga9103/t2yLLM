- Copy one of the examples in the directory of your choice and execute the script :
  
    - `home_assistant.py` is for **local** mode tested with Jabra Speak2
      
    - Both `dispatcher_example.py` and `llm_example.py` are **dsitributed mode** meant to be run from 2 differents terminals and without the web UI.
    - `llm_example_webui.py` is the version with the webui implementation.
    - use `from dispatcher import VoiceEngine` for the dispatcher with whisper backend and voice analysis
    - use `from llm_backend_async import AssistantEngine, logger` for the LLM backend
