

- dispatcher.py can be on an other computer given you edited the config files and changed the default 127.0.0.1 to your local network IP adress where the script resides (and that you installed t2yLLM on that second computer also).

- if you have low VRAM, scripts should be launched in this order to avoid OOM with vLLM:
  - 1 - llm_example.py
  - 2 - dispatcher_example.py
