- llm_basckend_async.py, metacontext, pokemon.txt and pokemon_phonetics_fr.txt all need to be in the same directory.

- dispatcher.py can be in an other diretory or even on an other computer given you edited the config files and changed the default 127.0.0.1 to your local network IP adress where the script resides.

- if you have low VRAM, scripts should be launched in this order :
  - 1 - llm_backend_async.py
  - 2 - dispatcher.py

- the directory sctructure by default should be like :
  - Desktop 1 : ./Mars/config/<yaml config file and loader> and ./Mars/Chat/llm_backend_async.py
  - Desktop 1 (or Desktop 2) : ./Mars/config/<yaml config file and loader> and ./Mars/somedir/dispatcher.py
