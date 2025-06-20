# activate / deactivate plugins
  - from server_config.yaml in the config directory

# yeelight plugin
  - First run the configuration wizard :
    - check your firewall setup as it might block discovery as shown in `yeelight_ufw.md`
    - `python yeelight_manager.py setup`
    - follow the instructions, it will create a .yaml config file with your rooms and lights
    - Now you should be able to send simple commands via voice to the LLM like "turn on the bedroom lights"
