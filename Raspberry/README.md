*DEPRECATED*

- The **rpi_server.py** script **should be copied to your raspberry Pi** in a directory of your choice. 
- The **config file and script** should be located in the **same directory** for it to work.
- You can either launch it at start with a custom script / service in a venv or use crontab to launch the venv and the script at boot time :<br>
  `@reboot ~/my_custom_dir/my_venv/bin/python ~/my_custom_dir/rpi_server.py >> ~/my_log_file.log 2>&1`
