# CADDY configuration

WebUI can be accessed on your local network e.g 192.168.X.0 (see server_config.yaml example) if you installed caddy<br>

`sudo apt install caddy`

you have to access it at **https://t2yllm.local:8766**, in order for that to work, on your server do :<br>

`sudo ufw deny from any to any port 8766 proto tcp`
`sudo ufw allow from 192.168.X.0/24 to any port 8766 proto tcp`
`sudo ufw reload`

you can check with :<br>
`sudo ufw status numbered`

check if *local_net_broadcast* is set to *True* in server_config.yaml
