# For yeelight, you first have to go through the app, create an account and enable LAN CONTROL for each light after you connected them to your local network.

# If your local network is let's say 192.168.6.0 and ufw is enabled then you have to allow multicast discovery then add these rules to ufw :

sudo ufw allow out to 239.255.255.250 port 1982 proto udp comment "Yeelight discovery out"
sudo ufw allow out to 192.168.6.0/24 port 55443 proto tcp comment "Yeelight control"
sudo ufw allow in from 192.168.6.0/24 to any port 1024:65535 proto udp comment "Yeelight callbacks"

sudo ufw reload

# Else discovery will be impossible
