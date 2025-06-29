import subprocess
import socket
import time
import logging
from pathlib import Path
import shutil
import psutil

logger = logging.getLogger("CaddyManager")


class CaddyManager:
    """Manages Caddy reverse proxy for local network"""

    def __init__(
        self, cert_dir: Path, network_domain: str = None, domain: str = "t2yllm.local"
    ):
        self.cert_dir = cert_dir
        self.domain = domain
        self.network_domain = network_domain
        self.caddy_process = None
        self.config_path = None
        self.pid_file = Path("/tmp/t2yllm_caddy.pid")

    def check_caddy_installed(self) -> bool:
        """Check if Caddy is installed on the system"""
        return shutil.which("caddy") is not None

    def get_local_ip(self) -> str:
        """Get the local IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            return "127.0.0.1"

    def generate_caddyfile(self) -> Path:
        """Generate Caddyfile configuration"""
        config_dir = Path("/tmp/t2yllm_caddy")
        config_dir.mkdir(exist_ok=True)

        subnet = f"{self.network_domain}/24" if self.network_domain else "127.0.0.1"

        caddyfile_path = config_dir / "Caddyfile"

        caddyfile_content = f"""
{{
    admin off
    auto_https disable_redirects
}}

https://:8766 {{
    tls {self.cert_dir / f"{self.domain}.crt.pem"} {self.cert_dir / f"{self.domain}.key.pem"}

    @allowed {{
        remote_ip {subnet}
        remote_ip 127.0.0.1
        remote_ip ::1
    }}
    
    handle @allowed {{
        header {{
            X-Content-Type-Options "nosniff"
            X-Frame-Options "DENY"
            X-XSS-Protection "1; mode=block"
            Referrer-Policy "strict-origin-when-cross-origin"
            -Server
        }}
        
        reverse_proxy https://127.0.0.1:8765 {{
            header_up Host {{http.request.host}}
            header_up X-Real-IP {{remote_host}}
            header_up X-Forwarded-For {{remote_host}}
            header_up X-Forwarded-Proto {{scheme}}
            
            transport http {{
                tls
                tls_insecure_skip_verify
            }}
        }}
    }}
    
    handle {{
        respond "Access denied" 403
    }}
}}
"""

        caddyfile_path.write_text(caddyfile_content)
        self.config_path = caddyfile_path

        return caddyfile_path

    def start(self) -> bool:
        """Start Caddy server"""
        if not self.check_caddy_installed():
            logger.error("\033[91mWARNING: Caddy is not installed!\033[0m")
            logger.error("\033[91mPlease install Caddy: sudo apt install caddy\033[0m")
            return False

        if self.is_running():
            logger.warning("Caddy is already running")
            return True

        try:
            config_path = self.generate_caddyfile()
            validate_cmd = ["caddy", "validate", "--config", str(config_path)]
            result = subprocess.run(validate_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"Caddy configuration validation failed: {result.stderr}")
                return False

            cmd = ["caddy", "run", "--config", str(config_path)]

            self.caddy_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )

            time.sleep(2)

            if self.caddy_process.poll() is not None:
                stdout, stderr = self.caddy_process.communicate()
                logger.error(f"Caddy failed to start: {stderr.decode()}")
                return False

            self.pid_file.write_text(str(self.caddy_process.pid))

            local_ip = self.get_local_ip()
            logger.info("\033[92mCaddy started successfully!\033[0m")
            logger.info(
                f"\033[92mLocal network access available at: https://{local_ip}:8766\033[0m"
            )
            logger.info(
                f"\033[92mFiltering on : {self.network_domain}/24\033[0m"
                if self.network_domain
                else ""
            )

            return True

        except Exception as e:
            logger.error(f"Failed to start Caddy: {e}")
            return False

    def stop(self):
        try:
            if self.pid_file.exists():
                pid = int(self.pid_file.read_text())
                try:
                    process = psutil.Process(pid)
                    if process.name() == "caddy":
                        process.terminate()
                        process.wait(timeout=5)
                        logger.info("Caddy was stopped")
                except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                    pass
                finally:
                    self.pid_file.unlink(missing_ok=True)

            if self.caddy_process:
                self.caddy_process.terminate()
                try:
                    self.caddy_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.caddy_process.kill()
                    self.caddy_process.wait()
                logger.info("Caddy subprocess stopped")

            if self.config_path and self.config_path.exists():
                self.config_path.unlink()

        except Exception as e:
            logger.error(f"Error stopping Caddy: {e}")

    def is_running(self) -> bool:
        """checks caddy"""
        if self.pid_file.exists():
            try:
                pid = int(self.pid_file.read_text())
                process = psutil.Process(pid)
                return process.name() == "caddy" and process.is_running()
            except (ValueError, psutil.NoSuchProcess):
                self.pid_file.unlink(missing_ok=True)
                return False
        return False

    def __del__(self):
        self.stop()
