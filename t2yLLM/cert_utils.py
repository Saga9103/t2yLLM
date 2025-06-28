from pathlib import Path
from datetime import datetime, timedelta
import ipaddress
import netifaces
import ssl
import os
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa

CERT_DIR = Path(__file__).resolve().parent / ".certs"


def _save(path: Path, data: bytes, mode: int = 0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    os.chmod(path, mode)


def _rsa():
    return rsa.generate_private_key(public_exponent=65537, key_size=4096)


def _collect_lan_ips() -> list[ipaddress.IPv4Address]:
    ips = []
    for iface in netifaces.interfaces():
        for family, _, _, addr in netifaces.ifaddresses(iface).get(
            netifaces.AF_INET, []
        ):
            ip = ipaddress.IPv4Address(addr["addr"])
            if not ip.is_loopback:
                ips.append(ip)
    return ips


def ensure_certs(domain: str = "t2yllm.local"):
    ca_key_f = CERT_DIR / "ca.key.pem"
    ca_crt_f = CERT_DIR / "ca.crt.pem"
    srv_key_f = CERT_DIR / f"{domain}.key.pem"
    srv_crt_f = CERT_DIR / f"{domain}.crt.pem"

    if not ca_key_f.exists():
        ca_key = _rsa()
        ca_cert = (
            x509.CertificateBuilder()
            .subject_name(
                x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "t2yLLM-Local-CA")])
            )
            .issuer_name(
                x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "t2yLLM-Local-CA")])
            )
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow() - timedelta(days=1))
            .not_valid_after(datetime.utcnow() + timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), True)
            .sign(ca_key, hashes.SHA256())
        )
        _save(
            ca_key_f,
            ca_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            ),
        )
        _save(ca_crt_f, ca_cert.public_bytes(serialization.Encoding.PEM))

    if not srv_key_f.exists():
        ca_key = serialization.load_pem_private_key(ca_key_f.read_bytes(), None)
        ca_cert = x509.load_pem_x509_certificate(ca_crt_f.read_bytes())
        srv_key = _rsa()
        lan_ips = _collect_lan_ips()
        san = [
            x509.DNSName(domain),
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        ] + [x509.IPAddress(ip) for ip in lan_ips]
        csr = (
            x509.CertificateSigningRequestBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, domain)]))
            .add_extension(x509.SubjectAlternativeName(san), False)
            .sign(srv_key, hashes.SHA256())
        )
        srv_cert = (
            x509.CertificateBuilder()
            .subject_name(csr.subject)
            .issuer_name(ca_cert.subject)
            .public_key(csr.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow() - timedelta(days=1))
            .not_valid_after(datetime.utcnow() + timedelta(days=825))
            .add_extension(
                csr.extensions.get_extension_for_class(
                    x509.SubjectAlternativeName
                ).value,
                False,
            )
            .sign(ca_key, hashes.SHA256())
        )
        _save(
            srv_key_f,
            srv_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            ),
        )
        _save(srv_crt_f, srv_cert.public_bytes(serialization.Encoding.PEM))

    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.load_cert_chain(certfile=srv_crt_f, keyfile=srv_key_f)

    return ctx, srv_key_f, srv_crt_f
