# this a joke file i make lol, and this unuse

import socket
import urllib.request
from typing import List, Union
from returns.result import Result, Success, Failure


class Network:
    @staticmethod
    def get_hostname() -> Result[str, Exception]:
        try:
            return Success(socket.gethostname())
        except Exception as e:
            return Failure(e)

    @staticmethod
    def get_local_ip() -> Result[str, Exception]:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
            return Success(ip)
        except Exception as e:
            return Failure(e)

    @staticmethod
    def get_public_ip() -> Result[str, Exception]:
        try:
            with urllib.request.urlopen("https://api.ipify.org", timeout=5) as response:
                ip = response.read().decode().strip()
            return Success(ip)
        except Exception as e:
            return Failure(e)

    @staticmethod
    def get_all_interfaces() -> Result[List[Union[str, int]], Exception]:
        try:
            addresses = socket.getaddrinfo(socket.gethostname(), None)
            ips = sorted({addr[4][0] for addr in addresses})
            return Success(ips)
        except Exception as e:
            return Failure(e)
