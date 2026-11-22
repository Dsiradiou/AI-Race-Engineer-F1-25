"""Réception des paquets UDP envoyés par F1 25.

Le récepteur ne décode rien : il met les datagrammes bruts dans une file `asyncio`.
Si le consommateur prend du retard, les paquets les plus anciens sont jetés :
en temps réel, une donnée fraîche vaut mieux qu'une donnée en retard.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


class _Protocol(asyncio.DatagramProtocol):
    """Pont entre la boucle asyncio et le récepteur."""

    def __init__(self, receiver: UdpReceiver) -> None:
        self._receiver = receiver

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        self._receiver._push(data)

    def error_received(self, exc: Exception) -> None:
        logger.warning("Erreur UDP : %s", exc)


class UdpReceiver:
    """Écoute un port UDP et fournit les paquets bruts reçus.

    Usage :
        receiver = UdpReceiver("127.0.0.1", 20777)
        await receiver.start()
        data = await receiver.receive()
        receiver.close()
    """

    def __init__(self, host: str, port: int, queue_size: int = 1000) -> None:
        self.host = host
        self.port = port
        self.received = 0
        """Nombre total de paquets reçus."""
        self.dropped = 0
        """Nombre de paquets jetés parce que la file était pleine."""
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=queue_size)
        self._transport: asyncio.DatagramTransport | None = None

    async def start(self) -> None:
        """Ouvre le socket UDP. Avec `port=0`, le système choisit un port libre (utile pour les tests)."""
        loop = asyncio.get_running_loop()
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: _Protocol(self), local_addr=(self.host, self.port)
        )
        self.host, self.port = self._transport.get_extra_info("sockname")[:2]
        logger.info("Écoute UDP sur %s:%d", self.host, self.port)

    async def receive(self) -> bytes:
        """Attend et renvoie le prochain paquet brut."""
        return await self._queue.get()

    def close(self) -> None:
        """Ferme le socket."""
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    def _push(self, data: bytes) -> None:
        self.received += 1
        if self._queue.full():
            self._queue.get_nowait()
            self.dropped += 1
        self._queue.put_nowait(data)
