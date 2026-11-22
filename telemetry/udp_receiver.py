"""Réception des paquets UDP envoyés par F1 25.

Le récepteur ne décode rien : il met les datagrammes bruts dans une file.
Si le consommateur prend du retard et que la file est pleine, le paquet le plus
ancien est jeté (en temps réel, une donnée fraîche vaut mieux qu'une donnée en retard),
**sauf** les paquets envoyés une seule fois (Event, FinalClassification) : perdus,
ils seraient perdus pour toujours (voir Q7).
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque

logger = logging.getLogger(__name__)

PACKET_ID_OFFSET = 6
"""Position de `m_packetId` dans l'en-tête (après format, année, versions)."""

PROTECTED_PACKET_IDS = frozenset({3, 8})
"""Event (3) et FinalClassification (8) : jamais jetés quand la file est pleine."""


def is_protected(data: bytes) -> bool:
    """Vrai si le paquet ne doit jamais être jeté (lecture d'un seul octet, sans décoder)."""
    return len(data) > PACKET_ID_OFFSET and data[PACKET_ID_OFFSET] in PROTECTED_PACKET_IDS


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
        self.queue_size = queue_size
        self.received = 0
        """Nombre total de paquets reçus."""
        self.dropped = 0
        """Nombre de paquets jetés parce que la file était pleine."""
        self._queue: deque[bytes] = deque()
        self._not_empty = asyncio.Event()
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
        while not self._queue:
            self._not_empty.clear()
            await self._not_empty.wait()
        return self._queue.popleft()

    def close(self) -> None:
        """Ferme le socket."""
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    def _push(self, data: bytes) -> None:
        self.received += 1
        if len(self._queue) >= self.queue_size:
            self._drop_oldest_unprotected()
        self._queue.append(data)
        self._not_empty.set()

    def _drop_oldest_unprotected(self) -> None:
        """Retire le plus ancien paquet non protégé (ou, à défaut, le plus ancien tout court)."""
        for i, queued in enumerate(self._queue):
            if not is_protected(queued):
                del self._queue[i]
                break
        else:
            self._queue.popleft()
        self.dropped += 1
