"""Tests du récepteur UDP (envoi réel de datagrammes en local, sur un port libre)."""

import asyncio
import socket

from telemetry.packet_parser import PacketId, validate_packet
from telemetry.udp_receiver import UdpReceiver
from tools.udp_probe import ProbeStats


def _send(port: int, *payloads: bytes) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for payload in payloads:
            sock.sendto(payload, ("127.0.0.1", port))


def test_receives_packets_in_order(make_packet) -> None:
    async def scenario() -> list[bytes]:
        receiver = UdpReceiver("127.0.0.1", 0)
        await receiver.start()
        try:
            _send(receiver.port, make_packet(PacketId.SESSION), make_packet(PacketId.LAP_DATA))
            return [await asyncio.wait_for(receiver.receive(), timeout=2) for _ in range(2)]
        finally:
            receiver.close()

    first, second = asyncio.run(scenario())
    assert validate_packet(first)[1] is PacketId.SESSION
    assert validate_packet(second)[1] is PacketId.LAP_DATA


def test_drops_oldest_when_queue_full() -> None:
    async def scenario() -> tuple[UdpReceiver, list[bytes]]:
        receiver = UdpReceiver("127.0.0.1", 0, queue_size=2)
        for i in range(5):
            receiver._push(bytes([i]))
        return receiver, [await receiver.receive() for _ in range(2)]

    receiver, kept = asyncio.run(scenario())
    assert receiver.received == 5
    assert receiver.dropped == 3
    assert kept == [bytes([3]), bytes([4])]


def test_probe_stats_counts_and_rejects(make_packet) -> None:
    stats = ProbeStats()
    stats.add(make_packet(PacketId.CAR_TELEMETRY))
    stats.add(make_packet(PacketId.CAR_TELEMETRY))
    stats.add(make_packet(PacketId.LAP_DATA, packet_format=2026))
    stats.add(bytes(5))
    assert stats.counts[(2025, PacketId.CAR_TELEMETRY)] == 2
    assert stats.valid == 2
    assert sum(stats.rejections.values()) == 2
    assert "CAR_TELEMETRY" in stats.report(elapsed=1.0)


def test_probe_rate_uses_active_window(make_packet) -> None:
    """20 paquets à 20 Hz reçus pendant une écoute de 120 s : la fréquence doit valoir 20/s, pas 20/120."""
    stats = ProbeStats()
    for i in range(20):
        stats.add(make_packet(PacketId.CAR_TELEMETRY), now=100.0 + i * 0.05)
    line = next(l for l in stats.report(elapsed=120.0).splitlines() if "CAR_TELEMETRY" in l)
    assert " 20.0 " in line
