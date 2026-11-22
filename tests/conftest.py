"""Outils partagés par les tests."""

from __future__ import annotations

import pytest

from telemetry.packet_parser import HEADER_STRUCT, PACKET_SIZES, PacketId


def build_packet(
    packet_id: int,
    packet_format: int = 2025,
    size: int | None = None,
    player_car_index: int = 0,
    session_time: float = 12.5,
) -> bytes:
    """Fabrique un faux paquet : en-tête valide + octets nuls jusqu'à la taille voulue.

    Par défaut, la taille est celle de la spec pour ce type de paquet.
    """
    header = HEADER_STRUCT.pack(
        packet_format, 25, 1, 5, 1, packet_id,
        0x1234_5678_9ABC_DEF0, session_time, 100, 100, player_car_index, 255,
    )
    if size is None:
        size = PACKET_SIZES.get(PacketId(packet_id), len(header)) if packet_id in PacketId._value2member_map_ else len(header)
    return header + bytes(max(0, size - len(header)))


@pytest.fixture
def make_packet():
    """Donne accès à `build_packet` dans les tests."""
    return build_packet
