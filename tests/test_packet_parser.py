"""Tests du décodage de l'en-tête et de la validation des paquets."""

import pytest

from telemetry.packet_parser import (
    HEADER_SIZE,
    PACKET_SIZES,
    PacketId,
    PacketValidationError,
    parse_header,
    validate_packet,
)


def test_header_size_matches_spec() -> None:
    assert HEADER_SIZE == 29


def test_parse_header_fields(make_packet) -> None:
    header = parse_header(make_packet(PacketId.LAP_DATA, player_car_index=7, session_time=42.0))
    assert header.packet_format == 2025
    assert header.game_year == 25
    assert header.packet_id == PacketId.LAP_DATA
    assert header.session_uid == 0x1234_5678_9ABC_DEF0
    assert header.session_time == pytest.approx(42.0)
    assert header.player_car_index == 7
    assert header.secondary_player_car_index == 255


def test_parse_header_too_short() -> None:
    with pytest.raises(PacketValidationError, match="trop court"):
        parse_header(bytes(10))


@pytest.mark.parametrize("packet_id", list(PacketId))
def test_validate_accepts_every_packet_type(make_packet, packet_id: PacketId) -> None:
    header, pid = validate_packet(make_packet(packet_id))
    assert pid is packet_id
    assert header.packet_id == packet_id


def test_validate_rejects_2026_format(make_packet) -> None:
    with pytest.raises(PacketValidationError, match="format 2026"):
        validate_packet(make_packet(PacketId.LAP_DATA, packet_format=2026))


def test_validate_rejects_unknown_packet_id(make_packet) -> None:
    # L'ID 16 (CarTelemetry2) n'existe qu'en format 2026.
    with pytest.raises(PacketValidationError, match="inconnu : 16"):
        validate_packet(make_packet(16, size=500))


@pytest.mark.parametrize("delta", [-1, +1, +100])
def test_validate_rejects_wrong_size(make_packet, delta: int) -> None:
    size = PACKET_SIZES[PacketId.CAR_DAMAGE] + delta
    with pytest.raises(PacketValidationError, match="CAR_DAMAGE"):
        validate_packet(make_packet(PacketId.CAR_DAMAGE, size=size))


def test_old_car_damage_size_is_rejected(make_packet) -> None:
    """Ancienne version de la spec F1 25 (953 octets) : en-tête correct, contenu différent (cf. Q6)."""
    with pytest.raises(PacketValidationError):
        validate_packet(make_packet(PacketId.CAR_DAMAGE, size=953))
