"""Décodage binaire des paquets UDP F1 25 (format 2025).

Pour l'instant : l'en-tête commun (29 octets) et la validation d'un paquet
avant tout décodage de son contenu. Référence : `docs/SPEC_F1_25.md`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

PACKET_FORMAT = 2025
"""Seul format accepté (réglage « UDP Format 2025 » du jeu)."""

HEADER_STRUCT = struct.Struct("<HBBBBBQfIIBB")
"""En-tête : little-endian (`<`), packé, 29 octets."""

HEADER_SIZE = HEADER_STRUCT.size


class PacketId(IntEnum):
    """Types de paquets F1 25 (champ `m_packetId` de l'en-tête)."""

    MOTION = 0
    SESSION = 1
    LAP_DATA = 2
    EVENT = 3
    PARTICIPANTS = 4
    CAR_SETUPS = 5
    CAR_TELEMETRY = 6
    CAR_STATUS = 7
    FINAL_CLASSIFICATION = 8
    LOBBY_INFO = 9
    CAR_DAMAGE = 10
    SESSION_HISTORY = 11
    TYRE_SETS = 12
    MOTION_EX = 13
    TIME_TRIAL = 14
    LAP_POSITIONS = 15


PACKET_SIZES: dict[PacketId, int] = {
    PacketId.MOTION: 1349,
    PacketId.SESSION: 753,
    PacketId.LAP_DATA: 1285,
    PacketId.EVENT: 45,
    PacketId.PARTICIPANTS: 1284,
    PacketId.CAR_SETUPS: 1133,
    PacketId.CAR_TELEMETRY: 1352,
    PacketId.CAR_STATUS: 1239,
    PacketId.FINAL_CLASSIFICATION: 1042,
    PacketId.LOBBY_INFO: 954,
    PacketId.CAR_DAMAGE: 1041,
    PacketId.SESSION_HISTORY: 1460,
    PacketId.TYRE_SETS: 231,
    PacketId.MOTION_EX: 273,
    PacketId.TIME_TRIAL: 101,
    PacketId.LAP_POSITIONS: 1131,
}
"""Taille exacte (en-tête compris) de chaque paquet, d'après la spec F1 25 v3."""


class PacketValidationError(ValueError):
    """Paquet rejeté : format, type ou taille ne correspondent pas à la spec."""


@dataclass(frozen=True, slots=True)
class PacketHeader:
    """En-tête commun à tous les paquets."""

    packet_format: int
    game_year: int
    game_major_version: int
    game_minor_version: int
    packet_version: int
    packet_id: int
    session_uid: int
    session_time: float
    frame_identifier: int
    overall_frame_identifier: int
    player_car_index: int
    secondary_player_car_index: int


def parse_header(data: bytes) -> PacketHeader:
    """Décode les 29 premiers octets d'un paquet, sans rien vérifier d'autre que la longueur minimale."""
    if len(data) < HEADER_SIZE:
        raise PacketValidationError(f"paquet trop court pour un en-tête : {len(data)} octets < {HEADER_SIZE}")
    return PacketHeader(*HEADER_STRUCT.unpack_from(data, 0))


def validate_packet(data: bytes) -> tuple[PacketHeader, PacketId]:
    """Vérifie qu'un paquet peut être décodé avec la spec F1 25 et renvoie son en-tête et son type.

    Trois contrôles, dans l'ordre : format (2025), type connu, taille exacte.
    Lève `PacketValidationError` si l'un échoue.
    """
    header = parse_header(data)
    if header.packet_format != PACKET_FORMAT:
        raise PacketValidationError(f"format {header.packet_format} non supporté (attendu {PACKET_FORMAT})")
    try:
        packet_id = PacketId(header.packet_id)
    except ValueError:
        raise PacketValidationError(f"type de paquet inconnu : {header.packet_id}") from None
    expected = PACKET_SIZES[packet_id]
    if len(data) != expected:
        raise PacketValidationError(f"{packet_id.name} : {len(data)} octets reçus, {expected} attendus")
    return header, packet_id
