"""Décodage binaire des paquets UDP F1 25 (format 2025).

1. `validate_packet` : en-tête commun (29 octets) + contrôles format / type / taille.
2. `parse_packet` : décodage du contenu des paquets utiles au projet.
Les structures elles-mêmes sont décrites dans `telemetry/packets.py`.
Référence : `docs/SPEC_F1_25.md`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Union

from telemetry import packets as p

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


# ---------------------------------------------------------------------------
# Paquets décodés
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PacketLapData:
    """Paquet 2 : temps et positions de toutes les voitures."""

    header: PacketHeader
    cars: tuple[p.LapData, ...]
    time_trial_pb_car_idx: int  # voiture « record perso » en Contre-la-montre (255 = aucune)
    time_trial_rival_car_idx: int  # voiture « rival » en Contre-la-montre (255 = aucune)

    @property
    def player(self) -> p.LapData:
        """Données de la voiture du joueur."""
        return self.cars[self.header.player_car_index]


@dataclass(frozen=True, slots=True)
class PacketCarTelemetry:
    """Paquet 6 : télémétrie de toutes les voitures."""

    header: PacketHeader
    cars: tuple[p.CarTelemetryData, ...]
    mfd_panel_index: int
    mfd_panel_index_secondary_player: int
    suggested_gear: int  # 0 = aucune suggestion

    @property
    def player(self) -> p.CarTelemetryData:
        """Données de la voiture du joueur."""
        return self.cars[self.header.player_car_index]


@dataclass(frozen=True, slots=True)
class PacketCarStatus:
    """Paquet 7 : état de toutes les voitures."""

    header: PacketHeader
    cars: tuple[p.CarStatusData, ...]

    @property
    def player(self) -> p.CarStatusData:
        """Données de la voiture du joueur."""
        return self.cars[self.header.player_car_index]


@dataclass(frozen=True, slots=True)
class PacketCarDamage:
    """Paquet 10 : usure et dégâts de toutes les voitures."""

    header: PacketHeader
    cars: tuple[p.CarDamageData, ...]

    @property
    def player(self) -> p.CarDamageData:
        """Données de la voiture du joueur."""
        return self.cars[self.header.player_car_index]


@dataclass(frozen=True, slots=True)
class PacketSession:
    """Paquet 1 : informations de session."""

    header: PacketHeader
    session: p.SessionData


@dataclass(frozen=True, slots=True)
class PacketEvent:
    """Paquet 3 : un événement (code de 4 lettres + détails éventuels)."""

    header: PacketHeader
    code: str
    details: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PacketMotionEx:
    """Paquet 13 : physique détaillée de la voiture du joueur."""

    header: PacketHeader
    motion: p.MotionExData


@dataclass(frozen=True, slots=True)
class PacketTimeTrial:
    """Paquet 14 : tours de référence en Contre-la-montre."""

    header: PacketHeader
    player_session_best: p.TimeTrialDataSet
    personal_best: p.TimeTrialDataSet
    rival: p.TimeTrialDataSet


ParsedPacket = Union[
    PacketLapData, PacketCarTelemetry, PacketCarStatus, PacketCarDamage,
    PacketSession, PacketEvent, PacketMotionEx, PacketTimeTrial,
]

_LAP_DATA_EXTRA = struct.Struct("<BB")
_CAR_TELEMETRY_EXTRA = struct.Struct("<BBb")


def _decode_lap_data(h: PacketHeader, data: bytes) -> PacketLapData:
    cars = p.LAP_DATA.unpack_array(data, HEADER_SIZE, p.NUM_CARS)
    pb, rival = _LAP_DATA_EXTRA.unpack_from(data, HEADER_SIZE + p.NUM_CARS * p.LAP_DATA.size)
    return PacketLapData(h, cars, pb, rival)


def _decode_car_telemetry(h: PacketHeader, data: bytes) -> PacketCarTelemetry:
    cars = p.CAR_TELEMETRY.unpack_array(data, HEADER_SIZE, p.NUM_CARS)
    extra = _CAR_TELEMETRY_EXTRA.unpack_from(data, HEADER_SIZE + p.NUM_CARS * p.CAR_TELEMETRY.size)
    return PacketCarTelemetry(h, cars, *extra)


def _decode_car_status(h: PacketHeader, data: bytes) -> PacketCarStatus:
    return PacketCarStatus(h, p.CAR_STATUS.unpack_array(data, HEADER_SIZE, p.NUM_CARS))


def _decode_car_damage(h: PacketHeader, data: bytes) -> PacketCarDamage:
    return PacketCarDamage(h, p.CAR_DAMAGE.unpack_array(data, HEADER_SIZE, p.NUM_CARS))


def _decode_session(h: PacketHeader, data: bytes) -> PacketSession:
    return PacketSession(h, p.unpack_session(data, HEADER_SIZE))


def _decode_event(h: PacketHeader, data: bytes) -> PacketEvent:
    code, details = p.unpack_event(data, HEADER_SIZE)
    return PacketEvent(h, code, details)


def _decode_motion_ex(h: PacketHeader, data: bytes) -> PacketMotionEx:
    return PacketMotionEx(h, p.MOTION_EX.unpack_from(data, HEADER_SIZE))


def _decode_time_trial(h: PacketHeader, data: bytes) -> PacketTimeTrial:
    sets = p.TIME_TRIAL_DATA_SET.unpack_array(data, HEADER_SIZE, 3)
    return PacketTimeTrial(h, *sets)


DECODERS: dict[PacketId, Callable[[PacketHeader, bytes], ParsedPacket]] = {
    PacketId.LAP_DATA: _decode_lap_data,
    PacketId.CAR_TELEMETRY: _decode_car_telemetry,
    PacketId.CAR_STATUS: _decode_car_status,
    PacketId.CAR_DAMAGE: _decode_car_damage,
    PacketId.SESSION: _decode_session,
    PacketId.EVENT: _decode_event,
    PacketId.MOTION_EX: _decode_motion_ex,
    PacketId.TIME_TRIAL: _decode_time_trial,
}
"""Paquets décodés par le projet. Les autres types sont validés puis ignorés."""


def parse_packet(data: bytes) -> ParsedPacket | None:
    """Valide puis décode un paquet brut.

    Renvoie `None` pour un type valide mais non utilisé par le projet.
    Lève `PacketValidationError` si le paquet n'est pas conforme à la spec.
    """
    header, packet_id = validate_packet(data)
    decoder = DECODERS.get(packet_id)
    return decoder(header, data) if decoder else None
