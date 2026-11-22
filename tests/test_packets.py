"""Tests du décodage du contenu des paquets (étape 1.3)."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import pytest

from telemetry import packets as p
from telemetry.packet_parser import (
    HEADER_SIZE,
    HEADER_STRUCT,
    PACKET_SIZES,
    PacketCarDamage,
    PacketCarStatus,
    PacketCarTelemetry,
    PacketEvent,
    PacketId,
    PacketLapData,
    PacketMotionEx,
    PacketSession,
    PacketTimeTrial,
    parse_packet,
)


def header(packet_id: PacketId, player_car_index: int = 0) -> bytes:
    """En-tête F1 25 valide."""
    return HEADER_STRUCT.pack(2025, 25, 1, 27, 1, packet_id, 42, 10.0, 1, 1, player_car_index, 255)


def sample(layout: p.Layout, seed: int):
    """Instance de test dont chaque champ a une valeur distincte, exactement représentable."""
    values = []
    for k, (_, fmt) in enumerate(layout.fields):
        count = int(fmt[:-1] or 1)
        kind = fmt[-1]
        items = []
        for j in range(count):
            n = seed * 7 + k * 3 + j
            items.append({"B": n % 250, "b": (n % 100) - 50, "H": 1000 + n, "h": -n,
                          "I": 100_000 + n, "f": n + 0.25}[kind])
        values.append(tuple(items) if count > 1 else items[0])
    return layout.cls(*values)


# --- Tailles : chaque structure doit donner exactement la taille de la spec ---------


@pytest.mark.parametrize(
    ("packet_id", "computed"),
    [
        (PacketId.LAP_DATA, HEADER_SIZE + 22 * p.LAP_DATA.size + 2),
        (PacketId.CAR_TELEMETRY, HEADER_SIZE + 22 * p.CAR_TELEMETRY.size + 3),
        (PacketId.CAR_STATUS, HEADER_SIZE + 22 * p.CAR_STATUS.size),
        (PacketId.CAR_DAMAGE, HEADER_SIZE + 22 * p.CAR_DAMAGE.size),
        (PacketId.SESSION, HEADER_SIZE + p.SESSION_SIZE),
        (PacketId.EVENT, HEADER_SIZE + 4 + p.EVENT_DETAILS_SIZE),
        (PacketId.MOTION_EX, HEADER_SIZE + p.MOTION_EX.size),
        (PacketId.TIME_TRIAL, HEADER_SIZE + 3 * p.TIME_TRIAL_DATA_SET.size),
    ],
)
def test_layout_sizes_match_spec(packet_id: PacketId, computed: int) -> None:
    assert computed == PACKET_SIZES[packet_id]


def test_event_detail_layouts_fit_in_union() -> None:
    assert max(layout.size for layout in p.EVENT_DETAIL_LAYOUTS.values()) == p.EVENT_DETAILS_SIZE


def test_layout_detects_name_mismatch() -> None:
    @dataclass
    class Two:
        a: int
        b: int

    with pytest.raises(TypeError):
        p.Layout(Two, [("a", "B"), ("c", "B")])


# --- LapData : test indépendant, octets écrits à la main d'après la spec ----------------


def test_lap_data_hand_packed_player_car() -> None:
    lap_fmt = "<IIHBHBHBHBfffBBBBBBBBBBBBBBBHHBfB"  # recopié de la spec, champ par champ
    empty = bytes(struct.calcsize(lap_fmt))
    player = struct.pack(
        lap_fmt,
        83_456, 41_200, 28_123, 0, 30_500, 0, 1_250, 0, 5_400, 1,
        2_345.5, 15_000.25, 0.0,
        3, 4, 0, 1, 1, 0, 0, 2, 1, 0, 0, 5, 1, 2, 0,
        0, 0, 0, 312.5, 3,
    )
    cars = [empty] * 22
    cars[5] = player
    data = header(PacketId.LAP_DATA, player_car_index=5) + b"".join(cars) + bytes([4, 255])

    packet = parse_packet(data)
    assert isinstance(packet, PacketLapData)
    lap = packet.player
    assert lap.last_lap_time_ms == 83_456
    assert lap.sector1_time_ms == 28_123
    assert lap.delta_to_race_leader_ms == 65_400  # 1 min + 5,4 s
    assert lap.lap_distance == pytest.approx(2_345.5)
    assert lap.car_position == 3
    assert lap.current_lap_num == 4
    assert lap.sector == 1
    assert lap.corner_cutting_warnings == 1
    assert lap.driver_status == 1
    assert lap.speed_trap_fastest_speed == pytest.approx(312.5)
    assert packet.time_trial_pb_car_idx == 4
    assert packet.time_trial_rival_car_idx == 255
    assert packet.cars[0].last_lap_time_ms == 0


# --- Aller-retour : encoder une valeur connue puis la décoder ------------------------


@pytest.mark.parametrize(
    ("packet_id", "layout", "extra", "packet_type"),
    [
        (PacketId.LAP_DATA, p.LAP_DATA, bytes([255, 255]), PacketLapData),
        (PacketId.CAR_TELEMETRY, p.CAR_TELEMETRY, struct.pack("<BBb", 255, 255, 7), PacketCarTelemetry),
        (PacketId.CAR_STATUS, p.CAR_STATUS, b"", PacketCarStatus),
        (PacketId.CAR_DAMAGE, p.CAR_DAMAGE, b"", PacketCarDamage),
    ],
)
def test_car_array_round_trip(packet_id, layout, extra, packet_type) -> None:
    cars = [sample(layout, seed) for seed in range(22)]
    data = header(packet_id, player_car_index=21) + b"".join(layout.pack(c) for c in cars) + extra
    packet = parse_packet(data)
    assert isinstance(packet, packet_type)
    assert packet.cars == tuple(cars)
    assert packet.player == cars[21]


def test_car_telemetry_suggested_gear() -> None:
    cars = b"".join(p.CAR_TELEMETRY.pack(sample(p.CAR_TELEMETRY, s)) for s in range(22))
    packet = parse_packet(header(PacketId.CAR_TELEMETRY) + cars + struct.pack("<BBb", 2, 255, 7))
    assert isinstance(packet, PacketCarTelemetry)
    assert (packet.mfd_panel_index, packet.suggested_gear) == (2, 7)


def test_motion_ex_round_trip() -> None:
    motion = sample(p.MOTION_EX, 3)
    packet = parse_packet(header(PacketId.MOTION_EX) + p.MOTION_EX.pack(motion))
    assert isinstance(packet, PacketMotionEx)
    assert packet.motion == motion
    assert len(packet.motion.wheel_slip_ratio) == 4


def test_time_trial_round_trip() -> None:
    sets = [sample(p.TIME_TRIAL_DATA_SET, s) for s in range(3)]
    packet = parse_packet(header(PacketId.TIME_TRIAL) + b"".join(p.TIME_TRIAL_DATA_SET.pack(x) for x in sets))
    assert isinstance(packet, PacketTimeTrial)
    assert (packet.player_session_best, packet.personal_best, packet.rival) == tuple(sets)


# --- Session ---------------------------------------------------------------------------


def _session_bytes(session_type: int, num_zones: int, num_samples: int) -> bytes:
    part1 = [0, 30, 22, 1, 5_412, session_type, 10, 0, 900, 900, 80, 0, 0, 255, 0, num_zones]
    zones = b"".join(struct.pack("<fb", z / 10, 1) for z in range(21))
    part2 = [0, 0, num_samples]
    samples = b"".join(struct.pack("<BBBbbbbB", session_type, 5 * (s % 50), 0, 30, 2, 22, 2, 10) for s in range(64))
    part3 = [0] * len(p.SESSION_PART3.names)
    part3[p.SESSION_PART3.names.index("game_mode")] = 5
    part3[p.SESSION_PART3.names.index("weekend_structure")] = tuple(range(12))
    part3[p.SESSION_PART3.names.index("sector3_lap_distance_start")] = 3_600.5
    flat3: list = []
    for v in part3:
        flat3.extend(v if isinstance(v, tuple) else [v])
    return (
        header(PacketId.SESSION)
        + p.SESSION_PART1.struct.pack(*part1)
        + zones
        + p.SESSION_PART2.struct.pack(*part2)
        + samples
        + p.SESSION_PART3.struct.pack(*flat3)
    )


def test_session_time_trial() -> None:
    packet = parse_packet(_session_bytes(session_type=18, num_zones=3, num_samples=2))
    assert isinstance(packet, PacketSession)
    s = packet.session
    assert s.is_time_trial and not s.is_race
    assert s.track_length == 5_412
    assert s.track_temperature == 30
    assert s.game_mode == 5
    assert s.weekend_structure == tuple(range(12))
    assert s.sector3_lap_distance_start == pytest.approx(3_600.5)
    assert len(s.marshal_zones) == 3
    assert s.marshal_zones[2].zone_start == pytest.approx(0.2)
    assert len(s.weather_forecast_samples) == 2
    assert s.weather_forecast_samples[1].time_offset == 5


def test_session_race() -> None:
    packet = parse_packet(_session_bytes(session_type=15, num_zones=0, num_samples=0))
    assert isinstance(packet, PacketSession)
    assert packet.session.is_race
    assert packet.session.marshal_zones == ()


# --- Event -----------------------------------------------------------------------------


def _event(code: str, details: bytes = b"") -> bytes:
    return header(PacketId.EVENT) + code.encode() + details.ljust(p.EVENT_DETAILS_SIZE, b"\0")


def test_event_fastest_lap() -> None:
    packet = parse_packet(_event("FTLP", struct.pack("<Bf", 0, 83.25)))
    assert isinstance(packet, PacketEvent)
    assert packet.code == "FTLP"
    assert packet.details == {"vehicle_idx": 0, "lap_time": pytest.approx(83.25)}


def test_event_flashback() -> None:
    packet = parse_packet(_event("FLBK", struct.pack("<If", 1234, 56.5)))
    assert isinstance(packet, PacketEvent)
    assert packet.details["flashback_frame_identifier"] == 1234
    assert packet.details["flashback_session_time"] == pytest.approx(56.5)


def test_event_buttons() -> None:
    packet = parse_packet(_event("BUTN", struct.pack("<I", 0x0001)))
    assert isinstance(packet, PacketEvent)
    assert packet.details == {"button_status": 1}


@pytest.mark.parametrize("code", ["SSTA", "SEND", "XXXX"])
def test_event_without_details(code: str) -> None:
    packet = parse_packet(_event(code))
    assert isinstance(packet, PacketEvent)
    assert packet.code == code
    assert packet.details == {}


# --- Types non décodés -------------------------------------------------------------------


@pytest.mark.parametrize("packet_id", [PacketId.MOTION, PacketId.PARTICIPANTS, PacketId.SESSION_HISTORY])
def test_unused_packet_types_return_none(packet_id: PacketId) -> None:
    data = header(packet_id) + bytes(PACKET_SIZES[packet_id] - HEADER_SIZE)
    assert parse_packet(data) is None
