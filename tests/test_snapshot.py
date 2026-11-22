"""Tests du TelemetrySnapshot (étape 1.4)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from telemetry import packets as p
from telemetry.packet_parser import (
    HEADER_STRUCT,
    PacketEvent,
    PacketHeader,
    PacketId,
    PacketLapData,
    parse_packet,
)
from telemetry.snapshot import MAX_EVENTS, TelemetrySnapshot

SAMPLES_DIR = Path(__file__).parent / "fixtures" / "samples"


def _header(packet_id: PacketId, session_uid: int = 1, session_time: float = 10.0, player: int = 0) -> PacketHeader:
    return PacketHeader(*HEADER_STRUCT.unpack(HEADER_STRUCT.pack(
        2025, 25, 1, 27, 1, packet_id, session_uid, session_time, 1, 1, player, 255)))


def _zero_lap(**changes) -> p.LapData:
    values = [tuple([0] * int(fmt[:-1])) if fmt[:-1] else (0.0 if fmt == "f" else 0) for _, fmt in p.LAP_DATA.fields]
    return dataclasses.replace(p.LapData(*values), **changes)


def _lap_packet(cars: list[p.LapData], player: int = 0, session_uid: int = 1) -> PacketLapData:
    return PacketLapData(_header(PacketId.LAP_DATA, session_uid, player=player), tuple(cars), 255, 255)


def _event(code: str, t: float = 5.0, session_uid: int = 1) -> PacketEvent:
    return PacketEvent(_header(PacketId.EVENT, session_uid, session_time=t), code, {})


# --- Données réelles : le snapshot se remplit avec les échantillons du jeu --------------


@pytest.mark.skipif(not any(SAMPLES_DIR.glob("*.bin")), reason="pas d'échantillons réels")
def test_snapshot_from_real_samples() -> None:
    snap = TelemetrySnapshot()
    assert not snap.is_ready
    for path in sorted(SAMPLES_DIR.glob("*.bin")):
        packet = parse_packet(path.read_bytes())
        if packet is not None:
            snap.update(packet)
    assert snap.is_ready
    assert snap.is_time_trial and not snap.is_race
    assert snap.session is not None and snap.session.track_id == 32  # Losail
    assert snap.telemetry is not None and snap.telemetry.speed == 319
    assert snap.status is not None and snap.status.visual_tyre_compound == 16
    assert snap.damage is not None and snap.motion_ex is not None
    assert len(snap.all_laps) == 22
    # Les événements BUTN (gâchette d'accélérateur) sont filtrés, les autres gardés.
    assert [e.code for e in snap.events] == ["PENA", "SPTP"]


# --- Mise à jour -------------------------------------------------------------------------


def test_keeps_latest_values() -> None:
    snap = TelemetrySnapshot()
    snap.update(_lap_packet([_zero_lap(current_lap_num=1)] * 22))
    snap.update(_lap_packet([_zero_lap(current_lap_num=2)] * 22))
    assert snap.lap is not None and snap.lap.current_lap_num == 2
    assert snap.packets_applied == 2


def test_player_index_from_header() -> None:
    cars = [_zero_lap(car_position=i + 1) for i in range(22)]
    snap = TelemetrySnapshot()
    snap.update(_lap_packet(cars, player=7))
    assert snap.player_index == 7
    assert snap.lap is not None and snap.lap.car_position == 8


def test_new_session_resets_state() -> None:
    snap = TelemetrySnapshot()
    snap.update(_lap_packet([_zero_lap(current_lap_num=5)] * 22, session_uid=1))
    snap.update(_event("FTLP", session_uid=1))
    snap.update(_event("SSTA", session_uid=2))  # nouvelle session
    assert snap.session_uid == 2
    assert snap.lap is None
    assert [e.code for e in snap.events] == ["SSTA"]


# --- Événements -------------------------------------------------------------------------


def test_button_events_are_ignored() -> None:
    snap = TelemetrySnapshot()
    for _ in range(10):
        snap.update(_event("BUTN"))
    snap.update(_event("FLBK", t=42.0))
    assert len(snap.events) == 1
    assert snap.events[0].code == "FLBK"
    assert snap.events[0].session_time == pytest.approx(42.0)


def test_event_history_is_bounded() -> None:
    snap = TelemetrySnapshot()
    for i in range(MAX_EVENTS + 10):
        snap.update(_event("OVTK", t=float(i)))
    assert len(snap.events) == MAX_EVENTS
    assert snap.events[0].session_time == pytest.approx(10.0)  # les plus anciens sont sortis


# --- Écarts avec les adversaires -----------------------------------------------------------


def _race(player_position: int) -> TelemetrySnapshot:
    """Course à 5 voitures actives, 1,5 s entre chaque ; le joueur est la voiture 0."""
    cars = [_zero_lap() for _ in range(22)]
    order = [0, 1, 2, 3, 4]
    order.remove(0)
    order.insert(player_position - 1, 0)
    for position, car_index in enumerate(order, start=1):
        gap = 0 if position == 1 else 1_500 + car_index  # écart avec la voiture de devant
        cars[car_index] = _zero_lap(car_position=position, result_status=2, delta_to_car_in_front_ms_part=gap)
    snap = TelemetrySnapshot()
    snap.update(_lap_packet(cars))
    return snap


def test_gaps_in_the_middle_of_the_pack() -> None:
    snap = _race(player_position=3)
    assert snap.gap_ahead_ms == 1_500  # écart du joueur (voiture 0) avec la voiture de devant
    behind = snap.car_at_position(4)
    assert behind is not None
    assert snap.gap_behind_ms == behind.delta_to_car_in_front_ms


def test_gaps_for_leader_and_last() -> None:
    assert _race(player_position=1).gap_ahead_ms is None
    assert _race(player_position=5).gap_behind_ms is None


def test_gap_with_minutes_part() -> None:
    cars = [_zero_lap() for _ in range(22)]
    cars[0] = _zero_lap(car_position=2, result_status=2,
                        delta_to_car_in_front_minutes_part=1, delta_to_car_in_front_ms_part=2_500)
    cars[1] = _zero_lap(car_position=1, result_status=2)
    snap = TelemetrySnapshot()
    snap.update(_lap_packet(cars))
    assert snap.gap_ahead_ms == 62_500
