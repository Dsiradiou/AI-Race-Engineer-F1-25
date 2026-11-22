"""Décodage de vrais paquets enregistrés depuis F1 25 (`tests/fixtures/samples/*.bin`).

Les échantillons sont produits par : python -m tools.udp_probe --save-samples tests/fixtures/samples
On vérifie que les valeurs décodées sont **plausibles** : un mauvais décalage d'octets
donne presque toujours des valeurs absurdes (vitesse de 40 000 km/h, usure de 1e38 %…).
Si aucun échantillon n'est présent, ces tests sont ignorés.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from telemetry.packet_parser import (
    PacketCarDamage,
    PacketCarStatus,
    PacketCarTelemetry,
    PacketEvent,
    PacketLapData,
    PacketMotionEx,
    PacketSession,
    PacketTimeTrial,
    parse_packet,
)

SAMPLES_DIR = Path(__file__).parent / "fixtures" / "samples"
SAMPLES = sorted(SAMPLES_DIR.glob("*.bin"))

pytestmark = pytest.mark.skipif(not SAMPLES, reason="aucun échantillon réel dans tests/fixtures/samples")


def _load(prefix: str):
    path = next((s for s in SAMPLES if s.name.startswith(prefix)), None)
    if path is None:
        pytest.skip(f"pas d'échantillon {prefix}")
    return parse_packet(path.read_bytes())


@pytest.mark.parametrize("path", SAMPLES, ids=[s.stem for s in SAMPLES])
def test_every_sample_is_valid(path: Path) -> None:
    parse_packet(path.read_bytes())  # lève une exception si format, type ou taille ne vont pas


def test_lap_data_plausible() -> None:
    packet = _load("02_LAP_DATA")
    assert isinstance(packet, PacketLapData)
    lap = packet.player
    assert 0 <= lap.current_lap_time_ms < 10 * 60_000
    assert lap.last_lap_time_ms == 0 or 30_000 < lap.last_lap_time_ms < 10 * 60_000
    assert -500 < lap.lap_distance < 8_000
    assert 1 <= lap.current_lap_num < 100
    assert lap.sector in (0, 1, 2)
    assert lap.current_lap_invalid in (0, 1)
    assert 0 <= lap.driver_status <= 4


def test_car_telemetry_plausible() -> None:
    packet = _load("06_CAR_TELEMETRY")
    assert isinstance(packet, PacketCarTelemetry)
    car = packet.player
    assert 0 <= car.speed < 400
    assert 0.0 <= car.throttle <= 1.0 and 0.0 <= car.brake <= 1.0
    assert -1.0 <= car.steer <= 1.0
    assert -1 <= car.gear <= 8
    assert 0 <= car.engine_rpm < 16_000
    assert all(0 <= t < 1_500 for t in car.brakes_temperature)
    assert all(0 <= t < 200 for t in car.tyres_surface_temperature)
    assert all(10.0 < psi < 40.0 for psi in car.tyres_pressure)


def test_car_status_plausible() -> None:
    packet = _load("07_CAR_STATUS")
    assert isinstance(packet, PacketCarStatus)
    car = packet.player
    assert 0.0 <= car.fuel_in_tank <= car.fuel_capacity <= 150.0
    assert car.actual_tyre_compound in (7, 8, 16, 17, 18, 19, 20, 21, 22)
    assert car.visual_tyre_compound in (7, 8, 16, 17, 18)
    assert 0 <= car.tyres_age_laps < 100
    assert 0.0 <= car.ers_store_energy <= 4_000_000.0  # 4 MJ max


def test_car_damage_plausible() -> None:
    packet = _load("10_CAR_DAMAGE")
    assert isinstance(packet, PacketCarDamage)
    car = packet.player
    assert all(0.0 <= w <= 100.0 for w in car.tyres_wear)
    assert all(0 <= d <= 100 for d in car.brakes_damage)
    assert 0 <= car.front_left_wing_damage <= 100


def test_session_plausible() -> None:
    packet = _load("01_SESSION")
    assert isinstance(packet, PacketSession)
    s = packet.session
    assert 3_000 < s.track_length < 8_000
    assert 0 <= s.session_type <= 18
    assert -10 < s.track_temperature < 70
    assert 0 < s.sector2_lap_distance_start < s.sector3_lap_distance_start < s.track_length


def test_motion_ex_plausible() -> None:
    packet = _load("13_MOTION_EX")
    assert isinstance(packet, PacketMotionEx)
    m = packet.motion
    assert all(abs(v) < 150 for v in m.wheel_speed)  # m/s
    assert abs(m.local_velocity_z) < 120


def test_time_trial_plausible() -> None:
    packet = _load("14_TIME_TRIAL")
    assert isinstance(packet, PacketTimeTrial)
    best = packet.player_session_best
    assert best.lap_time_ms == 0 or 30_000 < best.lap_time_ms < 10 * 60_000
    assert best.valid in (0, 1)


def test_events_have_known_codes() -> None:
    events = [s for s in SAMPLES if s.name.startswith("03_EVENT")]
    if not events:
        pytest.skip("pas d'échantillon d'événement")
    for path in events:
        packet = parse_packet(path.read_bytes())
        assert isinstance(packet, PacketEvent)
        assert packet.code.isalpha() and packet.code.isupper()
