"""Structures des paquets F1 25 (format 2025) et leur format binaire.

Chaque structure de la spec est décrite deux fois, côte à côte :
- une `dataclass` (noms de champs Python, types, unités en commentaire) ;
- une `Layout` : la liste ordonnée (nom, format `struct`) des champs dans le paquet.

`Layout` vérifie au chargement que les noms correspondent exactement aux champs
de la dataclass, et les tests vérifient que les tailles correspondent à la spec.
Ordre des tableaux de 4 roues : 0 = arrière gauche (RL), 1 = arrière droite (RR),
2 = avant gauche (FL), 3 = avant droite (FR).
"""

from __future__ import annotations

import dataclasses
import re
import struct
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")

NUM_CARS = 22
WHEELS = ("RL", "RR", "FL", "FR")

_FMT_RE = re.compile(r"^(\d*)([a-zA-Z?])$")


class Layout(Generic[T]):
    """Format binaire d'une structure : champs `(nom, format)` dans l'ordre de la spec.

    Le format est celui du module `struct` : `B` uint8, `b` int8, `H` uint16, `h` int16,
    `I` uint32, `Q` uint64, `f` float, `d` double ; `4f` = tableau de 4 floats (rendu en tuple).
    """

    def __init__(self, cls: type[T] | None, fields: list[tuple[str, str]]) -> None:
        self.cls = cls
        self.fields = list(fields)
        self.names = [name for name, _ in fields]
        self._counts: list[int] = []
        for name, fmt in fields:
            match = _FMT_RE.match(fmt)
            if match is None:
                raise ValueError(f"format invalide pour {name} : {fmt}")
            self._counts.append(int(match.group(1) or 1))
        self.struct = struct.Struct("<" + "".join(fmt for _, fmt in fields))
        if cls is not None:
            expected = [f.name for f in dataclasses.fields(cls)]  # type: ignore[arg-type]
            if self.names != expected:
                raise TypeError(f"{cls.__name__} : champs {expected} ≠ layout {self.names}")

    @property
    def size(self) -> int:
        """Taille en octets."""
        return self.struct.size

    def values_from(self, data: bytes, offset: int = 0) -> list[Any]:
        """Décode les valeurs à `offset`, en regroupant les tableaux en tuples."""
        flat = self.struct.unpack_from(data, offset)
        values: list[Any] = []
        i = 0
        for count in self._counts:
            if count == 1:
                values.append(flat[i])
            else:
                values.append(tuple(flat[i : i + count]))
            i += count
        return values

    def unpack_from(self, data: bytes, offset: int = 0) -> T:
        """Décode une structure à `offset`."""
        assert self.cls is not None
        return self.cls(*self.values_from(data, offset))

    def unpack_array(self, data: bytes, offset: int, count: int) -> tuple[T, ...]:
        """Décode `count` structures consécutives à partir de `offset`."""
        return tuple(self.unpack_from(data, offset + k * self.size) for k in range(count))

    def pack(self, obj: T) -> bytes:
        """Encode une structure (sert à fabriquer des paquets de test)."""
        flat: list[Any] = []
        for name, count in zip(self.names, self._counts):
            value = getattr(obj, name)
            flat.extend(value if count > 1 else [value])
        return self.struct.pack(*flat)


# ---------------------------------------------------------------------------
# LapData (paquet 2) — une entrée par voiture
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LapData:
    """Temps et position d'une voiture sur le tour."""

    last_lap_time_ms: int
    current_lap_time_ms: int
    sector1_ms_part: int
    sector1_minutes_part: int
    sector2_ms_part: int
    sector2_minutes_part: int
    delta_to_car_in_front_ms_part: int
    delta_to_car_in_front_minutes_part: int
    delta_to_race_leader_ms_part: int
    delta_to_race_leader_minutes_part: int
    lap_distance: float  # m, négatif avant d'avoir franchi la ligne
    total_distance: float  # m
    safety_car_delta: float  # s
    car_position: int
    current_lap_num: int
    pit_status: int  # 0 = rien, 1 = entre aux stands, 2 = dans la voie des stands
    num_pit_stops: int
    sector: int  # 0, 1, 2 (= secteurs 1, 2, 3)
    current_lap_invalid: int  # 0 = valide, 1 = invalide
    penalties: int  # s
    total_warnings: int
    corner_cutting_warnings: int
    num_unserved_drive_through_pens: int
    num_unserved_stop_go_pens: int
    grid_position: int
    driver_status: int  # 0 = garage, 1 = tour lancé, 2 = tour de rentrée, 3 = tour de sortie, 4 = en piste
    result_status: int
    pit_lane_timer_active: int
    pit_lane_time_in_lane_ms: int
    pit_stop_timer_ms: int
    pit_stop_should_serve_pen: int
    speed_trap_fastest_speed: float  # km/h
    speed_trap_fastest_lap: int  # 255 = non défini

    @property
    def sector1_time_ms(self) -> int:
        """Temps du secteur 1 en ms (0 si pas encore réalisé)."""
        return self.sector1_minutes_part * 60_000 + self.sector1_ms_part

    @property
    def sector2_time_ms(self) -> int:
        """Temps du secteur 2 en ms (0 si pas encore réalisé)."""
        return self.sector2_minutes_part * 60_000 + self.sector2_ms_part

    @property
    def delta_to_car_in_front_ms(self) -> int:
        """Écart avec la voiture de devant, en ms."""
        return self.delta_to_car_in_front_minutes_part * 60_000 + self.delta_to_car_in_front_ms_part

    @property
    def delta_to_race_leader_ms(self) -> int:
        """Écart avec le leader, en ms."""
        return self.delta_to_race_leader_minutes_part * 60_000 + self.delta_to_race_leader_ms_part


LAP_DATA = Layout(LapData, [
    ("last_lap_time_ms", "I"), ("current_lap_time_ms", "I"),
    ("sector1_ms_part", "H"), ("sector1_minutes_part", "B"),
    ("sector2_ms_part", "H"), ("sector2_minutes_part", "B"),
    ("delta_to_car_in_front_ms_part", "H"), ("delta_to_car_in_front_minutes_part", "B"),
    ("delta_to_race_leader_ms_part", "H"), ("delta_to_race_leader_minutes_part", "B"),
    ("lap_distance", "f"), ("total_distance", "f"), ("safety_car_delta", "f"),
    ("car_position", "B"), ("current_lap_num", "B"), ("pit_status", "B"), ("num_pit_stops", "B"),
    ("sector", "B"), ("current_lap_invalid", "B"), ("penalties", "B"), ("total_warnings", "B"),
    ("corner_cutting_warnings", "B"), ("num_unserved_drive_through_pens", "B"),
    ("num_unserved_stop_go_pens", "B"), ("grid_position", "B"), ("driver_status", "B"),
    ("result_status", "B"), ("pit_lane_timer_active", "B"),
    ("pit_lane_time_in_lane_ms", "H"), ("pit_stop_timer_ms", "H"), ("pit_stop_should_serve_pen", "B"),
    ("speed_trap_fastest_speed", "f"), ("speed_trap_fastest_lap", "B"),
])


# ---------------------------------------------------------------------------
# CarTelemetry (paquet 6) — une entrée par voiture
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CarTelemetryData:
    """Pédales, vitesse, moteur et températures d'une voiture."""

    speed: int  # km/h
    throttle: float  # 0.0 → 1.0
    steer: float  # -1.0 (gauche) → 1.0 (droite)
    brake: float  # 0.0 → 1.0
    clutch: int  # 0 → 100
    gear: int  # 1-8, 0 = neutre, -1 = marche arrière
    engine_rpm: int
    drs: int  # 0 = fermé, 1 = ouvert
    rev_lights_percent: int
    rev_lights_bit_value: int
    brakes_temperature: tuple[int, int, int, int]  # °C, ordre RL RR FL FR
    tyres_surface_temperature: tuple[int, int, int, int]  # °C
    tyres_inner_temperature: tuple[int, int, int, int]  # °C
    engine_temperature: int  # °C
    tyres_pressure: tuple[float, float, float, float]  # PSI
    surface_type: tuple[int, int, int, int]  # voir annexe de la spec (0 = asphalte, 1 = vibreur…)


CAR_TELEMETRY = Layout(CarTelemetryData, [
    ("speed", "H"), ("throttle", "f"), ("steer", "f"), ("brake", "f"), ("clutch", "B"), ("gear", "b"),
    ("engine_rpm", "H"), ("drs", "B"), ("rev_lights_percent", "B"), ("rev_lights_bit_value", "H"),
    ("brakes_temperature", "4H"), ("tyres_surface_temperature", "4B"), ("tyres_inner_temperature", "4B"),
    ("engine_temperature", "H"), ("tyres_pressure", "4f"), ("surface_type", "4B"),
])


# ---------------------------------------------------------------------------
# CarStatus (paquet 7) — une entrée par voiture
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CarStatusData:
    """Aides, carburant, pneus montés et ERS d'une voiture."""

    traction_control: int  # 0 = off, 1 = medium, 2 = full
    anti_lock_brakes: int
    fuel_mix: int  # 0 = lean, 1 = standard, 2 = rich, 3 = max
    front_brake_bias: int  # %
    pit_limiter_status: int
    fuel_in_tank: float  # kg
    fuel_capacity: float
    fuel_remaining_laps: float  # valeur affichée sur le MFD
    max_rpm: int
    idle_rpm: int
    max_gears: int
    drs_allowed: int
    drs_activation_distance: int  # m, 0 = DRS non disponible
    actual_tyre_compound: int  # 16 = C5 … 20 = C1, 21 = C0, 22 = C6, 7 = inter, 8 = pluie
    visual_tyre_compound: int  # 16 = tendre, 17 = medium, 18 = dur, 7 = inter, 8 = pluie
    tyres_age_laps: int
    vehicle_fia_flags: int  # -1 = inconnu, 0 = aucun, 1 = vert, 2 = bleu, 3 = jaune
    engine_power_ice: float  # W
    engine_power_mguk: float  # W
    ers_store_energy: float  # J
    ers_deploy_mode: int  # 0 = aucun, 1 = medium, 2 = hotlap, 3 = overtake
    ers_harvested_this_lap_mguk: float  # J
    ers_harvested_this_lap_mguh: float  # J
    ers_deployed_this_lap: float  # J
    network_paused: int


CAR_STATUS = Layout(CarStatusData, [
    ("traction_control", "B"), ("anti_lock_brakes", "B"), ("fuel_mix", "B"), ("front_brake_bias", "B"),
    ("pit_limiter_status", "B"), ("fuel_in_tank", "f"), ("fuel_capacity", "f"), ("fuel_remaining_laps", "f"),
    ("max_rpm", "H"), ("idle_rpm", "H"), ("max_gears", "B"), ("drs_allowed", "B"),
    ("drs_activation_distance", "H"), ("actual_tyre_compound", "B"), ("visual_tyre_compound", "B"),
    ("tyres_age_laps", "B"), ("vehicle_fia_flags", "b"), ("engine_power_ice", "f"), ("engine_power_mguk", "f"),
    ("ers_store_energy", "f"), ("ers_deploy_mode", "B"), ("ers_harvested_this_lap_mguk", "f"),
    ("ers_harvested_this_lap_mguh", "f"), ("ers_deployed_this_lap", "f"), ("network_paused", "B"),
])


# ---------------------------------------------------------------------------
# CarDamage (paquet 10) — une entrée par voiture
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CarDamageData:
    """Usure et dégâts d'une voiture (pourcentages)."""

    tyres_wear: tuple[float, float, float, float]  # %, ordre RL RR FL FR
    tyres_damage: tuple[int, int, int, int]
    brakes_damage: tuple[int, int, int, int]
    tyre_blisters: tuple[int, int, int, int]
    front_left_wing_damage: int
    front_right_wing_damage: int
    rear_wing_damage: int
    floor_damage: int
    diffuser_damage: int
    sidepod_damage: int
    drs_fault: int
    ers_fault: int
    gear_box_damage: int
    engine_damage: int
    engine_mguh_wear: int
    engine_es_wear: int
    engine_ce_wear: int
    engine_ice_wear: int
    engine_mguk_wear: int
    engine_tc_wear: int
    engine_blown: int
    engine_seized: int


CAR_DAMAGE = Layout(CarDamageData, [
    ("tyres_wear", "4f"), ("tyres_damage", "4B"), ("brakes_damage", "4B"), ("tyre_blisters", "4B"),
    ("front_left_wing_damage", "B"), ("front_right_wing_damage", "B"), ("rear_wing_damage", "B"),
    ("floor_damage", "B"), ("diffuser_damage", "B"), ("sidepod_damage", "B"), ("drs_fault", "B"),
    ("ers_fault", "B"), ("gear_box_damage", "B"), ("engine_damage", "B"), ("engine_mguh_wear", "B"),
    ("engine_es_wear", "B"), ("engine_ce_wear", "B"), ("engine_ice_wear", "B"), ("engine_mguk_wear", "B"),
    ("engine_tc_wear", "B"), ("engine_blown", "B"), ("engine_seized", "B"),
])


# ---------------------------------------------------------------------------
# TimeTrial (paquet 14) — uniquement en Contre-la-montre
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TimeTrialDataSet:
    """Un tour de référence en Contre-la-montre (meilleur de la session, record perso ou rival)."""

    car_idx: int
    team_id: int
    lap_time_ms: int
    sector1_time_ms: int
    sector2_time_ms: int
    sector3_time_ms: int
    traction_control: int
    gearbox_assist: int
    anti_lock_brakes: int
    equal_car_performance: int
    custom_setup: int
    valid: int  # 0 = invalide, 1 = valide


TIME_TRIAL_DATA_SET = Layout(TimeTrialDataSet, [
    ("car_idx", "B"), ("team_id", "B"), ("lap_time_ms", "I"), ("sector1_time_ms", "I"),
    ("sector2_time_ms", "I"), ("sector3_time_ms", "I"), ("traction_control", "B"), ("gearbox_assist", "B"),
    ("anti_lock_brakes", "B"), ("equal_car_performance", "B"), ("custom_setup", "B"), ("valid", "B"),
])


# ---------------------------------------------------------------------------
# MotionEx (paquet 13) — voiture du joueur uniquement
# ---------------------------------------------------------------------------

Wheels = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class MotionExData:
    """Physique détaillée de la voiture du joueur (sert à détecter blocages de roue et glissades)."""

    suspension_position: Wheels
    suspension_velocity: Wheels
    suspension_acceleration: Wheels
    wheel_speed: Wheels
    wheel_slip_ratio: Wheels  # glissement longitudinal : forte valeur au freinage = roue qui bloque
    wheel_slip_angle: Wheels
    wheel_lat_force: Wheels
    wheel_long_force: Wheels
    height_of_cog_above_ground: float
    local_velocity_x: float  # m/s
    local_velocity_y: float
    local_velocity_z: float
    angular_velocity_x: float  # rad/s
    angular_velocity_y: float
    angular_velocity_z: float
    angular_acceleration_x: float  # rad/s²
    angular_acceleration_y: float
    angular_acceleration_z: float
    front_wheels_angle: float  # rad
    wheel_vert_force: Wheels
    front_aero_height: float
    rear_aero_height: float
    front_roll_angle: float
    rear_roll_angle: float
    chassis_yaw: float  # rad, par rapport à la direction du mouvement (tête-à-queue si grand)
    chassis_pitch: float
    wheel_camber: Wheels
    wheel_camber_gain: Wheels


MOTION_EX = Layout(MotionExData, [
    ("suspension_position", "4f"), ("suspension_velocity", "4f"), ("suspension_acceleration", "4f"),
    ("wheel_speed", "4f"), ("wheel_slip_ratio", "4f"), ("wheel_slip_angle", "4f"),
    ("wheel_lat_force", "4f"), ("wheel_long_force", "4f"), ("height_of_cog_above_ground", "f"),
    ("local_velocity_x", "f"), ("local_velocity_y", "f"), ("local_velocity_z", "f"),
    ("angular_velocity_x", "f"), ("angular_velocity_y", "f"), ("angular_velocity_z", "f"),
    ("angular_acceleration_x", "f"), ("angular_acceleration_y", "f"), ("angular_acceleration_z", "f"),
    ("front_wheels_angle", "f"), ("wheel_vert_force", "4f"), ("front_aero_height", "f"),
    ("rear_aero_height", "f"), ("front_roll_angle", "f"), ("rear_roll_angle", "f"), ("chassis_yaw", "f"),
    ("chassis_pitch", "f"), ("wheel_camber", "4f"), ("wheel_camber_gain", "4f"),
])


# ---------------------------------------------------------------------------
# Session (paquet 1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MarshalZone:
    """Zone de commissaires et son drapeau."""

    zone_start: float  # fraction du tour (0..1)
    zone_flag: int  # -1 = inconnu, 0 = aucun, 1 = vert, 2 = bleu, 3 = jaune


MARSHAL_ZONE = Layout(MarshalZone, [("zone_start", "f"), ("zone_flag", "b")])


@dataclass(frozen=True, slots=True)
class WeatherForecastSample:
    """Prévision météo à `time_offset` minutes."""

    session_type: int
    time_offset: int  # minutes
    weather: int  # 0 = clair … 3 = pluie légère, 4 = forte pluie, 5 = orage
    track_temperature: int  # °C
    track_temperature_change: int  # 0 = hausse, 1 = baisse, 2 = stable
    air_temperature: int
    air_temperature_change: int
    rain_percentage: int  # 0-100


WEATHER_FORECAST_SAMPLE = Layout(WeatherForecastSample, [
    ("session_type", "B"), ("time_offset", "B"), ("weather", "B"), ("track_temperature", "b"),
    ("track_temperature_change", "b"), ("air_temperature", "b"), ("air_temperature_change", "b"),
    ("rain_percentage", "B"),
])

MAX_MARSHAL_ZONES = 21
MAX_WEATHER_SAMPLES = 64
MAX_SESSIONS_IN_WEEKEND = 12

SESSION_TYPE_TIME_TRIAL = 18
SESSION_TYPES_RACE = (15, 16, 17)

# Le paquet Session contient deux tableaux au milieu : on le décrit en trois blocs.
SESSION_PART1 = Layout(None, [
    ("weather", "B"), ("track_temperature", "b"), ("air_temperature", "b"), ("total_laps", "B"),
    ("track_length", "H"), ("session_type", "B"), ("track_id", "b"), ("formula", "B"),
    ("session_time_left", "H"), ("session_duration", "H"), ("pit_speed_limit", "B"), ("game_paused", "B"),
    ("is_spectating", "B"), ("spectator_car_index", "B"), ("sli_pro_native_support", "B"),
    ("num_marshal_zones", "B"),
])
SESSION_PART2 = Layout(None, [("safety_car_status", "B"), ("network_game", "B"), ("num_weather_forecast_samples", "B")])
SESSION_PART3 = Layout(None, [
    ("forecast_accuracy", "B"), ("ai_difficulty", "B"), ("season_link_identifier", "I"),
    ("weekend_link_identifier", "I"), ("session_link_identifier", "I"), ("pit_stop_window_ideal_lap", "B"),
    ("pit_stop_window_latest_lap", "B"), ("pit_stop_rejoin_position", "B"), ("steering_assist", "B"),
    ("braking_assist", "B"), ("gearbox_assist", "B"), ("pit_assist", "B"), ("pit_release_assist", "B"),
    ("ers_assist", "B"), ("drs_assist", "B"), ("dynamic_racing_line", "B"), ("dynamic_racing_line_type", "B"),
    ("game_mode", "B"), ("rule_set", "B"), ("time_of_day", "I"), ("session_length", "B"),
    ("speed_units_lead_player", "B"), ("temperature_units_lead_player", "B"),
    ("speed_units_secondary_player", "B"), ("temperature_units_secondary_player", "B"),
    ("num_safety_car_periods", "B"), ("num_virtual_safety_car_periods", "B"), ("num_red_flag_periods", "B"),
    ("equal_car_performance", "B"), ("recovery_mode", "B"), ("flashback_limit", "B"), ("surface_type", "B"),
    ("low_fuel_mode", "B"), ("race_starts", "B"), ("tyre_temperature", "B"), ("pit_lane_tyre_sim", "B"),
    ("car_damage", "B"), ("car_damage_rate", "B"), ("collisions", "B"), ("collisions_off_for_first_lap_only", "B"),
    ("mp_unsafe_pit_release", "B"), ("mp_off_for_griefing", "B"), ("corner_cutting_stringency", "B"),
    ("parc_ferme_rules", "B"), ("pit_stop_experience", "B"), ("safety_car", "B"), ("safety_car_experience", "B"),
    ("formation_lap", "B"), ("formation_lap_experience", "B"), ("red_flags", "B"),
    ("affects_licence_level_solo", "B"), ("affects_licence_level_mp", "B"), ("num_sessions_in_weekend", "B"),
    ("weekend_structure", f"{MAX_SESSIONS_IN_WEEKEND}B"), ("sector2_lap_distance_start", "f"),
    ("sector3_lap_distance_start", "f"),
])


@dataclass(frozen=True, slots=True)
class SessionData:
    """Informations sur la session : piste, type, météo, règles, aides."""

    weather: int
    track_temperature: int  # °C
    air_temperature: int  # °C
    total_laps: int
    track_length: int  # m
    session_type: int  # 18 = Contre-la-montre, 15-17 = course (voir annexe)
    track_id: int
    formula: int
    session_time_left: int  # s
    session_duration: int  # s
    pit_speed_limit: int  # km/h
    game_paused: int
    is_spectating: int
    spectator_car_index: int
    sli_pro_native_support: int
    num_marshal_zones: int
    safety_car_status: int  # 0 = aucune, 1 = SC, 2 = VSC, 3 = tour de formation
    network_game: int
    num_weather_forecast_samples: int
    forecast_accuracy: int
    ai_difficulty: int
    season_link_identifier: int
    weekend_link_identifier: int
    session_link_identifier: int
    pit_stop_window_ideal_lap: int
    pit_stop_window_latest_lap: int
    pit_stop_rejoin_position: int
    steering_assist: int
    braking_assist: int
    gearbox_assist: int
    pit_assist: int
    pit_release_assist: int
    ers_assist: int
    drs_assist: int
    dynamic_racing_line: int
    dynamic_racing_line_type: int
    game_mode: int  # 5 = Contre-la-montre (voir annexe)
    rule_set: int  # 2 = Contre-la-montre
    time_of_day: int  # minutes depuis minuit
    session_length: int
    speed_units_lead_player: int
    temperature_units_lead_player: int
    speed_units_secondary_player: int
    temperature_units_secondary_player: int
    num_safety_car_periods: int
    num_virtual_safety_car_periods: int
    num_red_flag_periods: int
    equal_car_performance: int
    recovery_mode: int
    flashback_limit: int
    surface_type: int
    low_fuel_mode: int
    race_starts: int
    tyre_temperature: int
    pit_lane_tyre_sim: int
    car_damage: int
    car_damage_rate: int
    collisions: int
    collisions_off_for_first_lap_only: int
    mp_unsafe_pit_release: int
    mp_off_for_griefing: int
    corner_cutting_stringency: int
    parc_ferme_rules: int
    pit_stop_experience: int
    safety_car: int
    safety_car_experience: int
    formation_lap: int
    formation_lap_experience: int
    red_flags: int
    affects_licence_level_solo: int
    affects_licence_level_mp: int
    num_sessions_in_weekend: int
    weekend_structure: tuple[int, ...]
    sector2_lap_distance_start: float  # m
    sector3_lap_distance_start: float  # m
    marshal_zones: tuple[MarshalZone, ...]  # seulement les zones valides
    weather_forecast_samples: tuple[WeatherForecastSample, ...]  # seulement les prévisions valides

    @property
    def is_time_trial(self) -> bool:
        """Vrai en Contre-la-montre."""
        return self.session_type == SESSION_TYPE_TIME_TRIAL

    @property
    def is_race(self) -> bool:
        """Vrai pendant une course."""
        return self.session_type in SESSION_TYPES_RACE


SESSION_SIZE = (
    SESSION_PART1.size + MAX_MARSHAL_ZONES * MARSHAL_ZONE.size + SESSION_PART2.size
    + MAX_WEATHER_SAMPLES * WEATHER_FORECAST_SAMPLE.size + SESSION_PART3.size
)


def unpack_session(data: bytes, offset: int) -> SessionData:
    """Décode le contenu du paquet Session (après l'en-tête)."""
    part1 = SESSION_PART1.values_from(data, offset)
    offset += SESSION_PART1.size
    zones = MARSHAL_ZONE.unpack_array(data, offset, MAX_MARSHAL_ZONES)
    offset += MAX_MARSHAL_ZONES * MARSHAL_ZONE.size
    part2 = SESSION_PART2.values_from(data, offset)
    offset += SESSION_PART2.size
    samples = WEATHER_FORECAST_SAMPLE.unpack_array(data, offset, MAX_WEATHER_SAMPLES)
    offset += MAX_WEATHER_SAMPLES * WEATHER_FORECAST_SAMPLE.size
    part3 = SESSION_PART3.values_from(data, offset)
    num_zones = min(part1[-1], MAX_MARSHAL_ZONES)
    num_samples = min(part2[-1], MAX_WEATHER_SAMPLES)
    return SessionData(*part1, *part2, *part3, zones[:num_zones], samples[:num_samples])


# ---------------------------------------------------------------------------
# Event (paquet 3)
# ---------------------------------------------------------------------------

EVENT_DETAILS_SIZE = 12
"""Taille de l'union `EventDataDetails` (la plus grande variante : SpeedTrap)."""

EVENT_DETAIL_LAYOUTS: dict[str, Layout[None]] = {
    "FTLP": Layout(None, [("vehicle_idx", "B"), ("lap_time", "f")]),
    "RTMT": Layout(None, [("vehicle_idx", "B"), ("reason", "B")]),
    "DRSD": Layout(None, [("reason", "B")]),
    "TMPT": Layout(None, [("vehicle_idx", "B")]),
    "RCWN": Layout(None, [("vehicle_idx", "B")]),
    "PENA": Layout(None, [
        ("penalty_type", "B"), ("infringement_type", "B"), ("vehicle_idx", "B"), ("other_vehicle_idx", "B"),
        ("time", "B"), ("lap_num", "B"), ("places_gained", "B"),
    ]),
    "SPTP": Layout(None, [
        ("vehicle_idx", "B"), ("speed", "f"), ("is_overall_fastest_in_session", "B"),
        ("is_driver_fastest_in_session", "B"), ("fastest_vehicle_idx_in_session", "B"),
        ("fastest_speed_in_session", "f"),
    ]),
    "STLG": Layout(None, [("num_lights", "B")]),
    "DTSV": Layout(None, [("vehicle_idx", "B")]),
    "SGSV": Layout(None, [("vehicle_idx", "B"), ("stop_time", "f")]),
    "FLBK": Layout(None, [("flashback_frame_identifier", "I"), ("flashback_session_time", "f")]),
    "BUTN": Layout(None, [("button_status", "I")]),
    "OVTK": Layout(None, [("overtaking_vehicle_idx", "B"), ("being_overtaken_vehicle_idx", "B")]),
    "SCAR": Layout(None, [("safety_car_type", "B"), ("event_type", "B")]),
    "COLL": Layout(None, [("vehicle1_idx", "B"), ("vehicle2_idx", "B")]),
}
"""Détails par code d'événement. Codes sans détail : SSTA, SEND, DRSE, CHQF, LGOT, RDFL."""

EVENT_CODES_WITHOUT_DETAILS = frozenset({"SSTA", "SEND", "DRSE", "CHQF", "LGOT", "RDFL"})


def unpack_event(data: bytes, offset: int) -> tuple[str, dict[str, Any]]:
    """Décode le code (4 caractères) et les détails d'un événement."""
    code = data[offset : offset + 4].decode("ascii", errors="replace")
    layout = EVENT_DETAIL_LAYOUTS.get(code)
    if layout is None:
        return code, {}
    return code, dict(zip(layout.names, layout.values_from(data, offset + 4)))
