"""État courant de la session, reconstruit à partir des paquets décodés.

Les paquets arrivent séparément et à des rythmes différents (télémétrie 20 fois
par seconde, session 2 fois, événements quand ils se produisent). Le snapshot
garde la **dernière valeur connue** de chaque information : c'est la vue unique
que liront les experts de la Phase 2.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from telemetry.packet_parser import (
    PacketCarDamage,
    PacketCarStatus,
    PacketCarTelemetry,
    PacketEvent,
    PacketHeader,
    PacketLapData,
    PacketMotionEx,
    PacketSession,
    ParsedPacket,
)
from telemetry.packets import CarDamageData, CarStatusData, CarTelemetryData, LapData, MotionExData, SessionData

logger = logging.getLogger(__name__)

IGNORED_EVENT_CODES = frozenset({"BUTN"})
"""Événements sans intérêt pour l'analyse : `BUTN` = état des boutons, envoyé à chaque appui (≈ 3/s)."""

MAX_EVENTS = 50
"""Nombre d'événements récents conservés."""

RESULT_STATUS_ACTIVE = 2


@dataclass(frozen=True, slots=True)
class EventRecord:
    """Un événement de session, horodaté."""

    code: str
    session_time: float  # s
    details: dict[str, Any]


@dataclass
class TelemetrySnapshot:
    """Dernier état connu de la session et de la voiture du joueur."""

    session_uid: int | None = None
    session_time: float = 0.0  # s, horodatage du dernier paquet reçu
    player_index: int = 0
    session: SessionData | None = None
    lap: LapData | None = None
    telemetry: CarTelemetryData | None = None
    status: CarStatusData | None = None
    damage: CarDamageData | None = None
    motion_ex: MotionExData | None = None
    all_laps: tuple[LapData, ...] = ()
    """Données de tour de toutes les voitures (pour les écarts avec les adversaires)."""
    events: deque[EventRecord] = field(default_factory=lambda: deque(maxlen=MAX_EVENTS))
    packets_applied: int = 0

    # --- Mise à jour -------------------------------------------------------------

    def update(self, packet: ParsedPacket) -> None:
        """Intègre un paquet décodé dans l'état courant."""
        self._check_session(packet.header)
        self.session_time = packet.header.session_time
        self.player_index = packet.header.player_car_index
        self.packets_applied += 1

        if isinstance(packet, PacketLapData):
            self.all_laps = packet.cars
            self.lap = packet.player
        elif isinstance(packet, PacketCarTelemetry):
            self.telemetry = packet.player
        elif isinstance(packet, PacketCarStatus):
            self.status = packet.player
        elif isinstance(packet, PacketCarDamage):
            self.damage = packet.player
        elif isinstance(packet, PacketMotionEx):
            self.motion_ex = packet.motion
        elif isinstance(packet, PacketSession):
            self.session = packet.session
        elif isinstance(packet, PacketEvent):
            if packet.code not in IGNORED_EVENT_CODES:
                self.events.append(EventRecord(packet.code, packet.header.session_time, packet.details))

    def _check_session(self, header: PacketHeader) -> None:
        """Nouvelle session (autre circuit, redémarrage…) : on repart d'un état vide."""
        if self.session_uid is not None and header.session_uid != self.session_uid:
            logger.info("Nouvelle session détectée (%x) : réinitialisation du snapshot", header.session_uid)
            self.reset()
        self.session_uid = header.session_uid

    def reset(self) -> None:
        """Efface tout l'état."""
        fresh = TelemetrySnapshot()
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(fresh, name))

    # --- Lecture -------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """Vrai dès qu'on connaît au moins le tour et la télémétrie du joueur."""
        return self.lap is not None and self.telemetry is not None

    @property
    def is_time_trial(self) -> bool:
        """Vrai en Contre-la-montre (faux tant que le paquet Session n'est pas arrivé)."""
        return self.session is not None and self.session.is_time_trial

    @property
    def is_race(self) -> bool:
        """Vrai pendant une course."""
        return self.session is not None and self.session.is_race

    @property
    def gap_ahead_ms(self) -> int | None:
        """Écart avec la voiture de devant, en ms (None si premier ou inconnu)."""
        if self.lap is None or self.lap.car_position <= 1:
            return None
        return self.lap.delta_to_car_in_front_ms

    @property
    def gap_behind_ms(self) -> int | None:
        """Écart avec la voiture de derrière, en ms (None si dernier ou inconnu).

        Le jeu ne donne que l'écart avec la voiture de **devant** : on le lit donc
        chez la voiture classée juste derrière nous.
        """
        if self.lap is None:
            return None
        behind = self.car_at_position(self.lap.car_position + 1)
        return behind.delta_to_car_in_front_ms if behind else None

    def car_at_position(self, position: int) -> LapData | None:
        """Données de tour de la voiture active classée à `position`."""
        for car in self.all_laps:
            if car.car_position == position and car.result_status == RESULT_STATUS_ACTIVE:
                return car
        return None
