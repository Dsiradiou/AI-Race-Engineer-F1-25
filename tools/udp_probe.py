"""Sonde UDP : affiche ce que le jeu envoie réellement, paquet par paquet.

Pour chaque combinaison (format, type de paquet), compte les paquets reçus,
relève leurs tailles et indique s'ils passent la validation de la spec F1 25.

Lancement (jeu en piste) :
    python -m tools.udp_probe --duration 30
    python -m tools.udp_probe --duration 60 --save-samples tests/fixtures/samples

Avec `--save-samples`, le dernier paquet valide de chaque type (et de chaque code
d'événement) est enregistré tel quel dans un fichier `.bin`, pour servir de données
réelles dans les tests. Les paquets contenant des noms de joueurs (Participants,
LobbyInfo) ne sont jamais enregistrés.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import time
from collections import Counter, defaultdict
from pathlib import Path

from config import settings
from telemetry.packet_parser import (
    HEADER_SIZE,
    PACKET_SIZES,
    PacketId,
    PacketValidationError,
    parse_header,
    validate_packet,
)
from telemetry.udp_receiver import UdpReceiver

PRIVATE_PACKETS = frozenset({PacketId.PARTICIPANTS, PacketId.LOBBY_INFO})
"""Paquets contenant des noms de joueurs : jamais enregistrés."""


class ProbeStats:
    """Statistiques accumulées par la sonde."""

    def __init__(self) -> None:
        self.counts: Counter[tuple[int, int]] = Counter()
        self.sizes: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
        self.rejections: Counter[str] = Counter()
        self.first_seen: dict[tuple[int, int], float] = {}
        self.last_seen: dict[tuple[int, int], float] = {}
        self.valid = 0
        self.first_header = None
        self.event_codes: Counter[str] = Counter()
        self.samples: dict[str, bytes] = {}
        """Dernier paquet valide par type (et par code d'événement), prêt à être enregistré."""

    def add(self, data: bytes, now: float | None = None) -> None:
        """Enregistre un paquet brut reçu à l'instant `now` (horloge monotone, en secondes)."""
        now = time.monotonic() if now is None else now
        try:
            header = parse_header(data)
        except PacketValidationError as exc:
            self.rejections[str(exc)] += 1
            return
        if self.first_header is None:
            self.first_header = header
        key = (header.packet_format, header.packet_id)
        self.counts[key] += 1
        self.sizes[key].add(len(data))
        self.first_seen.setdefault(key, now)
        self.last_seen[key] = now
        try:
            _, packet_id = validate_packet(data)
        except PacketValidationError as exc:
            self.rejections[str(exc)] += 1
            return
        self.valid += 1
        name = f"{packet_id.value:02d}_{packet_id.name}"
        if packet_id is PacketId.EVENT:
            code = data[HEADER_SIZE : HEADER_SIZE + 4].decode("ascii", errors="replace")
            self.event_codes[code] += 1
            name += f"_{code}"
        if packet_id not in PRIVATE_PACKETS:
            self.samples[name] = data

    def report(self, elapsed: float) -> str:
        """Tableau récapitulatif lisible en console.

        `actif(s)` : temps entre le premier et le dernier paquet de ce type ; `/s` : fréquence mesurée sur cette fenêtre.
        """
        total = sum(self.counts.values())
        lines = [f"\n=== Sonde UDP — {total} paquets en {elapsed:.0f} s — {self.valid} valides ==="]
        h = self.first_header
        if h is not None:
            lines.append(
                f"Jeu : format {h.packet_format}, année {h.game_year}, "
                f"version {h.game_major_version}.{h.game_minor_version:02d}, voiture joueur n°{h.player_car_index}"
            )
        lines.append(
            f"{'format':>6} {'id':>3} {'paquet':<21} {'reçus':>6} {'actif(s)':>9} {'/s':>6} {'taille(s)':>12} {'attendu':>8}"
        )
        for (fmt, pid), n in sorted(self.counts.items()):
            name = PacketId(pid).name if pid in PacketId._value2member_map_ else "?"
            expected = PACKET_SIZES.get(PacketId(pid)) if name != "?" else None
            sizes = ",".join(str(s) for s in sorted(self.sizes[(fmt, pid)]))
            active = self.last_seen[(fmt, pid)] - self.first_seen[(fmt, pid)]
            rate = (n - 1) / active if n > 1 and active > 0 else 0.0
            lines.append(
                f"{fmt:>6} {pid:>3} {name:<21} {n:>6} {active:>9.1f} {rate:>6.1f} {sizes:>12} {expected or '-':>8}"
            )
        if self.event_codes:
            codes = ", ".join(f"{code} × {n}" for code, n in self.event_codes.most_common())
            lines.append(f"Codes d'événements : {codes}")
        if self.rejections:
            lines.append("Rejets :")
            lines.extend(f"  {n:>6} × {reason}" for reason, n in self.rejections.most_common())
        return "\n".join(lines)

    def save_samples(self, directory: Path) -> list[Path]:
        """Écrit un fichier `.bin` par échantillon et renvoie les chemins créés."""
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, data in sorted(self.samples.items()):
            path = directory / f"{name}.bin"
            path.write_bytes(data)
            paths.append(path)
        return paths


async def run_probe(host: str, port: int, duration: float) -> ProbeStats:
    """Écoute pendant `duration` secondes et renvoie les statistiques."""
    receiver = UdpReceiver(host, port)
    await receiver.start()
    stats = ProbeStats()
    print(f"Écoute sur {receiver.host}:{receiver.port} pendant {duration:.0f} s… (mets la voiture en piste)")
    loop = asyncio.get_running_loop()
    end = loop.time() + duration
    try:
        while (remaining := end - loop.time()) > 0:
            with contextlib.suppress(asyncio.TimeoutError):
                stats.add(await asyncio.wait_for(receiver.receive(), timeout=remaining))
    finally:
        receiver.close()
    if receiver.dropped:
        print(f"Attention : {receiver.dropped} paquets jetés (file pleine).")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Sonde UDP F1 25")
    parser.add_argument("--host", default=settings.udp_host)
    parser.add_argument("--port", type=int, default=settings.udp_port)
    parser.add_argument("--duration", type=float, default=30.0, help="durée d'écoute en secondes")
    parser.add_argument("--save-samples", type=Path, help="dossier où enregistrer un paquet réel par type")
    args = parser.parse_args()
    stats = asyncio.run(run_probe(args.host, args.port, args.duration))
    print(stats.report(args.duration))
    if args.save_samples:
        paths = stats.save_samples(args.save_samples)
        print(f"{len(paths)} échantillons enregistrés dans {args.save_samples}")


if __name__ == "__main__":
    main()
