"""Sonde UDP : affiche ce que le jeu envoie réellement, paquet par paquet.

Pour chaque combinaison (format, type de paquet), compte les paquets reçus,
relève leurs tailles et indique s'ils passent la validation de la spec F1 25.

Lancement (jeu en piste) :
    python -m tools.udp_probe --duration 30
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import time
from collections import Counter, defaultdict

from config import settings
from telemetry.packet_parser import PACKET_SIZES, PacketId, PacketValidationError, parse_header, validate_packet
from telemetry.udp_receiver import UdpReceiver


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
            validate_packet(data)
            self.valid += 1
        except PacketValidationError as exc:
            self.rejections[str(exc)] += 1

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
        if self.rejections:
            lines.append("Rejets :")
            lines.extend(f"  {n:>6} × {reason}" for reason, n in self.rejections.most_common())
        return "\n".join(lines)


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
    args = parser.parse_args()
    stats = asyncio.run(run_probe(args.host, args.port, args.duration))
    print(stats.report(args.duration))


if __name__ == "__main__":
    main()
