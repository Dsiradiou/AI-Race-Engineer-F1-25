# Notes sur la spec UDP F1 25 (format 2025)

Résumé, rédigé pour le projet, des points de la spec officielle EA dont nous avons besoin.
Sources (locales, non versionnées car © EA) : `docs/specs/Data Output from F1 25 v3.pdf` et
`docs/specs/F1 25 Telemetry Output Structures.txt`.

## Règles générales

- Tous les paquets commencent par le même **en-tête de 29 octets**.
- Encodage **little-endian**, structures **packées** (aucun octet de remplissage) → en Python : préfixe `<` dans `struct`.
- **22 voitures** dans les tableaux (`cs_maxNumCarsInUDPData = 22`) ; notre voiture = `m_playerCarIndex` de l'en-tête.
- Valeurs attendues dans l'en-tête : `m_packetFormat = 2025`, `m_gameYear = 25`, `m_packetVersion = 1` pour tous les paquets.

## En-tête (29 octets) — format `struct` : `<HBBBBBQfIIBB`

| Champ | Type | Rôle |
|---|---|---|
| `m_packetFormat` | uint16 | 2025 |
| `m_gameYear` | uint8 | 25 |
| `m_gameMajorVersion` / `m_gameMinorVersion` | uint8 | version du jeu |
| `m_packetVersion` | uint8 | version du type de paquet (1) |
| `m_packetId` | uint8 | type de paquet (voir ci-dessous) |
| `m_sessionUID` | uint64 | identifiant unique de la session |
| `m_sessionTime` | float | horodatage dans la session (s) |
| `m_frameIdentifier` | uint32 | n° de frame (peut reculer après un flashback) |
| `m_overallFrameIdentifier` | uint32 | n° de frame global (ne recule jamais) |
| `m_playerCarIndex` | uint8 | index de notre voiture dans les tableaux |
| `m_secondaryPlayerCarIndex` | uint8 | 255 si pas d'écran partagé |

## Types de paquets

| ID | Paquet | Taille (octets) | Fréquence | Utile pour |
|---|---|---|---|---|
| 0 | Motion | 1349 | réglage du menu (20 Hz) | forces G, position |
| 1 | Session | 753 | 2/s | type de session, piste, météo, secteurs |
| 2 | LapData | 1285 | réglage du menu | temps, secteur, position, écarts, tour invalide |
| 3 | Event | 45 | à l'événement | meilleur tour, flashback, drapeaux… |
| 4 | Participants | 1284 | toutes les 5 s | noms, équipes |
| 5 | CarSetups | 1133 | 2/s | réglages |
| 6 | CarTelemetry | 1352 | réglage du menu | vitesse, pédales, rapport, températures |
| 7 | CarStatus | 1239 | réglage du menu | carburant, ERS, pneus (composé, âge) |
| 8 | FinalClassification | 1042 | fin de course | classement final |
| 9 | LobbyInfo | 954 | 2/s en lobby | multijoueur |
| 10 | CarDamage | 1041 | 10/s | usure pneus, dégâts |
| 11 | SessionHistory | 1460 | 20/s, une voiture à la fois | historique des tours et secteurs |
| 12 | TyreSets | 231 | 20/s, une voiture à la fois | trains de pneus |
| 13 | MotionEx | 273 | réglage du menu | physique détaillée (notre voiture) |
| 14 | TimeTrial | 101 | 1/s, **uniquement en Contre-la-montre** | meilleur tour session / perso / rival |
| 15 | LapPositions | 1131 | 1/s | positions à chaque tour |

Les tailles de la spec ont été **recalculées à partir des structures** : les 16 concordent (étape 1.1).

## Reconnaître le Contre-la-montre

- Paquet Session : `m_sessionType = 18` (*Time Trial*), `m_gameMode = 5`, `m_ruleSet = 2`.
- Paquet LapData : `m_timeTrialPBCarIdx` et `m_timeTrialRivalCarIdx` (255 si invalide) → les voitures « fantômes ».
- Paquet TimeTrial (ID 14) : envoyé seulement dans ce mode.
- Course : `m_sessionType` de 15 à 17 (*Race*, *Race 2*, *Race 3*).

## À ne pas confondre

Le mode **« 2026 Season Pack »** utilise un autre format (`m_packetFormat = 2026`, 24 voitures, paquet 16 en plus,
tailles différentes). Notre jeu est réglé sur **UDP Format 2025** : tout paquet d'un autre format doit être rejeté.
