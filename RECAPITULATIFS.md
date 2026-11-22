# Récapitulatifs des étapes — AI Race Engineer (F1 25)

Un récapitulatif est ajouté à la fin de chaque étape du `JOURNAL_DE_BORD.md`,
**avant** le feu vert et le commit. Les entrées sont ajoutées en bas, dans l'ordre chronologique.

Chaque bilan est aussi un **moment d'apprentissage** : les concepts d'IA liés à l'étape
y sont expliqués avec des exemples concrets tirés du projet.

## Modèle

```markdown
### Étape X.Y — <titre> (AAAA-MM-JJ)

- **Fait :** ce qui a été construit / modifié (fichiers).
- **Tests :** commandes lancées et résultat (ou « pas de test nécessaire » + pourquoi).
- **Mesures :** latence, RAM/VRAM… (si pertinent).
- **Concepts IA :** explication des notions abordées, illustrée par ce qu'on vient de construire (+ une question de vérification, consignée avec réponse et correction dans `QUESTIONS_PEDAGOGIQUES.md`).
- **Points ouverts :** limites connues, doutes, choses à vérifier.
- **Suite proposée :** prochaine étape.
- **Statut :** en attente du feu vert / validé et commité (`<hash>`).
```

---

### Étape 0.1 — Dépôt GitHub lié au dossier local (2026-11-22)

- **Fait :** suppression du dépôt Git vide créé par erreur dans `C:\Users\Siradiou` ; dépôt local initialisé dans `AI Race Engineer`, remote `origin` → `https://github.com/Dsiradiou/AI-Race-Engineer-F1-25.git`, branche `main`. Ajout de `JOURNAL_DE_BORD.md` et `RECAPITULATIFS.md`.
- **Tests :** `git status` → `main` à jour avec `origin/main` ; `git remote -v` OK.
- **Points ouverts :** choix de la version de Python (seul 3.14 est installé ; 3.12 recommandé pour la compatibilité PyTorch/CUDA, `webrtcvad`, Whisper).
- **Suite proposée :** étape 0.2 — installer Python et créer le venv.
- **Décisions :** Python 3.12 retenu pour l'étape 0.2.
- **Statut :** validé et commité.

### Étape 0.2 — Python 3.12 + environnement virtuel (2026-11-22)

- **Fait :** Python 3.12.10 (déjà présent via le Python install manager, à côté de 3.14.3) ; venv créé dans `.venv/` avec `py -V:3.12 -m venv .venv` ; pip mis à jour (26.3).
- **Tests :** `.venv\Scripts\python.exe --version` → 3.12.10 ; `sys.prefix` pointe bien sur `.venv`. GPU vérifié avec `nvidia-smi` : RTX 5060 Laptop, 8 Go VRAM (8151 MiB), pilote 621.01, *compute capability* 12.0.
- **Concepts IA :**
  - **Pourquoi un GPU pour l'IA ?** Un modèle d'IA, c'est surtout des **multiplications de matrices**. Quand Llama 3.1 8B génère un token, il multiplie un vecteur (la représentation du texte) par des matrices de poids, couche après couche — environ 8 milliards de multiplications-additions par token. Un CPU a ~16 cœurs très polyvalents ; le GPU a des **milliers de petits cœurs** qui font tous la même opération en parallèle. Pour ce calcul répétitif, le GPU est 10 à 100× plus rapide.
  - **VRAM vs RAM :** le GPU ne calcule que sur ce qui est dans **sa** mémoire (VRAM, 8 Go ici, très rapide : plusieurs centaines de Go/s). Si le modèle ne tient pas, une partie reste en RAM et doit traverser le bus PCIe (beaucoup plus lent) → c'est le même problème que l'*offloading* des experts MoE vu plus tôt. D'où notre budget : un LLM quantifié de ~5 Go + Whisper doivent tenir dans 8 Go de VRAM.
  - **CUDA et *compute capability* :** CUDA est la plateforme qui permet aux bibliothèques (PyTorch) de programmer le GPU NVIDIA. La *compute capability* 12.0 identifie l'architecture **Blackwell** (RTX 50xx) : PyTorch doit avoir été compilé pour elle (CUDA 12.8 minimum) — on le vérifiera en Phase 4 au lieu de le supposer.
  - **Pourquoi un venv et Python 3.12 ?** Les bibliothèques d'IA (PyTorch, Whisper) publient des versions compilées pour chaque version de Python ; elles arrivent souvent avec retard sur la plus récente (3.14). Le venv isole les dépendances du projet pour qu'elles soient reproductibles (`requirements.txt`).
  - **Question de vérification :** Q2 dans `QUESTIONS_PEDAGOGIQUES.md` (poids en RAM vs VRAM et vitesse de génération).
- **Points ouverts :** `.venv/` apparaît comme non suivi dans Git → il sera ignoré par le `.gitignore` de l'étape 0.3.
- **Suite proposée :** étape 0.3 — arborescence du projet, `.gitignore`, `requirements.txt`, `.env.example`, `config.py`.
- **Statut :** validé et commité.

### Étape 0.3 — Arborescence, configuration et dépendances (2026-11-22)

- **Fait :**
  - Paquets `telemetry/`, `analyzer/`, `voice/`, `ai/`, `tools/` avec un `__init__.py` qui documente leur rôle (aucun module vide « pour plus tard ») ; dossiers `tests/fixtures/` et `logs/` (gardés via `.gitkeep`).
  - `.gitignore` : `.venv/`, caches Python, `.env`, contenu de `logs/`.
  - `.env.example` : port UDP, URL et modèle Ollama, modèle Whisper, clé ElevenLabs (vide).
  - `config.py` : `Settings` (lus depuis `.env`, avec valeurs par défaut) + `Thresholds` (seuils de la section 6, non modifiables à l'exécution). Seul choix ajouté : palier d'usure « critique » fixé à 90 % (non précisé dans le cadrage, à affiner en Phase 7).
  - `requirements.txt` minimal (versions figées) : `python-dotenv`, `pytest`. Les autres bibliothèques seront ajoutées dans la phase qui les utilise.
  - `pyproject.toml` : métadonnées du projet + configuration de pytest.
- **Tests :** `pytest -v` → **4 passed** (`tests/test_config.py` : valeurs par défaut, surcharge par variables d'environnement, seuils, immutabilité). `.venv/` n'apparaît plus dans `git status`.
- **Concepts IA :**
  - **Paramètres réglés à la main vs paramètres appris.** Nos seuils (`lap_delta_s = 0.5`, `tyre_age_laps = 20`…) sont des règles écrites par un humain : c'est ce qui fait de notre router un système **à règles**. Dans un modèle d'apprentissage automatique, ces valeurs ne sont pas écrites : le modèle les **apprend à partir de données**. Les 8 milliards de *poids* de Llama sont exactement ça : des nombres ajustés automatiquement pendant l'entraînement. Le réglage manuel des seuils prévu en Phase 7 (rouler, observer, corriger) est une version artisanale de ce que fait l'entraînement.
  - **Hyperparamètres.** Certains réglages ne sont pas appris mais choisis avant ou autour du modèle : taille du modèle, température de génération, nombre d'experts actifs (top-k) dans un MoE… Dans `config.py`, `OLLAMA_MODEL` et `WHISPER_MODEL` relèvent de ce type de choix.
  - **Reproductibilité.** Versions figées dans `requirements.txt` + paramètres centralisés dans un seul fichier = on peut refaire exactement la même expérience plus tard. En IA, un résultat qu'on ne peut pas reproduire ne vaut presque rien.
  - **Question de vérification :** Q3 dans `QUESTIONS_PEDAGOGIQUES.md`.
- **Points ouverts :** aucun.
- **Suite proposée :** étape 0.4 — `README.md` minimal (objectif, architecture, installation).
- **Statut :** validé et commité.

### Étape 0.4 — README minimal (2026-11-22)

- **Fait :** `README.md` réécrit : présentation, statut, schéma d'architecture (Mermaid, rendu automatiquement par GitHub), tableau MoE applicatif vs neuronal, stack, budget mémoire, installation Windows, configuration de F1 25, lancement des tests, structure, liens vers les fichiers de suivi, limites connues. Rédigé en français, comme le reste du projet.
- **Tests :** liens internes du README vérifiés (fichiers existants) ; `pytest` → 4 passed. Le rendu du schéma Mermaid se vérifie sur GitHub après le push.
- **Concepts IA :**
  - **Entraînement vs inférence.** Un modèle d'IA a deux vies. **L'entraînement** : on lui montre d'énormes quantités de données et on ajuste ses paramètres (Q3) — c'est long, coûteux, fait une seule fois par Meta pour Llama. **L'inférence** : on utilise le modèle figé pour produire une réponse — c'est ce que fera notre projet à chaque message radio. Le README le dit implicitement : « tout tourne en local » ne concerne que l'inférence ; nous n'entraînerons aucun des modèles utilisés (Llama, Whisper).
  - **Documenter un système d'IA.** Un projet d'IA sérieux documente ses choix de modèles, ses contraintes de ressources et ses limites (c'est l'idée des *model cards* publiées avec les modèles). Le tableau du budget mémoire et la section « Limites connues » du README jouent ce rôle ; on y ajoutera les mesures de latence et de RAM/VRAM des phases 4 à 6.
  - **Question de vérification :** Q5 dans `QUESTIONS_PEDAGOGIQUES.md`.
- **Points ouverts :** README en français ; une version anglaise pourra être ajoutée en Phase 7 si le projet doit viser des recruteurs internationaux.
- **Suite proposée :** Phase 1, étape 1.1 — vérifier la spec officielle des paquets UDP F1 25.
- **Statut :** validé et commité.

### Étape 1.1 — Vérification de la spec UDP F1 25 (2026-11-22)

- **Fait :** recherche de la spec officielle et recoupement avec deux bibliothèques communautaires.
  - **Spec officielle EA :** publiée sur le *Game Info Hub* des forums EA (« F1® 25 / 2026 Season Pack UDP Specification », version 9.0 d'après les résultats de recherche). **La page refuse les lectures automatiques (HTTP 403)** : je n'ai pas pu la lire directement.
  - **Recoupement :** `f1-packets` 2026.1.0 (Python, PyPI, dépôt `P403n1x87/f1-packets`, qui embarque un `spec.h` « © 2026 Electronic Arts ») et `@deltazeroproduction/f1-udp-parser` (TypeScript, supporte F1 22 à F1 25).
- **Constats :**
  1. **F1 25 a deux modes UDP**, au choix dans les paramètres du jeu : « F1 25 » et « F1 25 : 2026 Season Pack ». Un nouveau joueur est par défaut sur le mode 2026 Season Pack. Le champ `m_packetFormat` de l'en-tête vaut **2026** dans ce mode ; la valeur 2025 pour l'autre mode n'est **pas encore confirmée** par une source lue.
  2. **L'en-tête (29 octets) a la même structure** dans les sources lues : `m_packetFormat` (uint16), `m_gameYear`, `m_gameMajorVersion`, `m_gameMinorVersion`, `m_packetVersion`, `m_packetId` (uint8), `m_sessionUID` (uint64), `m_sessionTime` (float), `m_frameIdentifier`, `m_overallFrameIdentifier` (uint32), `m_playerCarIndex`, `m_secondaryPlayerCarIndex` (uint8).
  3. **Types de paquets :** IDs 0 à 15 (Motion, Session, LapData, Event, Participants, CarSetups, CarTelemetry, CarStatus, FinalClassification, LobbyInfo, CarDamage, SessionHistory, TyreSets, MotionEx, TimeTrial, LapPositions) ; le mode 2026 ajoute **l'ID 16 `CarTelemetry2`**.
  4. **Les tailles et le contenu des paquets diffèrent selon le mode** (le mode 2026 décrit des tableaux de 24 voitures ; des tailles ont aussi changé entre versions de la spec F1 25, ex. CarDamage 953 → 1041 octets). Les tailles relevées dans les sources communautaires sont incomplètes et ne concordent pas toutes : **à confirmer**.
- **Conclusion :** on ne code pas les structures détaillées sur la foi de sources secondaires. Deux vérifications sont nécessaires :
  - **Spec officielle** téléchargée par le pilote du projet depuis un navigateur (gardée en local, non versionnée : document © EA).
  - **Mesure réelle** : capturer les paquets du jeu et relever, pour chaque ID, `m_packetFormat` et la taille exacte → la vérité terrain.
- **Concepts IA :**
  - **Le schéma des données, un contrat.** Un paquet UDP n'est qu'une suite d'octets ; sa signification vient d'un *schéma* (la spec) qui dit « octets 0-1 = format, octet 6 = ID… ». Si on lit les données avec le mauvais schéma, on n'obtient pas forcément une erreur : on obtient des **nombres faux qui ont l'air vrais**. C'est le cas le plus dangereux en IA : un modèle alimenté par des données mal décodées produit des réponses fausses avec assurance (*garbage in, garbage out*).
  - **Valider à l'entrée.** D'où une règle de base des pipelines de données : vérifier ce qu'on reçoit (format, version, taille) **avant** de l'interpréter, et rejeter ce qui ne correspond pas.
  - **Question de vérification :** Q6 dans `QUESTIONS_PEDAGOGIQUES.md`.
- **Décision :** le pilote du projet joue à **F1 25 de base** (pas au 2026 Season Pack) → on cible le **format 2025** (`m_packetFormat` attendu = 2025). Le parser vérifiera ce champ et rejettera tout autre format.
- **Décision :** l'assistant doit aussi fonctionner en **Contre-la-montre** (*Time Trial*). Conséquences : lire le type de session (paquet Session) et le paquet TimeTrial (ID 14, à confirmer dans la spec 2025) ; le router désactive l'expert Strategy hors course ; l'expert Pace gagne un delta en direct vs meilleur tour perso. Étapes ajoutées au journal (1.4b, 2.2b) et validations élargies (1.8, 2.6).
- **Points ouverts :** confirmer dans les paramètres du jeu que le mode UDP est bien « F1 25 » (d'après la recherche, le mode 2026 peut être sélectionné par défaut) ; spec officielle du format 2025 à récupérer ; tailles réelles à mesurer avec la sonde.
- **Suite proposée :** étape 1.2 — `udp_receiver.py` + décodage **de l'en-tête seul** (identique dans tous les modes) + un petit outil de sonde qui compte les paquets par ID, format et taille, à lancer avec le jeu pour obtenir la vérité terrain.
- **Complément — documents reçus et vérifiés :**
  - **Réglages du jeu** (capture `docs/game screenshot/telemetry settings.png`) : UDP Telemetry On, Broadcast Off, IP `127.0.0.1`, port `20777`, Send Rate 20 Hz, **UDP Format 2025** → conformes.
  - **Spec officielle** dans `docs/specs/` : `Data Output from F1 25 v3.pdf` (31 pages) + `F1 25 Telemetry Output Structures.txt` = **notre référence**. Les deux PDF et le `.txt` « 2026 Season Pack » concernent l'autre mode : gardés pour mémoire, non utilisés.
  - **Vérification des tailles :** les 16 tailles annoncées dans le PDF sont identiques à celles du `.txt`, et je les ai **recalculées à partir des champs des structures** : toutes concordent (un premier écart sur Session venait d'une erreur de saisie de ma part ; recalculé en lisant les champs directement dans le fichier → 753 octets, conforme).
  - **Différences 2025 / 2026 confirmées** par comparaison des deux `.txt` : 22 → 24 voitures, forces G en `float` → `int16` quantifiés, Session 753 → 926 octets, paquet 16 en plus.
  - **Contre-la-montre** : `m_sessionType = 18`, `m_gameMode = 5` ; le paquet TimeTrial (ID 14, 101 octets, 1/s) **existe bien en format 2025** et n'est envoyé qu'en Contre-la-montre ; LapData contient aussi les index des voitures « meilleur tour perso » et « rival ».
  - Synthèse rédigée pour le projet : `docs/SPEC_F1_25.md` (versionnée). Les documents EA (`docs/specs/`) sont exclus de Git (`.gitignore`), car © EA.
  - `pypdf` installé dans le venv pour lire les PDF (outil ponctuel, non ajouté à `requirements.txt`).
- **Statut :** validé et commité.

### Étape 1.2 — Récepteur UDP, validation de l'en-tête et sonde (2026-11-22)

- **Fait :**
  - `telemetry/packet_parser.py` : constantes de la spec (format 2025, en-tête `<HBBBBBQfIIBB` = 29 octets, `PacketId`, tailles des 16 paquets), `parse_header()` et `validate_packet()` — les deux contrôles de Q6 : **format + type** (en-tête) puis **taille exacte** (contenu). Tout paquet non conforme lève `PacketValidationError`.
  - `telemetry/udp_receiver.py` : `UdpReceiver` asynchrone (`asyncio`) qui place les datagrammes bruts dans une file ; si la file est pleine, **le plus ancien est jeté** (compteurs `received` / `dropped`).
  - `tools/udp_probe.py` : sonde en ligne de commande (`python -m tools.udp_probe --duration 30`) qui compte les paquets par (format, type), relève leurs tailles et leur fréquence, et liste les rejets avec leur raison.
  - `tests/conftest.py` : fabrique de faux paquets (`build_packet`) pour tester sans le jeu.
- **Tests :** `pytest` → **32 passed** : en-tête décodé champ par champ, les 16 types acceptés à leur taille exacte, rejet du format 2026, de l'ID 16, des tailles fausses (dont l'ancienne taille CarDamage 953) ; réception réelle de datagrammes en local ; politique « jeter le plus ancien ». **Test de bout en bout** : sonde lancée en ligne de commande + 62 faux paquets envoyés → 60 valides, 2 rejets corrects (« format 2026 non supporté », « CAR_DAMAGE : 953 octets reçus, 1041 attendus »).
- **Concepts IA :**
  - **Flux temps réel vs lot (*streaming* vs *batch*).** En *batch*, on a toutes les données avant de calculer (ex. entraîner un modèle sur 20 courses enregistrées). En *streaming*, les données arrivent en continu et on décide au fil de l'eau : c'est notre cas — environ 20 paquets par seconde **et par type**, soit plus de 100 paquets/s au total.
  - **Producteur, consommateur et file d'attente.** Le jeu produit à un rythme qu'on ne contrôle pas ; notre analyse consomme à son propre rythme. La file absorbe les à-coups. Si le consommateur est trop lent (plus tard : le LLM, qui met parfois plus d'une seconde à répondre), la file se remplit : il faut une **politique de débordement** (*backpressure*). Nous gardons les données **les plus fraîches** : un ingénieur de course doit parler de la situation **actuelle**, pas d'il y a 3 secondes.
  - **Valider à l'ingestion** : application directe de Q6 — chaque paquet est contrôlé dès son arrivée, avant d'atteindre les experts et le LLM.
  - **Question de vérification :** Q7 dans `QUESTIONS_PEDAGOGIQUES.md`.
- **Points ouverts :** les paquets **Event** et **FinalClassification** (envoyés une seule fois) ne devront jamais être jetés par la file (conclusion de Q7, à intégrer quand on branchera l'analyse, étape 1.4 au plus tard) ; l'enregistreur (1.6) ne devra rien jeter. **Test avec le vrai jeu** à faire par le pilote du projet (voir ci-dessous) → il confirmera la vérité terrain (format, tailles et fréquences réels). Les accents peuvent s'afficher mal si la sortie de la sonde est redirigée vers un fichier ; en console, pas de problème.
- **Suite proposée :** étape 1.3 — décodage du contenu des paquets utiles (LapData, CarTelemetry, CarStatus, CarDamage, Session, TimeTrial) + tests.
- **Test en jeu (Contre-la-montre, 120 s) — vérité terrain :**
  - **12 580 paquets reçus, 12 580 valides, 0 jeté** ; format 2025, année 25, jeu en version 1.27, voiture joueur n°0.
  - **Les 14 types reçus ont exactement la taille de la spec.** Absents (normal en Contre-la-montre) : FinalClassification (8) et LobbyInfo (9). Le paquet **TimeTrial (14) est bien présent**.
  - **Fréquences cohérentes avec la spec** une fois rapportées au temps où chaque paquet circulait : 1 514 paquets « au rythme du menu » à 20 Hz ≈ 76 s de roulage ; Session (202 à 2/s) et TimeTrial (102 à 1/s) ≈ 100 s de session. La colonne `/s` de la première version de la sonde divisait par toute la durée d'écoute (120 s, menus compris) → **corrigée** : la sonde affiche maintenant, par type, la fenêtre active (`actif(s)`) et la fréquence mesurée sur cette fenêtre (+ 1 test, **33 passed**).
  - **Event : ~3 par seconde**, bien plus que prévu pour des « événements rares ». Probablement des événements `BUTN` (état des boutons, envoyés à chaque appui) : à vérifier et à filtrer à l'étape 1.3.
  - **Volume de données** : à 20 Hz, ≈ 160 Ko/s, soit **≈ 575 Mo par heure** de paquets bruts → à prendre en compte pour l'enregistreur (1.6) : ne garder que les types utiles et/ou compresser.
- **Statut :** validé et commité.
