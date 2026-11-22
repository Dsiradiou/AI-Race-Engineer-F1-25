# AI Race Engineer — F1 25

> Document de cadrage à donner à Claude Code au démarrage du projet.
> Lis-le en entier, puis travaille **phase par phase** selon la méthode décrite en section 9.

---

## 1. Objectif

Construire un **ingénieur de course IA temps réel** pour **F1 25** :

- Il lit la **télémétrie UDP** du jeu en direct.
- Il détecte les situations de course importantes (perte de temps, mauvaise entrée/sortie de virage, usure pneus, dégâts…).
- Il **parle au pilote** (audio dans le casque) avec un style radio F1, court et direct.
- Il **écoute le pilote** : on peut lui poser des questions à la voix ("How are my tires?", "What's my gap to P2?").
- Il utilise une architecture **Mixture of Experts (MoE)** : un *router* active un ou plusieurs *experts* spécialisés, comme dans une vraie équipe F1 (ingénieur de piste, stratège, ingénieur mécanique).
- Tout tourne **en local** sur la machine du pilote (pas de dépendance cloud obligatoire).

Contexte : projet personnel d'un étudiant en 3e année du cycle ingénieur (spécialisation IA & Big Data). Il doit être **propre, documenté, testé et présentable sur GitHub** (portfolio / CV / PFE).

---

## 2. Ressources matérielles et contraintes

| Composant | Détail |
|---|---|
| Machine | HP Omen 16 (Windows) |
| CPU | AMD Ryzen 9 série HX |
| GPU | NVIDIA RTX 5060 (≈ 8 Go VRAM) |
| RAM | **16 Go** (≈ 15,1 Go utilisables) |
| Jeu | F1 25 sur la même machine (UDP en `127.0.0.1`) |
| Temps de travail | ≈ 8 h / semaine |

**Budget mémoire à respecter** (tout tourne en même temps) :

```
F1 25            ~4-6 Go
Python (script)  ~0,5 Go
Ollama + LLM 8B  ~6-8 Go (VRAM + RAM)
Whisper small    ~1-2 Go
-------------------------
Total visé       ≤ 14 Go
```

Conséquences :
- Pas de modèle > 8B en local (Mixtral 8x7B est exclu : ~26 Go).
- Un seul modèle LLM chargé à la fois ; les experts partagent le même modèle avec des **system prompts différents**.
- Surveiller la RAM/VRAM à chaque phase (`nvidia-smi`, gestionnaire des tâches).

---

## 3. Architecture MoE

> Précision honnête : il ne s'agit pas d'un MoE neuronal (comme Mixtral) mais d'un **MoE applicatif** : un router à règles + des experts spécialisés (logique de détection + prompt dédié). C'est le même principe de spécialisation, adapté au cas métier.

```
F1 25 (UDP, port 20777)
        │
        ▼
  [ UDP Receiver + Parser ]  → TelemetrySnapshot (état courant de la course)
        │
        ▼
  [ TelemetryRouter ]        → décide quels experts activer (0, 1 ou plusieurs)
        │
 ┌──────┼───────────────┐
 ▼      ▼               ▼
Pace   Strategy        Meca
 │      │               │
 └──────┼───────────────┘
        ▼
  [ Aggregator ]   → fusionne / priorise les alertes
        ▼
  [ LLM local (Ollama) ]  → formule le message radio (system prompt par expert)
        ▼
  [ TTS ]  → audio dans le casque

Micro → [ VAD + Whisper STT ] → question du pilote → LLM → TTS
```

### Experts

| Expert | Déclencheurs (exemples) | Message type |
|---|---|---|
| **Pace** (ingénieur de piste) | delta tour > 0,5 s, perte de temps par secteur, freinage tardif, sous-virage en sortie, survirage, lock-up | "You're losing 3 tenths in sector 2, brake 10 metres later into turn 4." |
| **Strategy** (stratège) | âge des pneus, gap devant/derrière, fenêtre undercut/overcut, tour de course, carburant | "Undercut window opens in 2 laps, car behind is within 1.8." |
| **Meca** (ingénieur mécanique) | dégâts (aileron, etc.), températures, ERS, usure freins | "Front wing at 73 percent, avoid the kerbs." |
| **Coach vocal** | question du pilote | Répond à la question avec le snapshot courant. |

### Deux modes de déclenchement

1. **Automatique (événementiel)** : le router détecte une situation → l'expert parle.
2. **À la demande** : le pilote pose une question à la voix.

### Règles de parole (anti-bavardage)

- Priorité : **critique > performance > info > réponse à une question** (une réponse à une question du pilote ne doit jamais être perdue).
- Ne jamais couper un message en cours pour une alerte non critique.
- Cooldown ≈ **15 s** entre deux messages non critiques.
- Messages **courts** : 1-2 phrases max, style radio F1, en **anglais**.
- Commande vocale prévue : "Engineer, standby" pour couper les alertes automatiques.

---

## 4. Stack technique

| Couche | Outil | Remarque |
|---|---|---|
| Langage | Python 3.11+ (venv) | `asyncio` pour la concurrence |
| Télémétrie | `socket` + `struct` | parser écrit/validé sur la **spec UDP F1 25** |
| LLM local | **Ollama** + `llama3.1:8b` | API REST `http://localhost:11434` ; alternatives : Mistral 7B, Qwen 2.5 7B |
| STT | `openai-whisper` (modèle `small`) ou `faster-whisper` | GPU CUDA ; `ffmpeg` requis |
| VAD | `webrtcvad` | détection de silence |
| TTS | `pyttsx3` (offline, de base) → ElevenLabs (option, voix naturelle) ou TTS local alternatif | commencer simple |
| Audio | `sounddevice`, `pygame` | capture micro / lecture |
| Tests | `pytest` | indispensable pour router + experts |

Points de vigilance à **vérifier avant de coder** (ne pas supposer) :
- Le format exact des paquets **F1 25** (la spec officielle EA/Codemasters ; les specs F1 23/24 diffèrent sur certains champs). Une bibliothèque tierce de parsing F1 peut ne pas supporter F1 25 : tester d'abord, sinon écrire le parser soi-même.
- La compatibilité PyTorch/CUDA avec la RTX 5060 (architecture récente) : installer une version de PyTorch récente avec support CUDA adapté.
- Les noms/valeurs des champs télémétrie (unités, échelles) en comparant avec les données réelles du jeu.

---

## 5. Structure du projet

```
ai_race_engineer/
├── main.py                 # orchestrateur asyncio
├── config.py               # seuils, ports, modèles, clés API (via .env)
├── .env.example            # variables d'environnement (ne jamais commit .env)
├── requirements.txt
├── README.md
├── telemetry/
│   ├── udp_receiver.py     # réception des paquets UDP
│   ├── packet_parser.py    # décodage binaire (struct)
│   └── snapshot.py         # TelemetrySnapshot (état courant)
├── analyzer/
│   ├── router.py           # TelemetryRouter
│   ├── expert_pace.py
│   ├── expert_strategy.py
│   ├── expert_meca.py
│   └── aggregator.py
├── voice/
│   ├── voice_in.py         # micro + VAD + Whisper
│   └── voice_out.py        # TTS + file de priorité + cooldown
├── ai/
│   ├── llm_client.py       # client Ollama
│   └── prompts.py          # system prompts par expert
├── tests/
│   ├── fixtures/           # snapshots fictifs, paquets enregistrés
│   └── test_*.py
├── tools/
│   ├── record_session.py   # enregistre les paquets UDP bruts pour rejouer sans le jeu
│   └── replay_session.py   # rejoue une session enregistrée
└── logs/
    └── session_YYYYMMDD.json
```

Astuce clé : prévoir dès la Phase 1 un **enregistreur/rejoueur de paquets UDP**. Ça permet de développer et tester analyzer + IA **sans lancer le jeu** à chaque fois.

---

## 6. Seuils par défaut (`config.py`)

| Paramètre | Valeur par défaut |
|---|---|
| Delta tour (alerte pace) | > 0,5 s vs meilleur tour perso |
| Âge pneus (avertissement) | > 20 tours |
| Gap danger derrière | < 2,0 s |
| Dégâts | > 0 % |
| Usure pneus (paliers) | 50 % / 75 % / critique |
| Cooldown messages non critiques | 15 s |

Tous les seuils sont **configurables** et à affiner selon le style de conduite (Phase 7).

---

## 7. Phases, livrables et tests de validation

Cadence visée : ≈ 8 h/semaine → **~2 mois** au total.

### Phase 1 — Télémétrie UDP (≈ 1 semaine)
- Activer l'UDP dans F1 25 (Paramètres → Télémétrie → UDP **On**, port **20777**, IP `127.0.0.1`).
- Écrire `udp_receiver.py` + `packet_parser.py` + `TelemetrySnapshot`.
- Écrire `tools/record_session.py` et `replay_session.py`.
- **Test de validation :** lancer une session dans F1 25 → vitesse, tour, secteur, pneus, gap défilent en console en temps réel ; une session enregistrée se rejoue à l'identique.

### Phase 2 — Analyzer MoE (≈ 1-2 semaines)
- `TelemetryRouter`, `ExpertPace`, `ExpertStrategy`, `ExpertMeca`, `Aggregator`.
- **Test de validation :** tests `pytest` avec faux snapshots — chaque expert se déclenche (ou non) comme prévu ; le router peut activer plusieurs experts ; alertes visibles en console sur une session rejouée.

### Phase 3 — Sortie vocale TTS (≈ 3-4 jours)
- `VoiceOut.speak(text)`, file de priorité, cooldown.
- **Test de validation :** `speak("Box this lap, tyres are gone")` s'entend dans le casque ; deux messages rapprochés respectent priorité et cooldown.

### Phase 4 — Entrée vocale STT (≈ 4-5 jours)
- `VoiceIn.listen() -> str` avec VAD, Whisper `small` sur GPU.
- **Test de validation :** "How are my tyres?" / "What's my gap?" transcrits en console en < 2 s.

### Phase 5 — LLM local (Ollama) (≈ 1 semaine)
- Installer Ollama, `ollama pull llama3.1:8b`, `llm_client.py`, `prompts.py` (un system prompt par expert).
- Injecter le snapshot dans le prompt, nettoyer la sortie pour le TTS.
- **Test de validation :** snapshot fictif (tour 28, pneus 78 % d'usure, gap 1,8 s derrière) → recommandation de pit cohérente en 1-2 phrases, latence mesurée.

### Phase 6 — Intégration complète (≈ 1-2 semaines)
- `main.py` : `asyncio` pour UDP + STT + LLM + TTS en parallèle.
- Priorités, gestion des conflits, logs JSON de session.
- **Test de validation :** course de 5 tours avec le système actif → au moins une alerte automatique pertinente + une question vocale avec réponse cohérente ; budget RAM/VRAM respecté.

### Phase 7 — Polish (≈ 1 semaine)
- Ajuster seuils, prompts, voix, cooldown ; commande "Engineer, standby".
- Optionnel : mini dashboard console, résumé post-course généré par le LLM depuis les logs.
- **Critère de succès :** un Grand Prix complet sans avoir envie de couper les alertes (pertinentes, bien placées, bon ton).

---

## 8. Conventions de code

- Python 3.11+, typage (`type hints`) et docstrings courtes sur les fonctions publiques.
- Un module = une responsabilité ; pas de logique métier dans `main.py`.
- Aucun secret en dur : clés API dans `.env` (jamais commité) + `.env.example`.
- Logs via `logging` (pas de `print` en dehors du debug console de la Phase 1).
- Tests `pytest` pour le parser, le router et chaque expert (fixtures dans `tests/fixtures/`).
- Commits petits et fréquents, messages clairs (`feat: ...`, `fix: ...`, `test: ...`), un commit par étape logique.
- Git + GitHub dès le départ ; `README.md` qui évolue avec le projet (schéma d'architecture, installation, démo).

---

## 9. Méthode de travail avec Claude Code

1. **Une phase à la fois.** Ne pas anticiper les phases suivantes ; ne pas créer de fichiers vides "pour plus tard".
2. **Au début de chaque phase :** reformuler en quelques lignes ce qui va être construit et le test de validation visé, puis demander confirmation si un choix de conception est ambigu.
3. **Petits pas :** implémenter par incréments testables (un module, son test, un commit).
4. **Tester avant d'avancer :** une phase n'est terminée que lorsque son **test de validation** (section 7) passe. Lancer réellement le code et les tests, ne pas supposer que ça marche.
5. **Vérifier plutôt que supposer** sur tout ce qui est externe : format des paquets F1 25, versions de bibliothèques, compatibilité CUDA/RTX 5060, noms de modèles Ollama. En cas de doute, le dire et proposer comment le vérifier.
6. **Toujours rejouer avant de jouer :** développer l'analyzer et l'IA sur des sessions enregistrées (`replay_session.py`), et réserver les tests en jeu réel aux fins de phase.
7. **Surveiller les ressources :** après les phases 4, 5 et 6, mesurer RAM/VRAM et latence (STT, LLM, TTS) et les noter dans le README.
8. **Rester simple d'abord :** TTS basique avant ElevenLabs, règles avant tout raffinement ; n'ajouter de la complexité que si un test ou une mesure le justifie.
9. **Documenter au fil de l'eau :** à la fin de chaque phase, mettre à jour le README (ce qui marche, comment lancer, limites connues).
10. **Compte rendu de fin de phase :** 3-5 lignes — ce qui est fait, résultat du test de validation, points ouverts, proposition pour la phase suivante.

---

## 10. Pour démarrer (première session Claude Code)

1. Créer le dossier du projet, initialiser Git, créer l'environnement virtuel Python 3.11+.
2. Générer l'arborescence de la section 5 (dossiers + `__init__.py`, `requirements.txt`, `.gitignore`, `.env.example`, `README.md` minimal).
3. Démarrer la **Phase 1** : `udp_receiver.py`, `packet_parser.py`, `snapshot.py`, puis l'enregistreur/rejoueur UDP.
4. Valider avec une vraie session F1 25, puis commit.

---

## 11. Valeur du projet (portfolio / CV)

- Système **temps réel** complet : flux UDP → analyse → LLM local → voix, bidirectionnel.
- Architecture **MoE applicative** documentée et testée, adaptée à un cas métier concret.
- **Inférence locale** sous contrainte mémoire (16 Go) : choix de modèles, mesure de latence et de ressources.
- Livrables à soigner pour la présentation : README avec schéma d'architecture, courte vidéo de démo, métriques (latence STT/LLM/TTS, usage RAM/VRAM), repo GitHub propre.

Ligne CV suggérée :
> **AI Race Engineer** — Système temps réel d'analyse de télémétrie F1 25 avec architecture Mixture of Experts, inférence LLM locale (Llama 3.1 8B via Ollama) et interface vocale bidirectionnelle (Whisper STT + TTS).
