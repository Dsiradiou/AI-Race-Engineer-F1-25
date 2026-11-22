# Journal de bord — AI Race Engineer (F1 25)

Liste des étapes du projet, dans l'ordre. On avance **une étape à la fois** :
implémentation → tests si besoin → récapitulatif (dans `RECAPITULATIFS.md`) →
**feu vert du pilote du projet** → commit → étape suivante.

Référence complète : `AI_RACE_ENGINEER.md`.

**Légende :** `[ ]` à faire · `[~]` en cours · `[x]` terminé (validé + commité)

## Objectif pédagogique

Au-delà du logiciel, ce projet sert à **comprendre comment fonctionne un modèle d'IA**,
et en particulier un LLM : sa structure, le traitement des données, le fonctionnement
du Mixture of Experts, et plus largement les bases de l'IA appliquée.
Chaque phase liste les **concepts IA** à comprendre ; ils sont expliqués avec des exemples
tirés du projet lors des bilans (section « Concepts IA » de `RECAPITULATIFS.md`).

---

## Étape 0 — Mise en place

- [x] 0.1 Créer le dépôt GitHub et le lier au dossier local
- [x] 0.2 Choisir et installer la version de Python (3.12 recommandé), créer le venv
- [ ] 0.3 Créer l'arborescence du projet (`__init__.py`, `requirements.txt`, `.gitignore`, `.env.example`, `config.py`)
- [ ] 0.4 Écrire un `README.md` minimal (objectif, architecture, installation)

> **Concepts IA :** pourquoi l'IA tourne sur GPU (calcul matriciel parallèle), CPU vs GPU, VRAM vs RAM, CUDA, gestion des dépendances.

## Phase 1 — Télémétrie UDP

- [ ] 1.1 Vérifier la spec officielle des paquets UDP F1 25 (en-tête, IDs de paquets, tailles)
- [ ] 1.2 `telemetry/udp_receiver.py` — réception des paquets UDP (port 20777)
- [ ] 1.3 `telemetry/packet_parser.py` — décodage binaire (`struct`) des paquets utiles + tests
- [ ] 1.4 `telemetry/snapshot.py` — `TelemetrySnapshot` (état courant de la course)
- [ ] 1.5 Affichage console en temps réel (vitesse, tour, secteur, pneus, gap)
- [ ] 1.6 `tools/record_session.py` — enregistrement des paquets bruts
- [ ] 1.7 `tools/replay_session.py` — rejeu d'une session enregistrée
- [ ] 1.8 **Validation :** session réelle F1 25 affichée en console + rejeu identique

> **Concepts IA :** pipeline de données (données brutes → données structurées), format binaire et sérialisation, flux temps réel vs batch, extraction de caractéristiques (*features*), qualité des données et reproductibilité (rejeu).

## Phase 2 — Analyzer MoE

- [ ] 2.1 `analyzer/router.py` — `TelemetryRouter` + tests
- [ ] 2.2 `analyzer/expert_pace.py` + tests
- [ ] 2.3 `analyzer/expert_strategy.py` + tests
- [ ] 2.4 `analyzer/expert_meca.py` + tests
- [ ] 2.5 `analyzer/aggregator.py` — fusion et priorisation des alertes + tests
- [ ] 2.6 **Validation :** `pytest` vert + alertes visibles en console sur une session rejouée

> **Concepts IA :** Mixture of Experts — router (*gating*), experts, activation top-k, paramètres totaux vs actifs ; **MoE applicatif (ce projet) vs MoE neuronal (Mixtral)** ; systèmes à règles vs modèles appris ; tests d'un système de décision.

## Phase 3 — Sortie vocale (TTS)

- [ ] 3.1 `voice/voice_out.py` — `VoiceOut.speak(text)` avec pyttsx3
- [ ] 3.2 File de priorité + cooldown (15 s non critique) + tests
- [ ] 3.3 **Validation :** message audible dans le casque, priorité et cooldown respectés

> **Concepts IA :** synthèse vocale — approches à règles (pyttsx3) vs neuronales (modèle acoustique + vocodeur), compromis qualité / latence / ressources.

## Phase 4 — Entrée vocale (STT)

- [ ] 4.1 Vérifier PyTorch + CUDA sur la RTX 5060, installer `ffmpeg`
- [ ] 4.2 `voice/voice_in.py` — capture micro + VAD (`webrtcvad`)
- [ ] 4.3 Transcription Whisper `small` sur GPU
- [ ] 4.4 **Validation :** questions transcrites en < 2 s + mesure RAM/VRAM

> **Concepts IA :** du son au texte — échantillonnage audio, spectrogramme log-Mel, architecture **Transformer encodeur-décodeur** de Whisper, tailles de modèles (tiny → large), précision fp16/int8, inférence GPU, VAD.

## Phase 5 — LLM local (Ollama)

- [ ] 5.1 Installer Ollama, vérifier et télécharger le modèle (`llama3.1:8b`)
- [ ] 5.2 `ai/llm_client.py` — client REST Ollama
- [ ] 5.3 `ai/prompts.py` — system prompt par expert + injection du snapshot
- [ ] 5.4 Nettoyage de la sortie pour le TTS
- [ ] 5.5 **Validation :** snapshot fictif → recommandation cohérente en 1-2 phrases + latence mesurée

> **Concepts IA (cœur du projet) :** tokenisation, embeddings, architecture **Transformer décodeur** (attention, couches, têtes), génération autorégressive token par token, température / top-p, fenêtre de contexte, system prompt et *prompt engineering*, **quantification** (pourquoi un 8B tient dans ~5 Go), KV cache, latence (temps du 1er token, tokens/s), hallucinations.

## Phase 6 — Intégration complète

- [ ] 6.1 `main.py` — orchestrateur `asyncio` (UDP + STT + LLM + TTS en parallèle)
- [ ] 6.2 Gestion des priorités et des conflits entre alertes et questions
- [ ] 6.3 Logs JSON de session (`logs/`)
- [ ] 6.4 **Validation :** course de 5 tours, ≥ 1 alerte pertinente + 1 question/réponse, budget RAM/VRAM respecté

> **Concepts IA :** système d'IA de bout en bout, latence cumulée d'un pipeline (STT + LLM + TTS), concurrence, inférence sous contrainte mémoire, observabilité (logs).

## Phase 7 — Polish

- [ ] 7.1 Ajuster seuils, prompts, voix et cooldown
- [ ] 7.2 Commande vocale « Engineer, standby »
- [ ] 7.3 (Optionnel) Mini dashboard console
- [ ] 7.4 (Optionnel) Résumé post-course généré par le LLM
- [ ] 7.5 README final : schéma, métriques (latence, RAM/VRAM), démo vidéo
- [ ] 7.6 **Validation :** un Grand Prix complet avec des alertes pertinentes et bien placées

> **Concepts IA :** évaluation d'un système d'IA (métriques, retours utilisateur), réglage des hyperparamètres, résumé de texte par LLM, limites et biais.

## Bonus pédagogique (optionnel, à valider le moment venu)

- [ ] B.1 Mini-MoE **neuronal** jouet en PyTorch (router appris + experts + top-k) pour comparer concrètement avec le MoE applicatif du projet
- [ ] B.2 Explorer le LLM de l'intérieur : tokenisation d'un message radio, probabilités des tokens suivants, effet de la température
