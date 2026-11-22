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
