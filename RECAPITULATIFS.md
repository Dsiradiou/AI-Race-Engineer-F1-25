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
- **Concepts IA :** explication des notions abordées, illustrée par ce qu'on vient de construire (+ une question de vérification pour tester la compréhension).
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
