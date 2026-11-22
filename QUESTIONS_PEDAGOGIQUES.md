# Questions pédagogiques — AI Race Engineer (F1 25)

Les questions de vérification posées pendant les bilans, avec ta réponse et la correction.
Elles sont ajoutées dans l'ordre chronologique, sous l'étape où elles ont été posées.

**Sommaire**

| # | Étape | Thème |
|---|---|---|
| Q1 | Discussion (avant 0.2) | MoE neuronal : mémoire vs paramètres actifs |
| Q2 | 0.2 | Poids en RAM vs VRAM : vitesse de génération |

---

## Q1 — Pourquoi Mixtral ne tient-il pas là où tient un modèle dense de 13 B ?

*Contexte : discussion sur le coût d'un MoE neuronal (2026-11-22).*

### Question

Pourquoi Mixtral, qui n'utilise que ~13 B paramètres par token, ne tient-il pas dans une machine capable de faire tourner un modèle dense de 13 B ?

### Ta réponse

> Parce que pour chaque milliard de paramètres il faut nb paramètres × 2 en mémoire, ce qui n'est pas disponible sur la machine.

### Correction — à moitié juste

**Ce qui est juste :** en fp16, un paramètre occupe 2 octets, donc 1 milliard de paramètres ≈ 2 Go. C'est la bonne façon d'estimer la mémoire.

**Ce qui manque :** la question compare deux modèles qui font le **même calcul** par token (~13 B paramètres actifs). Si la règle s'appliquait aux paramètres actifs, les deux auraient besoin de la même mémoire. L'élément clé :

> **La mémoire dépend des paramètres totaux, pas des paramètres actifs.**

| | Paramètres à charger | Mémoire Q4 (~0,6 octet/paramètre) |
|---|---|---|
| Modèle dense 13 B | 13 B (tous actifs) | ~8 Go ✅ |
| Mixtral 8x7B | **47 B** (les 8 experts) | ~26 Go ❌ |

**Pourquoi charger les 8 experts si 2 seulement calculent ?** Le router choisit ses experts **à chaque token et à chaque couche** (Mixtral a 32 couches). Dans « Box this lap, tyres are gone », « Box » peut aller aux experts 2 et 5, et « tyres » aux experts 1 et 7. Le choix change à chaque mot et ne se prévoit pas : tous les experts doivent être prêts en mémoire. Les laisser en RAM et les charger à la demande (*offloading*) est possible, mais très lent.

### À retenir

- **Mémoire = paramètres totaux. Vitesse = paramètres actifs.**
- Le MoE rend le modèle **rapide**, pas **léger**.

---

## Q2 — Pourquoi des poids en RAM ralentissent-ils autant la génération ?

*Contexte : étape 0.2 — Python 3.12, venv, vérification du GPU (2026-11-22).*

### Question

Pourquoi un modèle dont une partie des poids est en RAM (et pas en VRAM) génère-t-il ses tokens beaucoup plus lentement, alors que le GPU est toujours aussi puissant ?

### Ta réponse

> Même si le GPU est puissant, il faut au modèle le temps de charger les poids depuis la RAM, ce qui rajoute cette lenteur sur la génération des tokens.

### Correction — juste, avec une précision importante

Ce chargement ne se fait pas une seule fois au démarrage : il se répète **à chaque token**.

Pour générer un token, le modèle passe par **toutes ses couches**, donc lit **tous ses poids**. Pour Llama 3.1 8B quantifié, ça fait ~5 Go lus pour un seul token. Une réponse de 30 tokens représente ~150 Go de lecture mémoire.

> **La génération d'un LLM est limitée par la mémoire (*memory-bound*), pas par le calcul.** Le GPU passe la plupart de son temps à attendre les poids.

Ordre de grandeur sur la RTX 5060 (débits approximatifs) :

| Où sont les poids | Débit de lecture | Temps pour lire 5 Go | Tokens/s maximum |
|---|---|---|---|
| Tout en VRAM | ~380 Go/s | ~13 ms | **~75** |
| 3 Go en VRAM + 2 Go en RAM | VRAM ~380 + RAM ~60 Go/s | 8 + 33 = ~41 ms | **~24** |

Avec seulement 40 % du modèle en RAM, la génération devient **3 fois plus lente** : la RAM est ~6 fois plus lente que la VRAM, et ces 2 Go dominent le temps total.

En pratique, Ollama ne recopie pas les couches vers le GPU à chaque token : il **calcule ces couches sur le CPU**, là où sont leurs poids. Le goulot reste la vitesse de lecture de la RAM.

### À retenir

- **Vitesse maximale de génération ≈ débit mémoire ÷ taille du modèle.**
- C'est pour ça que la **quantification** accélère autant : un modèle 2× plus petit = 2× moins d'octets à lire par token (à mesurer en Phase 5).
