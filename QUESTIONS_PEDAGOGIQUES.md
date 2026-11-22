# Questions pédagogiques — AI Race Engineer (F1 25)

Les questions de vérification posées pendant les bilans, avec ta réponse et la correction.
Elles sont ajoutées dans l'ordre chronologique, sous l'étape où elles ont été posées.

**Sommaire**

| # | Étape | Thème |
|---|---|---|
| Q1 | Discussion (avant 0.2) | MoE neuronal : mémoire vs paramètres actifs |
| Q2 | 0.2 | Poids en RAM vs VRAM : vitesse de génération |
| Q3 | 0.3 | Paramètres vs hyperparamètres |
| Q4 | 0.3 | Paramètres et hyperparamètres d'un MoE |
| Q5 | 0.4 | Entraînement vs inférence : coût mémoire |

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

---

## Q3 — Paramètre ou hyperparamètre ?

*Contexte : étape 0.3 — `config.py` et les seuils des experts (2026-11-22).*

### Question

Quelle est la différence entre un **paramètre** et un **hyperparamètre** d'un modèle ?
Imaginons qu'on remplace un jour la règle « alerte si delta > 0,5 s » par un petit modèle entraîné sur nos sessions enregistrées pour détecter les pertes de temps : le seuil 0,5 s deviendrait-il un paramètre ou un hyperparamètre ? Et le nombre d'époques d'entraînement (le nombre de passes sur les données), lui ?

### Ta réponse

> Je ne connais pas la réponse, peux-tu me l'expliquer ?

### Explication

**Le test pour les distinguer :** est-ce que l'**entraînement** modifie cette valeur en regardant les données ?
- **Oui → paramètre.** Le modèle l'apprend.
- **Non, c'est moi qui la choisis avant l'entraînement → hyperparamètre.** Elle règle *comment* le modèle apprend ou *à quoi il ressemble*.

**Réponse à la question :**
- **Le seuil 0,5 s → deviendrait un paramètre.** On ne l'écrirait plus à la main : le modèle le trouverait à partir des données.
- **Le nombre d'époques → hyperparamètre.** On le fixe avant l'entraînement ; l'entraînement ne le modifie pas.

**Exemple concret avec nos sessions :**

1. On enregistre 20 courses et on étiquette chaque tour : « perte de temps significative » (1) ou « normal » (0), avec son delta.
2. On entraîne une **régression logistique** à une entrée : `probabilité_alerte = sigmoïde(w × delta + b)`.
3. `w` et `b` démarrent au hasard. À chaque passe sur les données (une **époque**), l'algorithme compare ses prédictions aux étiquettes et ajuste `w` et `b` pour se tromper un peu moins.
4. À la fin, l'alerte se déclenche quand `w × delta + b > 0`, c'est-à-dire quand `delta > −b / w`. Si l'entraînement donne `w = 10` et `b = −4,2`, le seuil appris est **0,42 s** : il vient des données, pas d'une intuition.

Dans cet exemple :

| Valeur | Type | Qui la fixe ? |
|---|---|---|
| `w`, `b` (donc le seuil 0,42 s) | **Paramètres** | L'entraînement, à partir des données |
| Nombre d'époques (ex. 100) | Hyperparamètre | Moi, avant l'entraînement |
| Taux d'apprentissage (taille de chaque correction de `w` et `b`) | Hyperparamètre | Moi |
| Choix d'une régression logistique plutôt qu'un réseau de neurones | Hyperparamètre (architecture) | Moi |

**Le même principe dans Llama 3.1 8B :**
- **Paramètres :** les ~8 milliards de poids, tous appris pendant l'entraînement. C'est ce que contient le fichier de ~5 Go téléchargé par Ollama.
- **Hyperparamètres d'architecture**, fixés par les chercheurs avant l'entraînement : 32 couches, vecteurs de taille 4096, 32 têtes d'attention, vocabulaire de ~128 000 tokens.
- **Réglages d'inférence**, choisis par nous à l'utilisation : température, longueur maximale de la réponse. Ils ne changent pas le modèle, seulement la façon dont on s'en sert.

**Comment choisit-on les hyperparamètres ?** Par essais : on entraîne avec plusieurs valeurs (50, 100, 200 époques…) et on garde celle qui donne les meilleurs résultats sur des données **mises de côté** (jeu de validation), jamais vues pendant l'entraînement. Trop peu d'époques : le modèle n'a pas assez appris (**sous-apprentissage**). Trop : il apprend par cœur les données d'entraînement et se trompe sur des courses nouvelles (**surapprentissage**).

### À retenir

- **Paramètre = appris à partir des données. Hyperparamètre = choisi par l'humain avant ou autour de l'apprentissage.**
- Aujourd'hui, nos seuils sont réglés à la main : dans un modèle entraîné, ils deviendraient des paramètres.
- Les hyperparamètres se choisissent en testant sur des données de validation.

---

## Q4 — Paramètres et hyperparamètres d'un MoE

*Contexte : étape 0.3, question de vérification après l'explication de Q3 (2026-11-22).*

### Question

Dans Mixtral 8x7B, classe ces trois éléments en **paramètre** ou **hyperparamètre**, en appliquant le test de Q3 :
1. le nombre d'experts activés par token (top-2) ;
2. les poids du router (la petite couche qui décide vers quels experts envoyer chaque token) ;
3. le nombre total d'experts (8).

### Ta réponse

> 1 hyperparamètre, 2 paramètre, 3 hyperparamètre.

### Correction — tout juste ✅

1. **Top-2 → hyperparamètre.** Les chercheurs ont choisi « 2 experts par token » avant l'entraînement ; l'entraînement ne le change jamais. C'est un compromis : plus de k = plus de calcul et (un peu) plus de qualité.
2. **Poids du router → paramètres.** Le router est une petite couche linéaire (une matrice de taille 4096 × 8 par couche) dont les poids sont appris **en même temps** que les experts. C'est ce qui rend le MoE neuronal « intelligent » : personne n'écrit de règle, le router apprend tout seul quel expert est bon pour quel type de token.
3. **8 experts → hyperparamètre (d'architecture).** Fixé avant l'entraînement, comme le nombre de couches.

**Pour aller un cran plus loin — un troisième type de valeur :** le **choix** des experts pour un token donné (« *tyres* → experts 1 et 7 ») n'est ni un paramètre ni un hyperparamètre. C'est une **activation** : un résultat calculé à l'exécution, à partir des paramètres (poids du router) et de l'entrée (le token). Il change à chaque token et n'est stocké nulle part.

**Comparaison avec notre MoE applicatif :** chez nous, le router n'a **aucun** paramètre appris : ses règles et ses seuils sont écrits à la main dans le code et dans `config.py`. C'est la différence fondamentale entre les deux approches.

### À retenir

| Type | Fixé par | Exemple Mixtral | Exemple dans notre projet |
|---|---|---|---|
| **Hyperparamètre** | L'humain, avant | top-2, 8 experts, 32 couches | modèle Ollama, cooldown 15 s |
| **Paramètre** | L'entraînement | poids du router et des experts | *(aucun : tout est réglé à la main)* |
| **Activation** | Le calcul, à chaque entrée | experts choisis pour « *tyres* » | experts activés pour un snapshot |

---

## Q5 — Pourquoi peut-on utiliser Llama 8B, mais pas l'entraîner ?

*Contexte : étape 0.4 — README, entraînement vs inférence (2026-11-22).*

### Question

En inférence, Llama 3.1 8B quantifié tient dans ~5 Go de VRAM sur ta machine. Pourtant, il serait impossible de l'**entraîner** (même seulement le réentraîner un peu) sur cette même machine. D'après toi, qu'est-ce que l'entraînement doit garder en mémoire **en plus** des poids eux-mêmes ? (Indice : repense à l'exemple de Q3 — à chaque époque, l'algorithme « ajuste `w` et `b` pour se tromper un peu moins ». Que lui faut-il pour savoir dans quel sens les ajuster ?)

### Ta réponse

> Il lui faudrait retenir aussi les paramètres et hyperparamètres.

### Correction — pas tout à fait

- **Les paramètres**, ce sont justement les poids : ils sont déjà comptés dans les ~5 Go. Ce n'est pas un « en plus ».
- **Les hyperparamètres** sont une poignée de nombres (nombre d'époques, taux d'apprentissage…) : quelques octets, négligeable.

Ce qui manque, c'est la réponse à l'indice : **pour savoir dans quel sens ajuster `w`, il faut un gradient.**

**1. Les gradients — un par paramètre.**
Dans l'exemple de Q3, le gradient de `w` répond à la question : « si j'augmente un tout petit peu `w`, l'erreur augmente-t-elle ou diminue-t-elle, et de combien ? ». Si l'erreur diminue quand `w` augmente, on augmente `w`. Avec 2 paramètres (`w`, `b`), il faut 2 gradients. Avec Llama : **8 milliards de gradients**, autant que de poids.

**2. Les états de l'optimiseur — deux valeurs par paramètre.**
L'optimiseur le plus utilisé (Adam) ne suit pas le gradient bêtement : il garde pour chaque poids une **moyenne des gradients passés** (l'élan, pour ne pas zigzaguer) et une **moyenne de leurs carrés** (pour adapter la taille du pas). Encore 2 × 8 milliards de valeurs.

**3. Les activations — les résultats intermédiaires.**
Le calcul du gradient (la *rétropropagation*, *backpropagation*) remonte le réseau de la dernière couche à la première. Pour ça, il a besoin des résultats calculés par chaque couche lors du passage « aller ». Il faut donc les garder en mémoire, pour chaque token du texte d'entraînement. En inférence, on les jette au fur et à mesure.

**4. Une précision plus élevée.**
On ne peut pas entraîner directement en 4 bits : les petites corrections se perdraient dans l'arrondi. L'entraînement se fait en 16 ou 32 bits.

**Le bilan pour Llama 3.1 8B** (entraînement complet, précision mixte, Adam) :

| Élément | Octets par paramètre | Total pour 8 B |
|---|---|---|
| Poids (bf16) | 2 | 16 Go |
| Gradients (bf16) | 2 | 16 Go |
| Copie des poids en fp32 | 4 | 32 Go |
| Adam : moyenne des gradients | 4 | 32 Go |
| Adam : moyenne des carrés | 4 | 32 Go |
| **Sous-total** | **16** | **~128 Go** |
| + activations | variable | plusieurs dizaines de Go |

Comparé aux ~5 Go de l'inférence en Q4 : **environ 25 à 30 fois plus de mémoire.**

**Une correction de ma question :** j'ai écrit « même seulement le réentraîner un peu », c'est trop fort. Il existe une technique, **QLoRA**, qui gèle les poids d'origine (gardés en 4 bits) et n'entraîne que de petites matrices ajoutées (~0,1 à 1 % des paramètres). Gradients et états d'Adam ne concernent alors que ces petites matrices. Pour un 8B, ça reste juste sur 8 Go de VRAM, mais c'est faisable avec des outils optimisés. Un entraînement **complet**, lui, est hors de portée.

### À retenir

- Entraîner = poids **+ gradients + états de l'optimiseur + activations**, en précision élevée : ~16 octets par paramètre au lieu de ~0,6 en inférence Q4.
- Le **gradient** indique, pour chaque paramètre, dans quel sens le modifier pour réduire l'erreur ; la **rétropropagation** le calcule en remontant le réseau.
- Pour adapter un gros modèle avec peu de mémoire : **LoRA / QLoRA** (on n'entraîne qu'une toute petite partie).
