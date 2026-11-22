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
| Q6 | 1.1 | Décoder avec le mauvais schéma |
| Q7 | 1.2 | Quand ne faut-il pas jeter de données ? |
| Q8 | 1.3 | Quantification : des forces G aux LLM |

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

---

## Q6 — Que se passe-t-il si on décode avec le mauvais schéma ?

*Contexte : étape 1.1 — F1 25 a deux modes UDP dont les paquets n'ont pas le même contenu (2026-11-22).*

### Question

Imaginons que le jeu envoie des paquets `LapData` au format « 2026 Season Pack » (24 voitures), mais que notre parser les décode avec la structure du mode « F1 25 » (22 voitures, champs dans un autre ordre ou de tailles différentes).
1. Est-ce que Python va forcément lever une erreur ?
2. Quel genre de valeurs verrait-on dans la console, et pourquoi est-ce plus dangereux qu'un plantage ?
3. Quelle vérification simple, faite avant de décoder, permettrait d'éviter le problème ?

### Ta réponse

> Python ne lèvera pas forcément d'erreur, les valeurs seraient fausses. Avant de décoder, on pourrait vérifier l'en-tête du paquet.

### Correction — juste, avec deux précisions

**1. Pas forcément d'erreur — ça dépend de la fonction utilisée.**
- `struct.unpack(format, data)` exige que `data` ait **exactement** la taille du format : sinon il lève `struct.error`. Là, on a de la chance, ça plante.
- `struct.unpack_from(format, data, offset)` lit seulement les octets dont il a besoin, à partir de `offset`, et ignore le reste. C'est ce qu'on utilise pour lire la voiture n°5 d'un tableau de 22. Si le paquet est plus long que prévu : **aucune erreur**, juste des octets lus au mauvais endroit.

**2. Des valeurs fausses… qui peuvent avoir l'air vraies.**
Si un champ a changé de taille au début de la structure, tous les champs suivants sont **décalés**. On lit alors les octets du champ voisin :
- parfois c'est absurde (`position = 187`, `temps au tour = 4 294 967 ms`) → on s'en rend compte ;
- parfois c'est **plausible** (une vitesse de 212 km/h qui est en fait un autre champ) → personne ne s'en rend compte.

C'est le vrai danger : l'erreur est **silencieuse** et se propage. L'expert Pace déclenche une alerte sur une fausse perte de temps, le LLM la formule avec assurance, et le TTS dit au pilote « *you're losing 3 tenths in sector 2* » alors que c'est faux. Un plantage, au moins, se voit tout de suite.

**3. Vérifier l'en-tête : oui — mais ça ne suffit pas toujours.**
On vérifie dans l'en-tête :
- `m_packetFormat == 2025` → c'est bien le format qu'on sait lire ;
- `m_packetId` connu → on sait quelle structure appliquer.

Mais l'en-tête peut dire la vérité et le contenu avoir quand même changé : on l'a vu à l'étape 1.1, **dans le même format F1 25**, le paquet CarDamage est passé de 953 à 1041 octets entre deux versions de la spec. D'où une **deuxième vérification**, complémentaire :
- `len(data) == taille_attendue[packet_id]` → le contenu a bien la forme qu'on croit.

L'en-tête dit **ce que le paquet prétend être** ; la taille vérifie **qu'il l'est vraiment**. Si l'un des deux ne correspond pas : on rejette le paquet et on l'écrit dans les logs, plutôt que de deviner.

### À retenir

- Mauvais schéma = valeurs fausses, souvent **sans erreur** : c'est pire qu'un plantage, car ça se propage jusqu'au pilote.
- Valider avant de décoder : **format + ID** (en-tête) **et taille** (contenu).
- Principe général en IA et en données : *garbage in, garbage out* — un modèle ne peut pas corriger des données d'entrée fausses ; il les transforme en réponses fausses mais convaincantes.

---

## Q7 — Quand ne faut-il pas jeter de données ?

*Contexte : étape 1.2 — notre récepteur jette les paquets les plus anciens quand sa file est pleine (2026-11-22).*

### Question

Notre récepteur privilégie la **fraîcheur** : file pleine → on jette le paquet le plus ancien. C'est le bon choix pour l'ingénieur de course en direct.
1. Dans notre projet, cite **une situation** où jeter des paquets serait au contraire une **mauvaise** idée.
2. Parmi les types de paquets F1 25, lequel serait le plus gênant à perdre, et pourquoi ? (Indice : regarde la colonne « Fréquence » de `docs/SPEC_F1_25.md`.)

### Ta réponse

> Si on jetait des paquets concernant un moment du tour important, comme un lock-up ou un spin qui expliquerait notre temps. Ce serait le paquet Event.

### Correction — point 2 juste ✅, point 1 à préciser

**Point 2 — le paquet Event : exactement.** Regarde sa fréquence dans la spec : « *when the event occurs* ». Il est envoyé **une seule fois**, sans répétition. Si on le perd, l'information est perdue **pour toujours**. Exemples :
- `FLBK` (*flashback*) : le temps de session **recule**. Sans cet événement, notre futur delta en Contre-la-montre comparerait des temps incohérents.
- `PENA` (pénalité), `FTLP` (meilleur tour), `SEND` (fin de session) : impossible de les reconstituer autrement.
Même chose pour **FinalClassification**, envoyé une seule fois en fin de course.

**Point 1 — le lock-up : bonne intuition, mais un détail change tout.**
- Le lock-up et le tête-à-queue ne sont **pas** des événements du paquet Event (la spec n'en a pas). On devra les **détecter nous-mêmes** dans les données continues : freinage fort + roue qui glisse (`m_wheelSlipRatio` du paquet MotionEx), ou vitesse de rotation anormale.
- Or ces données continues sont **redondantes dans le temps** : à 20 Hz, un lock-up qui dure ~0,5 s est décrit par ~10 paquets consécutifs. En perdre 1 ou 2 ne fait pas disparaître le lock-up ; on le voit encore sur les autres. C'est pour ça que jeter des paquets CarTelemetry de temps en temps est acceptable en direct.

**La vraie situation où il ne faut rien jeter : l'enregistrement.** `tools/record_session.py` (étape 1.6) doit garder **tous** les paquets : la validation 1.8 exige un rejeu **identique** à la session réelle, et la trace du meilleur tour (Contre-la-montre) doit être complète. En direct on privilégie la fraîcheur ; à l'enregistrement, l'exhaustivité.

**Conséquence pour notre code :** on notera que les paquets **Event** (et FinalClassification) ne devront jamais être jetés, même file pleine.

### À retenir

- **Données discrètes** (un événement, envoyé une fois) : perdues = perdues. À protéger.
- **Données continues échantillonnées** (télémétrie à 20 Hz) : redondantes, une perte ponctuelle est tolérable… tant que le phénomène dure plus longtemps que l'intervalle entre deux mesures (50 ms à 20 Hz). Un événement plus bref que ça peut passer entre deux échantillons, même sans aucune perte : c'est une limite de l'**échantillonnage**, pas de la file.
- Temps réel → **fraîcheur** ; enregistrement / données d'entraînement → **exhaustivité**.

---

## Q8 — Quantification : des forces G aux LLM

*Contexte : étape 1.3 — en comparant les specs, on a vu que le format 2026 change le type des forces G (2026-11-22).*

### Question

Dans le format 2025, les forces G sont des `float` (4 octets). Dans le format 2026, ce sont des `int16` (2 octets) « quantifiés » : le jeu envoie `round(g × 1000)` et on retrouve la valeur en divisant par 1000.

1. Le jeu mesure une force latérale de **3,14159 g**. Quelle valeur reçoit-on après quantification ? Quelle est la plus petite différence de force G qu'on peut encore distinguer ?
2. Un `int16` va de −32 768 à 32 767. Quelle est la force G maximale représentable ? Est-ce suffisant pour une F1 (environ 5 à 6 g en virage, parfois 50 g ou plus lors d'un gros choc) ?
3. Quel est le lien avec Llama 3.1 8B « en Q4 » qu'on utilisera en Phase 5 ?

### Ta réponse

> 1. On reçoit 3141. 2. En float 9999, et c'est largement suffisant. 3. Je ne sais pas.

### Correction — 1 presque juste, 2 à revoir, 3 expliqué

**1. Presque : 3142, pas 3141.** 3,14159 × 1000 = 3141,59, et `round` arrondit au **plus proche** → **3142**. (3141 serait une *troncature*, qui coupe les décimales.) À la lecture : 3142 / 1000 = **3,142 g**.
La plus petite différence visible est **0,001 g** : c'est le **pas** de quantification. 3,1415 g et 3,1419 g deviennent tous deux 3142 : la différence est perdue. L'erreur maximale est d'un demi-pas, ±0,0005 g.

**2. À revoir : le maximum est 32,767 g.** Le type de stockage est `int16`, pas `float` : la plus grande valeur envoyée est 32 767, donc la plus grande force représentable est 32 767 / 1000 = **32,767 g** (et −32,768 g de l'autre côté).
- En virage (5 à 6 g) : **largement suffisant**, tu as raison sur ce point.
- Lors d'un gros choc (50 g ou plus) : **la valeur sature** et reste bloquée à 32,767 g. On saura qu'il y a eu un choc violent, mais pas sa force exacte.

C'est le compromis central de la quantification : avec un nombre fixe de valeurs possibles (65 536 pour un `int16`), le **facteur d'échelle** répartit ces valeurs entre **étendue** et **précision**.

| Facteur d'échelle | Pas (précision) | Plage représentable |
|---|---|---|
| × 1000 (choix d'EA) | 0,001 g | ±32,7 g |
| × 100 | 0,01 g | ±327 g |
| × 10 000 | 0,0001 g | ±3,3 g → sature déjà en virage ❌ |

**3. Le lien avec Llama 3.1 8B en Q4 : c'est exactement le même principe, poussé beaucoup plus loin.**

Les 8 milliards de poids de Llama sont des nombres à virgule, stockés à l'origine sur 16 bits (2 octets) → **16 Go**. Quantifier en **Q4**, c'est stocker chaque poids comme un **entier sur 4 bits**, soit seulement **16 valeurs possibles** (de −8 à 7), plus un **facteur d'échelle** :

> poids ≈ entier × facteur d'échelle

Exemple avec un bloc de 4 poids :
- Poids d'origine : `0,12 ; −0,03 ; 0,07 ; −0,11`
- Plus grande valeur absolue = 0,12 → facteur d'échelle = 0,12 / 7 ≈ 0,0171
- Entiers stockés (4 bits chacun) : `round(poids / 0,0171)` = `7 ; −2 ; 4 ; −6`
- Poids reconstruits : `0,120 ; −0,034 ; 0,069 ; −0,103` → proches des originaux, mais pas identiques.

**L'astuce qui rend ça possible :** au lieu d'un seul facteur d'échelle pour tout le modèle (comme le × 1000 unique d'EA), chaque **petit bloc** de poids (par exemple 32) a **son propre** facteur, adapté à sa plage de valeurs. On règle ainsi le compromis étendue / précision bloc par bloc. En comptant ces facteurs, on arrive à environ 4,5 bits par poids en moyenne.

**Ce qu'on y gagne :**
- **Mémoire** : ~16 Go → **~5 Go**. C'est ce qui permet de faire tenir Llama 8B dans tes 8 Go de VRAM à côté du jeu.
- **Vitesse** : la génération étant limitée par la lecture mémoire (Q2), 3 fois moins d'octets à lire par token = génération nettement plus rapide.

**Ce qu'on y perd :** un peu de précision sur chaque poids, donc une légère baisse de qualité. Les réseaux de neurones supportent bien ce « bruit » : une erreur sur un poids isolé compte peu, parce que chaque résultat est une somme de milliers de produits et que les petites erreurs se compensent en partie. En dessous de 4 bits, la qualité se dégrade en revanche nettement.

### À retenir

- Quantifier = remplacer un nombre à virgule par **un entier + un facteur d'échelle**.
- Le facteur d'échelle fixe le compromis : **pas fin = petite plage**, **grande plage = pas grossier**. Hors de la plage, la valeur **sature**.
- Les LLM appliquent la même idée avec des **facteurs par bloc** : Q4 = 16 valeurs par poids, ~3× moins de mémoire que 16 bits, génération plus rapide, petite perte de qualité.
