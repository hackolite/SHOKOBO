# Mini GPT pédagogique

Ce dossier contient un mini-cours **exécutable** pour comprendre un GPT *decoder-only* depuis zéro avec PyTorch.

## Objectif

L'objectif n'est pas la performance.
L'objectif est de comprendre concrètement le chemin suivant:

texte -> tokens -> token IDs -> embeddings -> positions -> Q/K/V -> attention -> multi-head attention -> transformer blocks -> logits -> loss -> backpropagation -> génération

## Installation

```bash
cd /home/runner/work/SHOKOBO/SHOKOBO/mini_gpt
pip install -r requirements.txt
```

## Lancement

```bash
python 01_tokenizer.py
python 02_dataset.py
python 03_attention.py
python 04_transformer.py
python 05_model.py
python 06_train.py
python 07_generate.py
```

## Notebook

Lance Jupyter puis ouvre `mini_gpt_pedagogique.ipynb`:

```bash
jupyter notebook mini_gpt_pedagogique.ipynb
```

## Structure du projet

```text
mini_gpt/
├── 01_tokenizer.py
├── 02_dataset.py
├── 03_attention.py
├── 04_transformer.py
├── 05_model.py
├── 06_train.py
├── 07_generate.py
├── mini_gpt_pedagogique.ipynb
├── data/
│   └── tiny_corpus.txt
├── checkpoints/
├── README.md
└── requirements.txt
```

## Architecture

1. **Tokenizer** — transforme le texte en entiers.
2. **Dataset** — crée les couples `(X, Y)` pour prédire le token suivant.
3. **Attention** — implémente manuellement `softmax(QK^T / sqrt(d_k))V`.
4. **Transformer Block** — assemble attention, résidus, LayerNorm et FFN.
5. **MiniGPT** — ajoute embeddings de tokens, embeddings de position et tête de sortie.
6. **Training** — entraîne le modèle avec `AdamW` et `CrossEntropyLoss`.
7. **Generation** — produit du texte token par token.

## Exemples d'expériences

- modifier `embedding_dim`
- modifier `num_heads`
- modifier `num_layers`
- modifier `context_length`
- modifier `learning_rate`
- modifier `temperature`
- modifier `top_k`
- retirer temporairement le masque causal
- comparer 1 tête et plusieurs têtes
- visualiser les poids d'attention

## Exercices

Chaque script se termine par de petits exercices et leur correction.
Le notebook reprend aussi ces exercices sous forme interactive.

## Deux parcours

### Comprendre : le mode par défaut

Les sept chapitres et les premières cellules du notebook restent au niveau
caractère, avec une attention manuelle dont les poids sont visualisables.
L'entraînement sans argument utilise le petit corpus fourni, 2 couches,
une dimension de 64, 4 têtes et un contexte de 32 tokens. Le CPU suffit ;
un GPU devient recommandé lorsque le corpus ou le modèle grossit.

```bash
python /home/runner/work/SHOKOBO/SHOKOBO/mini_gpt/06_train.py
python /home/runner/work/SHOKOBO/SHOKOBO/mini_gpt/07_generate.py --prompt "bonjour "
```

### Passer à l'échelle : options indépendantes

| Option | Choix |
|---|---|
| Corpus | `--corpus /chemin/absolu/corpus.txt`, fichier UTF-8 local |
| Tokenizer | `--tokenizer char` (défaut) ou `bpe`, avec `--vocab-size 2000` |
| Taille | `--preset tiny`, `small` ou `medium` |
| Architecture | `--context-length`, `--embedding-dim`, `--num-heads`, `--num-layers`, `--ffn-dim` |
| Position | `--position-encoding learned` (défaut) ou `rope` |
| Attention | `--attention-backend manual` (défaut) ou `sdpa` |
| Génération | `--kv-cache`, désactivé par défaut |
| Calcul | `--device cpu` ou `cuda`, `--precision float32` ou `bfloat16` |

| Preset | Contexte | Dimension | Têtes | Couches | FFN |
|---|---:|---:|---:|---:|---:|
| tiny | 32 | 64 | 4 | 2 | 256 |
| small | 128 | 128 | 4 | 4 | 512 |
| medium | 256 | 256 | 8 | 6 | 1024 |

Les options explicites remplacent les valeurs du preset. La dimension doit être
divisible par le nombre de têtes ; RoPE exige aussi une dimension par tête paire.
Le nombre de paramètres dépend également du vocabulaire. Ces presets restent
des modèles pédagogiques, pas des LLM prêts pour la production.

Pour BPE uniquement, installer la dépendance facultative :

```bash
pip install -r /home/runner/work/SHOKOBO/SHOKOBO/mini_gpt/requirements-bpe.txt
```

Exemple avec un **corpus externe suffisamment long**, à fournir soi-même :

```bash
python /home/runner/work/SHOKOBO/SHOKOBO/mini_gpt/06_train.py \
  --corpus /chemin/absolu/corpus.txt \
  --checkpoint-dir /tmp/mini-gpt-avance \
  --preset small --tokenizer bpe --vocab-size 2000 \
  --position-encoding rope --attention-backend sdpa \
  --device cuda --precision bfloat16 --stride 64 --epochs 5

python /home/runner/work/SHOKOBO/SHOKOBO/mini_gpt/07_generate.py \
  --checkpoint /tmp/mini-gpt-avance/mini_gpt.pt \
  --prompt "bonjour " --kv-cache --max-new-tokens 80 \
  --device cuda --precision bfloat16
```

Sur CPU, utiliser `--device cpu --precision float32`. Le corpus fourni est trop
petit pour les grands contextes ou pour apprendre un BPE utile.

### Corpus et validation sans fuite

Le texte brut est séparé en deux portions contiguës **avant** l'apprentissage du
tokenizer et la création des fenêtres. La dernière fraction (10 % par défaut,
`--validation-fraction`) sert uniquement à la validation. Le vocabulaire caractère
et les fusions BPE sont appris sur l'entraînement seul. Les caractères inconnus
du tokenizer caractère deviennent `<unk>` ; leur décodage affiche `?`.

Le prétraitement lit le texte par blocs bornés et écrit les IDs dans des fichiers
binaires int64 sous le dossier `tokens` du répertoire de checkpoint. Le dataset
utilise un memory mapping : ni tous les IDs ni toutes les fenêtres ne sont
matérialisés en RAM. Les blocs de prétraitement constituent des frontières de
tokenisation BPE : aucune fusion ne traverse ces frontières. L'apprentissage BPE
utilise au plus le premier million de caractères d'entraînement, en blocs de
4096 caractères, afin de borner les statistiques du tokenizer. Tout le corpus
d'entraînement est ensuite encodé pour le modèle. Cette sélection privilégie le
début du corpus : veiller à ce qu'il soit représentatif. Le BPE byte-level conserve
les 256 octets et `<unk>` : son vocabulaire comporte au moins 257 entrées, même
si `--vocab-size` demande moins.

L'échantillonnage d'entraînement se fait avec remise, pour éviter une permutation
d'indices aussi grande que le corpus. Une époque correspond au nombre de fenêtres,
mais ne garantit pas de visiter chacune une fois. `--stride` espace les fenêtres ;
`--batch-size`, `--learning-rate`, `--epochs`, `--seed` et `--max-steps` contrôlent
l'entraînement. Chaque portion doit contenir plus de tokens que le contexte choisi :
une portion trop courte provoque une erreur, jamais une réutilisation du train
comme validation. Prévoir de l'espace disque pour les IDs et ne pas partager le
même répertoire de sortie entre entraînements simultanés.

### RoPE, cache et attention optimisée

- **RoPE** remplace les embeddings de position appris par des rotations de Q/K.
  Cela ne rend pas le contexte illimité et nécessite un nouvel entraînement.
- **KV cache** conserve les K/V de chaque couche en inférence uniquement.
  Après le prompt, seul le nouveau token est calculé tant que la fenêtre tient
  dans le contexte. Au débordement, toute la fenêtre glissante est recalculée
  avec des positions repartant de zéro, comme sans cache. Ce choix préserve la
  sémantique du modèle mais supprime le gain de cache lorsque la fenêtre est pleine.
- **SDPA** utilise `torch.nn.functional.scaled_dot_product_attention`. PyTorch
  sélectionne Flash Attention lorsque GPU, dtype et formes le permettent, sinon
  un autre backend compatible, y compris sur CPU. `sdpa` ne garantit donc pas
  l'utilisation effective de Flash. Sur GPU compatible, `bfloat16` favorise son
  utilisation. Les poids restent visualisables via le chemin manuel.
- Le masque causal reste actif, y compris avec un cache dont les requêtes et
  les clés ont des longueurs différentes. Le dropout de sortie conserve le
  comportement du chapitre pédagogique.

### Checkpoints et comparaisons

Les nouveaux checkpoints contiennent la configuration, l'état complet du
tokenizer (vocabulaire caractère ou sérialisation BPE), les poids et les métriques.
La génération recharge aussi les anciens checkpoints contenant seulement `vocab`
et la configuration historique, avec positions apprises et attention manuelle.
Un checkpoint BPE exige la dépendance facultative à son chargement.
Ne charger que des checkpoints de confiance ; le chargement utilise
`weights_only=True`.

L'entraînement affiche les losses train/validation pondérées par token, le nombre
de paramètres, leur taille en octets, le débit (validation comprise) et le pic de
mémoire CUDA. La taille des paramètres n'est **pas** la mémoire totale du processus ;
le pic CPU n'est pas mesuré. Pour comparer le cache sur le même modèle :

```bash
python /home/runner/work/SHOKOBO/SHOKOBO/mini_gpt/07_generate.py \
  --checkpoint /tmp/mini-gpt-avance/mini_gpt.pt \
  --prompt "bonjour " --max-new-tokens 80 --benchmark
```

Le benchmark effectue un échauffement puis compare génération gloutonne avec/sans
cache, temps, tokens/s, mémoire CUDA et égalité des textes. Répéter les mesures
sur le même matériel, avec le même dtype, prompt et contexte. Pour comparer
manuel/SDPA, relancer avec `--attention-backend manual` puis `sdpa`.
Les arrondis flottants peuvent légèrement modifier les logits ou la génération.

Comparer les losses seulement avec **le même tokenizer et la même validation**.
Les losses et débits en tokens/s caractère/BPE ne sont pas directement comparables :
un token ne couvre pas la même quantité de texte. Comparer aussi les textes produits
sur les mêmes prompts, les temps pour une quantité comparable de texte et la mémoire.

Les tests CPU utilisent la bibliothèque standard :

```bash
python -m unittest discover -s /home/runner/work/SHOKOBO/SHOKOBO/mini_gpt/tests -v
```

L'entraînement distribué, les corpus massifs prêts à l'emploi, le post-entraînement
et la qualité d'un véritable LLM restent hors périmètre.
