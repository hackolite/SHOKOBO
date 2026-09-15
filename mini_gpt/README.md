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

## Limites

- tokenizer caractère par caractère, très simple
- corpus minuscule
- modèle très petit
- entraînement CPU possible mais lent par rapport à un vrai LLM
- pas de BPE, RoPE, KV cache, Flash Attention ni entraînement distribué
