# ♟️ AI\_chess\_player

A Proximal Policy Optimization (PPO)–Transformer agent trained to play chess from scratch using reinforcement learning.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/SargisVardanian/AI_chess_player.git
cd AI_chess_player

# 2. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install python-chess

# 3. Run a self-play game
python game.py
```

Upon first execution, `game.py` will automatically download the pretrained model weights from GitHub Releases.

---

## Training (CLI)

Use the `train.py` wrapper to override hyperparameters without editing code:

```bash
python train.py --epochs 10 --batch 8 --save-int 200 --lr 1e-4
```

Common options:

```bash
python train.py --ckpt my_ckpt.pth --log my_log.csv --device mps --resume
```

---

## Pretrained Weights

| Filename                                   | Size   | SHA-256 Digest |
| ------------------------------------------ | ------ | -------------- |
| `chess_model_transformer_weights_exp2.pth` | 157 MB | `ed20e8b5…`    |

The weights are fetched at runtime from:

```
https://github.com/SargisVardanian/AI_chess_player/releases/download/v1.0-weights/chess_model_transformer_weights_exp2.pth
```

---

## Repository Structure

```
AI_chess_player/
├─ game.py                   # Single-game simulator + auto-weight download
├─ train_ppo_value.py        # PPO training loop with value critic
├─ Chess_background.py       # Environment, reward shaping, and utilities
├─ chess_model_reworked.py   # Transformer–CNN hybrid network
├─ images/                   # Chess piece sprite assets
├─ .github/
│   └─ workflows/
│       └─ run_game.yml      # “play-demo” GitHub Actions workflow
├─ .gitattributes            # Git LFS configuration
├─ .gitignore
└─ README.md
```

---

## Technical Overview

### PPO Training (`train_ppo_value.py`)

We optimize a policy π\_θ(a|s) via the clipped surrogate objective \[1]:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \big[ \min(r_t(\theta) \hat{A}_t,\; \mathrm{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)\,\hat{A}_t) \big],
$$

where
$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$
and the advantage estimate $\hat{A}_t$ uses one-step TD bootstrap with discount $\gamma=0.99$.

The total loss includes a value-function term and entropy regularization:

$$
L(\theta) = L^{\text{CLIP}}(\theta) + c_{v}\,L^{V}(\theta)\;-\;c_{e}\,H[\pi_\theta](s_t),
$$

with $c_v=0.9$, $c_e=0.02$, and Kullback–Leibler penalty weight $\beta=0.005$.

Each epoch iterates over a batch of 64 parallel `ChessGame` environments, applying gradient updates every step $t$. Models are checkpointed every 50 steps.

---

### Model Architecture (`Model.py`)

The agent uses a Transformer-only encoder with embedding-based inputs:

1. **Square Tokens (64)**
   Each square gets a token built by summing:
   * a learnable **position embedding** (one per square), and
   * a learnable **piece embedding** (empty + 12 pieces).

2. **CLS Token (Global Rule State)**
   The CLS token is a **sum of categorical embeddings**:
   side-to-move, four castling bits, en-passant file, halfmove bucket,
   and repetition bucket.

3. **Transformer Encoder**
   Six layers ($d_{\text{model}} = 512,\; h = 16$) process the 65 tokens
   (`[CLS] + 64 squares`).

4. **Heads**
   * **Policy:** MLP on the CLS token → 4672 logits.
   * **Value:** MLP on the CLS token → scalar $V(s)$.

---

## Interactive Execution on GitHub

### Codespaces

Launch an interactive development environment in seconds:

[![Open in Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?repo=SargisVardanian/AI_chess_player)

### GitHub Actions Demo

1. Go to **Actions → play-demo**.
2. Click **Run workflow**.
3. Inspect the logs for a step-by-step record of the agent’s moves.

The workflow `.github/workflows/run_game.yml` sets up Python 3.10, installs dependencies, and executes `python game.py`.

---

## References

\[1] Schulman, J. et al. “Proximal Policy Optimization Algorithms.” arXiv:1707.06347 (2017).
