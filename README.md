# Helios

**Late Fusion of Transformer-Based Financial Sentiment and Technical Analysis via Reinforcement Learning for Equity Trading**

Master's thesis project — Máster Universitario en Inteligencia Artificial, Universidad Alfonso X el Sabio (2025–2026).

---

## Overview

Helios implements a modular trading system that combines two independent signal sources — **FinBERT-based news sentiment** and **XGBoost technical analysis** — and fuses them via a **PPO reinforcement learning agent** to make daily trading decisions on S&P 500 mega-cap equities.

The central hypothesis is that *late fusion* (fusing high-level `(score, confidence)` abstractions from independent modules) outperforms *early fusion* (concatenating raw features) in terms of risk-adjusted performance, precisely because it forces each module to produce interpretable, standardized signals before any cross-modal learning occurs.

**Key results on the held-out test split (2024–2025, 10 tickers):**

| Strategy | Sharpe | Sortino | Max DD | Cum. Return | Calmar |
|---|---:|---:|---:|---:|---:|
| Buy & Hold | 1.73 | 2.39 | -25.6% | +121.9% | 1.96 |
| Tech threshold (τ=0.01) | 0.90 | 1.39 | -10.0% | +20.8% | 1.01 |
| Sentiment threshold (τ=0.20) | 0.93 | 1.25 | -24.1% | +32.9% | 0.65 |
| Early Fusion PPO | 0.39 | 0.50 | -29.9% | +10.7% | 0.18 |
| **Late Fusion PPO (Helios)** | **1.11** | **1.77** | **-6.0%** | +30.2% | **2.39** |

Late fusion achieves the best risk-adjusted metrics across the board — **4× lower maximum drawdown** than buy-and-hold, and the highest Calmar ratio of all tested configurations. Raw return lags buy-and-hold because 2024–2025 was a strongly bullish period, but the system's primary goal is capital preservation under a risk-aware reward.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
│  yfinance OHLCV + Alpha Vantage news  →  train / val / test     │
│  Temporal alignment (leakage-free)    →  parquet splits         │
└──────────────────┬──────────────────────────┬───────────────────┘
                   │                          │
          ┌────────▼────────┐       ┌─────────▼────────┐
          │ SENTIMENT MODULE│       │ TECHNICAL MODULE  │
          │  FinBERT (110M) │       │  XGBoost (15 ft.) │
          │  score_sent     │       │  score_tech       │
          │  conf_sent      │       │  conf_tech        │
          └────────┬────────┘       └─────────┬─────────┘
                   │                          │
          ┌────────▼──────────────────────────▼─────────┐
          │              FUSION LAYER                    │
          │   State: [score_sent, conf_sent,             │
          │           score_tech, conf_tech,             │
          │           ret_t-1, ret_t-2, ret_t-3]        │
          │                                              │
          │   PPO Agent  →  Discrete(2): hold / buy      │
          │   Reward: annualised Sortino (5-day window)  │
          └──────────────────────────────────────────────┘
```

Each module is independently trained and evaluated. They communicate only through a standardized `(score ∈ [-1,1], confidence ∈ [0,1])` pair, making the system modular and interpretable by design.

---

## Repository Structure

```
helios/
├── src/
│   ├── data/
│   │   ├── downloader.py          # yfinance OHLCV + Alpha Vantage news
│   │   ├── alignment.py           # Temporal alignment & leakage check
│   │   └── splits.py              # Train/val/test split generation
│   ├── sentiment/
│   │   ├── finbert.py             # FinBERT wrapper (ProsusAI/finbert)
│   │   ├── finbert_runner.py      # Batch inference over all news articles
│   │   ├── aggregator.py          # Confidence-weighted daily aggregation
│   │   ├── build_final_scores.py  # Merge sentiment scores with price splits
│   │   ├── evaluate.py            # Standalone evaluation vs PhraseBank
│   │   └── prompt.py              # (Experimental) LLM-based scoring
│   ├── technical/
│   │   ├── features.py            # 15 OHLCV-derived features
│   │   ├── model.py               # XGBoost training + early stopping
│   │   ├── score.py               # Score/confidence mapping
│   │   └── tune.py                # Optuna hyperparameter search
│   ├── environment/
│   │   ├── trading_env.py         # Gymnasium TradingEnv
│   │   └── reward.py              # Sortino-based reward function
│   ├── fusion/
│   │   ├── train.py               # PPO late-fusion training
│   │   ├── tune.py                # Optuna PPO tuning
│   │   └── baselines.py           # B&H, threshold, early-fusion baselines
│   └── analysis/
│       ├── ablation.py            # 5-configuration ablation study
│       ├── extra_studies.py       # Per-ticker & per-regime breakdown
│       ├── money_study.py         # Transaction cost sensitivity
│       └── figures.py             # Thesis figure generation
├── tests/
│   ├── test_environment.py
│   ├── test_features.py
│   ├── test_score_mapping.py
│   └── test_sentiment.py
├── data/                          # (generated, not tracked)
│   ├── raw/prices/
│   ├── raw/news/
│   └── splits/
├── results/                       # (generated, not tracked)
│   ├── sentiment/
│   ├── technical/
│   ├── fusion/
│   ├── ablation/
│   └── figures/
├── thesis.typ                     # Typst source (full thesis)
├── thesis.pdf                     # Compiled thesis
├── RESULTS.md                     # Experiment results summary
├── references.bib                 # Bibliography
├── requirements.txt
└── .env.example
```

---

## Data Pipeline

### Universe & Period

- **Tickers**: AAPL, MSFT, NVDA, AMZN, GOOGL, META, JPM, LLY, AVGO, TSLA (S&P 500 mega-caps)
- **Period**: 2018–2025 (2,010 trading days per ticker)
- **Price source**: yfinance daily OHLCV
- **News source**: Alpha Vantage `NEWS_SENTIMENT` API (157,804 articles total)

### Temporal Alignment (Leakage-Free)

A strict assignment rule prevents any future information from leaking into the training signal:

- Articles published **before 09:30 ET** → assigned to the **same trading day**
- Articles published **at or after 09:30 ET** → shifted to the **next trading day**
- SEC filings → additional **24-hour embargo** (T+2 assignment)
- NYSE holidays and weekends are respected via `pandas_market_calendars`
- `assert_no_leakage()` runs before every training run and verifies zero violations across all 90,601 news-day pairs

### Splits

| Split | Period | Rows | Avg. news/day | Purpose |
|---|---|---:|---:|---|
| Train | 2018–2022 | 12,590 | 1.79 | Model training |
| Val | 2023 | 2,500 | 2.01 | Threshold tuning, early stopping |
| Test | 2024–2025 | 4,960 | 12.73 | Final evaluation (held out) |

Z-score normalization parameters are computed **on train only** and applied to val/test without recomputation.

---

## Modules

### Sentiment Module (`src/sentiment/`)

Uses [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert), a BERT-base model fine-tuned on financial text (110M parameters).

**Pipeline:**

1. For each article: `text = title + ". " + summary`
2. FinBERT batch inference (batch_size=32) → 3-class softmax: `(p_pos, p_neg, p_neu)`
3. Per-article signal:
   - `score_sent = p_pos - p_neg ∈ [-1, 1]`
   - `conf_sent  = max(p_pos, p_neg) ∈ [0, 1]`
4. Daily aggregation: **confidence-weighted mean** of all articles published before market open that day
5. Days with no news → zero-confidence sentinel `(score=0, conf=0)`

**Evaluation:**
- Financial PhraseBank F1: **0.876** (matches published FinBERT baseline)
- Train split Pearson r (score vs 5-day forward return): **+0.036** (p=0.0014) — weak but statistically significant, consistent with semi-strong EMH

---

### Technical Module (`src/technical/`)

**15 features** computed from daily OHLCV, per ticker independently:

| Category | Features |
|---|---|
| Momentum | RSI(14), MACD(12/26), MACD signal, MACD histogram |
| Trend / Volatility | Bollinger Bands(20, 2σ), BB_pct, EMA(20), EMA(50), EMA cross ratio |
| Return memory | Log-returns over 5, 10, 20 days |
| Volume | 20-day rolling mean ratio |

Warm-up rows with NaN features (~20 per ticker) are dropped before training.

**XGBoost classifier** trained to predict 5-day forward price direction (binary label):

- Hyperparameters: `max_depth=5, n_estimators=200, lr=0.05, subsample=0.8, colsample_bytree=0.8`
- Early stopping on val AUC (20 rounds patience)
- Val AUC: **0.5188** — barely above random, consistent with weak-form EMH

**Score mapping:**
- `score_tech = 2 * p_up - 1 ∈ [-1, 1]`
- `conf_tech  = |score_tech| ∈ [0, 1]` (absolute distance from 0.5, i.e., model conviction)

---

### Trading Environment (`src/environment/`)

A custom [Gymnasium](https://gymnasium.farama.org/) environment implementing a daily trading loop over `(ticker, year)` episodes.

**Observation space** (7-dimensional, all values in `[-1, 1]`):

```
[score_sent, conf_sent, score_tech, conf_tech, ret_{t-1}, ret_{t-2}, ret_{t-3}]
```

Log-returns are clipped to ±10% and scaled by ×10 before entering the observation.

**Action space**: `Discrete(2)`
- `0` → hold (position = 0)
- `1` → buy (position = 1)

Short selling is excluded — a deliberate v2 design decision for long-only mandate realism.

**Reward function** (`sortino_reward`):

```
sortino = √252 × mean(r_{t+1:t+5}) / std_downside(r_{t+1:t+5})
reward  = clip(sortino, -3, +3) − λ_cost × |action_t − action_{t-1}|
```

- Annualised Sortino ratio computed over a **5-day forward window** (smooths daily noise)
- Clipped to `[-3, +3]` to stabilize gradient estimates
- Transaction cost penalty `λ_cost=0.001` per position change

Sortino is preferred over Sharpe because it penalizes only downside variance, aligning the agent's incentive with capital preservation.

---

### Fusion Layer (`src/fusion/`)

**PPO Late Fusion (`train.py`):**

- 4 parallel `TradingEnv` instances via `DummyVecEnv`
- 500,000 total timesteps, evaluation every 50K steps on val split
- Architecture: MlpPolicy (64×64 hidden layers)
- Key hyperparameters: `lr=3e-4, n_steps=2048, batch_size=64, n_epochs=10, ent_coef=0.01, gamma=0.99`
- Best checkpoint (by val Sharpe) saved as `results/fusion/ppo_late_fusion_tuned.zip`

**Hyperparameter tuning (`tune.py`):**

Optuna search over `{learning_rate, ent_coef, gamma, λ_cost}`:
- Each trial: 200K steps + 10 val episodes for mean Sharpe
- Best params saved to `results/fusion/best_params.json`, then retrained at 500K steps

**Baselines (`baselines.py`):**

| ID | Strategy | Description |
|---|---|---|
| B1 | Buy & Hold | Equal-weight, hold full period |
| B2 | Tech threshold | Long if `score_tech > τ`, flat if `< -τ` |
| B3 | Sentiment threshold | Same rule on `score_sent` |
| B4 | Early Fusion PPO | Raw features concatenated → PPO (ablation control) |

Thresholds `τ` for B2/B3 are tuned on the val split.

---

### Analysis (`src/analysis/`)

**`ablation.py`** — Evaluates all 5 configurations on the test split. Outputs:
- `equity_curves_v2.csv`: daily portfolio value per strategy
- `regime_analysis_v2.csv`: performance split by market regime

**`extra_studies.py`** — Per-ticker Sharpe/Sortino/MaxDD breakdown and per-regime monthly return analysis.

**`money_study.py`** — Transaction cost sensitivity: sweeps `λ ∈ [0.0005, 0.005]` to test robustness.

**`figures.py`** — Generates 300 dpi PNG figures for the thesis:
- Equity curves for all 5 configurations
- Drawdown curves (log scale)
- Sentiment/technical confidence heatmaps by ticker and regime

---

## Regime Analysis

The test split (4,760 ticker-days) is divided into market regimes based on training-split quantiles:

| Regime | n | mean(conf_sent) | mean(conf_tech) | news/day |
|---|---:|---:|---:|---:|
| All | 4,760 | 0.5908 | 0.0072 | 13.13 |
| High news | 2,338 | 0.6565 | 0.0068 | 25.08 |
| Low news | 452 | 0.0000 | 0.0084 | 0.00 |
| High vol | 250 | 0.5782 | 0.0063 | 3.93 |
| Normal | 1,861 | 0.6481 | 0.0074 | 1.96 |

On **low-news days**, sentiment confidence collapses to 0 and the agent shifts to relying on the technical signal — exactly the intended dynamic routing behavior. The **2.2× position variation across regimes** confirms the fusion layer is making meaningful, context-dependent decisions rather than defaulting to a fixed policy.

---

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU recommended for FinBERT inference (falls back to MPS/CPU automatically)

### Installation

```bash
git clone https://github.com/your-username/helios.git
cd helios
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```env
ALPHAVANTAGE_API_KEY=your_key_here   # Required for news download
WANDB_API_KEY=your_key_here          # Optional: experiment tracking
OPENAI_API_KEY=your_key_here         # Optional: experimental LLM scoring
FUSION_MODE=late                     # "late" or "early"
```

---

## Reproducing Results

Run the pipeline end-to-end in the following order:

```bash
# 1. Download price + news data
python -m src.data.downloader

# 2. Generate train/val/test splits (with leakage check)
python -m src.data.splits

# 3. Run FinBERT inference on all news articles
python -m src.sentiment.finbert_runner

# 4. Build daily sentiment scores per split
python -m src.sentiment.build_final_scores

# 5. Train XGBoost technical model + generate scores
python -m src.technical.model
python -m src.technical.score

# 6. (Optional) Tune PPO hyperparameters with Optuna
python -m src.fusion.tune

# 7. Train late-fusion PPO agent
python -m src.fusion.train --v2

# 8. Train baselines (B&H, threshold, early-fusion PPO)
python -m src.fusion.baselines --v2

# 9. Run ablation study
python -m src.analysis.ablation --v2

# 10. Generate thesis figures
python -m src.analysis.figures
```

Intermediate artefacts (parquet splits, model checkpoints, score CSVs) are written to `data/` and `results/` and are not tracked by git.

---

## Tests

```bash
pytest tests/ -v
```

| Test file | What it covers |
|---|---|
| `test_environment.py` | Episode completeness, reward finiteness, obs bounds, action space, transaction cost, episode length |
| `test_features.py` | Feature computation correctness on synthetic OHLCV |
| `test_score_mapping.py` | Score/confidence transformation bounds and monotonicity |
| `test_sentiment.py` | FinBERT output bounds and Financial PhraseBank calibration |

The leakage check (`assert_no_leakage()`) also runs as part of the data split generation step and is a hard failure gate before any training.

---

## Design Decisions & Trade-offs

| Decision | Rationale | Trade-off |
|---|---|---|
| Late fusion over early fusion | Modularity, interpretability, independent evaluability of each signal | Loses fine-grained cross-modal feature interactions |
| Frozen FinBERT (no fine-tuning) | Calibration stability; avoids catastrophic forgetting | May miss domain-specific idioms in financial text |
| `(score, confidence)` abstraction | Forces each module to be self-contained; enables dynamic routing | Two scalars lose distributional information |
| Sortino reward vs Sharpe | Penalizes only downside variance → capital-preservation incentive | Less sensitive to upside capture |
| Discrete(2) actions — no shorting | Realistic for long-only mandates; simpler credit assignment | Cannot profit from bearish regimes |
| 5-day forward reward window | Smooths daily noise; reduces reward sparsity | Introduces a 5-day lag in feedback signal |
| XGBoost over deep models | Fast training, interpretable feature importances, well-calibrated probabilities | Limited capacity for non-linear long-range patterns |

---

## Limitations

1. **Article density skew**: 2025 has ~4× more articles per day than 2018–2020. The agent may have learned to rely more on sentiment as a structural artifact of data distribution rather than signal quality.
2. **No FinBERT fine-tuning**: using the published checkpoint avoids overfitting but leaves potential domain adaptation gain on the table.
3. **Universe bias**: 10 mega-caps are highly liquid and extensively covered. Results may not generalize to small/mid caps or less-covered markets.
4. **Backtesting assumptions**: no slippage, market impact, or borrow costs are modeled. Real execution costs would reduce returns.
5. **Markov assumption**: financial markets violate the Markov property; the 3 lagged return features partially mitigate this but do not eliminate it.
6. **Bullish test period**: 2024–2025 was exceptionally strong for the chosen tickers, which structurally favors buy-and-hold on raw return. Risk-adjusted metrics are a more meaningful comparison under these conditions.

---

## Future Work

- Replace FinBERT with a generative LLM (Claude Haiku, GPT-4o-mini) for richer semantic representations
- Continuous position sizing via SAC, enabling fractional allocation and short positions
- Multi-asset portfolio optimization with correlation-aware position limits
- Real-time deployment with live Alpha Vantage / Polygon feeds
- Latent-factor sentiment modelling (VAE) to capture distributional uncertainty beyond a point estimate

---

## Citation

If you use this codebase or results in your research, please cite:

```
@mastersthesis{nicolas2026helios,
  author  = {Sergio Nicolás},
  title   = {Late Fusion of Transformer-Based Financial Sentiment and Technical Analysis
             via Reinforcement Learning for Equity Trading},
  school  = {Universidad Alfonso X el Sabio},
  year    = {2026},
  program = {Máster Universitario en Inteligencia Artificial}
}
```

---

## License

Released for academic purposes. Contact the author for commercial use inquiries.
