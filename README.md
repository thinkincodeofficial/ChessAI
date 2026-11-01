# I Trained an AI to Play Chess in 200 Lines of Python

A chess AI that teaches itself by playing against itself. The neural network is
written from scratch in NumPy (forward pass, backprop and Adam by hand). There
are no opening books, no human games and no Stockfish. `python-chess` only
supplies the rules.

📺 Watch the video: https://youtu.be/tch24nmX3V4

## Setup

```bash
pip install -r requirements.txt
```

A trained network (`chess_net.npz`) is included, so you can run `play` or `watch` straight away.
Running `train` without `--resume` starts from scratch and overwrites it.

## Usage

```bash
python chess_ai.py train                    # 30 generations x 50 self-play games
python chess_ai.py train --resume           # keep training the saved model
python chess_ai.py play                     # you play White, AI looks 2 moves ahead
python chess_ai.py play --color black --depth 3
python chess_ai.py watch                    # AI vs MaterialBot, move by move
python plot_progress.py                     # chart of training progress -> training_progress.png
```

Moves can be typed in SAN (`e4`, `Nf3`, `O-O`, `exd5`) or UCI (`e2e4`, `e7e8q`).

## How it works

1. **Encode**: each board becomes 769 numbers: 12 piece types × 64 squares, plus whose turn it is.
2. **Network**: 769 → 256 → 64 → 1. The output is a score from −1 (Black winning) to +1 (White winning).
3. **Search**: minimax. The AI tries every move, asks the network to score each resulting
   position (in one batched pass), and picks the best. Checkmates are scored exactly.
4. **Self-play**: the AI plays itself. 10% of moves are random so it explores new ideas.
5. **Learn (TD-λ, the TD-Gammon trick)**: a game ends in checkmate (±1), a draw (0), or at 200 plies,
   where whoever is ahead on material gets a small reward. Working backwards from the end, each position's
   target blends that real result with the network's own guess one move later (λ = 0.7).
   Every position is also mirrored (colours swapped), which doubles the data for free.
6. **Repeat**: after each generation the network plays 10 games against **MaterialBot**,
   a bot that only counts material, and the result goes into `training_log.csv`.

## Results (55 generations, about 12 minutes on a laptop CPU)

| Network | vs MaterialBot (20 games) |
|---|---|
| Untrained | 1 W, 2 D, 17 L |
| Trained | 4 W, 11 D, 5 L |

## Files

| File | What it is |
|---|---|
| `chess_ai.py` | The whole AI (exactly 200 lines) |
| `plot_progress.py` | Draws the training chart |
| `chess_net.npz` | Saved network weights (created by `train`) |
| `training_log.csv` | Loss and benchmark results per generation (created by `train`) |

## Challenge

Train it for longer than I did and tell me if yours can actually checkmate someone. Mine can't.

## License

MIT
