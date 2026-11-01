# context.md: the chess AI in 200 lines

A living reference for anyone writing about this project: the script, narration, graphics or YouTube description. It covers every file, every class, every function and every number in `code/`. Line numbers refer to `chess_ai.py` as it is now (exactly 200 lines).

**Rule:** if a mechanic, number or behaviour isn't in the code and in this file, it doesn't go in the script.

---

## 1. The pitch

A chess AI that teaches itself by playing against itself. The neural network is written from scratch in NumPy: forward pass, backprop and the Adam optimiser are all hand-written. There are no opening books, no human games and no Stockfish. `python-chess` only supplies the rules (legal moves, checkmate, drawing the board).

- **Star:** the network (`Net`). It is 769 → 256 → 64 → 1, has 213,633 numbers and outputs one score between −1 and +1.
- **Rival:** `MaterialBot`, a bot that only counts material (pawn 1, knight 3, bishop 3, rook 5, queen 9).
- **Hook:** a 200-line Python file learns chess with no human help, and it still gets checkmated (the cold open).

---

## 2. Files in `code/`

| File | What it is |
|---|---|
| `chess_ai.py` | The whole AI: 200 lines, 12 functions, 2 classes, 3 modes (`train`, `play`, `watch`) |
| `plot_progress.py` | Reads `training_log.csv` and saves `training_progress.png` (2 charts) |
| `chess_net.npz` | The trained network weights (6 arrays, float32) |
| `training_log.csv` | One row per generation: `generation,positions,loss,wins,draws,losses` (55 rows, see §9) |
| `requirements.txt` | `chess>=1.10`, `numpy>=1.24`, `matplotlib>=3.7` (matplotlib is only for the chart) |
| `README.md` | Setup, usage, how it works, results table |
| `prof` | A binary cProfile dump from an **older** version of the code (its line numbers don't match the current file). It is the source of the TXT_02 profiler overlay. Read it with `python -c "import pstats; pstats.Stats('prof').sort_stats('tottime').print_stats(12)"` |

Tested with Python 3.14, python-chess 1.11.2 and NumPy on Windows 11.

---

## 3. How to run it

```bash
pip install -r requirements.txt
python chess_ai.py train                    # 30 generations x 50 self-play games, writes chess_net.npz + training_log.csv
python chess_ai.py train --resume           # keep training the saved model, appends to the log
python chess_ai.py play                     # you play White, the AI looks 2 moves ahead
python chess_ai.py play --color black --depth 3
python chess_ai.py watch                    # one game: AI (White) vs MaterialBot, move by move
python plot_progress.py                     # training_progress.png
```

### Command-line flags (`chess_ai.py` lines 186–200)

| Flag | Default | Used by | Meaning |
|---|---|---|---|
| `mode` | (required) | all | `train`, `play` or `watch` |
| `--generations` | `30` | train | How many generations to run |
| `--games` | `50` | train | Self-play games per generation |
| `--resume` | off | train | Load `chess_net.npz` and continue the generation count from the log |
| `--depth` | `2` | play, watch | How many plies the search looks ahead |
| `--color` | `white` | play | Your colour |

**There is no `--seed` flag and no frame loop.** Runs aren't reproducible unless you seed them yourself before calling the functions: `random.seed(s); np.random.seed(s)`. All the video's game footage was made like that (see §10). The `play` mode is fully deterministic, because the network has no randomness at inference time.

You can type moves as SAN (`e4`, `Nf3`, `O-O`, `exd5`) or UCI (`e2e4`, `e7e8q`). Both work, because python-chess's `push_san` accepts UCI too.

---

## 4. How the program flows

### `train` (the learning loop)
```
for each generation:
    play 50 games against itself           self_play_game() x 50
    keep the last 250 games in memory      (5 generations x 50 games)
    double the data by mirroring            add_mirrored()
    2 passes over the data in batches of 256  Net.train_step()
    play 10 games vs MaterialBot           arena()
    save the network, append a log row, print one line
```

### `play` (you vs the AI)
```
loop until game over:
    draw the board                          show()
    your turn  -> read a move from the keyboard (or "quit")
    AI's turn  -> search(depth 2) and print the move, its eval and the time taken
print the result and how the game ended
```

### `watch` (AI vs MaterialBot)
`arena(Net.load(), games=1, depth=2, watch=True)`: one game with the AI as White and **both sides searching at depth 2**. Before every move it draws the board, prints `AI  plays e3` / `Bot plays Nf6`, then waits 0.3 s. At the end it prints `AI vs MaterialBot: W wins, D draws, L losses`.

---

## 5. Every class and function, layer by layer

The file is split into 5 numbered sections, and the comments in the code use the same headings.

### Top of the file (lines 1–9)

| Line | Code | Notes |
|---|---|---|
| 1–3 | Module docstring | The title and the usage line |
| 4 | `import argparse, os, random, time` | `random` drives exploration and MaterialBot noise. `time` provides the timings and the 0.3 s watch delay |
| 5 | `os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")` | **The speed fix.** NumPy split each tiny matrix multiply across every CPU thread. One thread is "~50x faster" (from the code's own comment). It has to come before `import numpy` |
| 6–7 | `import numpy as np`, `import chess` | The only two dependencies |
| 9 | `MODEL, LOG = "chess_net.npz", "training_log.csv"` | File names, relative to the current folder |

### Section 1: turn a chess board into numbers

#### `encode(board)`: lines 12–18
- **In:** a `chess.Board`. **Out:** a float32 vector of length **769**.
- 12 piece types × 64 squares = 768 "light switches", plus 1 side-to-move bit.
- Index = `(piece_type - 1 + 6 * (not piece.color)) * 64 + square`
  - piece_type: 1 pawn, 2 knight, 3 bishop, 4 rook, 5 queen, 6 king (python-chess numbering)
  - White pieces use planes 0–5 and Black pieces use planes 6–11
  - square: 0 = a1, 7 = h1, 56 = a8, 63 = h8
- `x[768] = board.turn`: 1 if White is to move, 0 if Black.
- **Not encoded:** castling rights, en passant, move counters, repetition history. The network only sees where the pieces are and whose turn it is.
- At the start position exactly **33** of the 769 numbers are 1: 32 pieces plus "White to move".

#### `add_mirrored(X, y)`: lines 20–23
- "Free extra data": for every position it adds the colour-swapped mirror image with the opposite score.
- It reshapes the 768 piece inputs to `(N, 2 colours, 6 pieces, 8 ranks, 8 files)`, swaps the colour axis and flips the ranks upside down (files stay the same). It also flips the side-to-move bit (`1 - x`) and negates the target (`-y`).
- It doubles the training data, so the `positions` column in the log is already ×2.

### Section 2: the brain

#### `class Net`: lines 26–63
A fully connected network with hand-written backprop.

| Method | Lines | What it does |
|---|---|---|
| `__init__(sizes=(769, 256, 64, 1))` | 27–30 | He initialisation: `randn(a, b) * sqrt(2 / a)`, then **cast to float32**. Biases start at zero. It also makes one Adam `[m, v]` pair per parameter array (6 arrays) and sets the step counter `t = 0`. |
| `forward(X)` | 32–38 | `X` is `(N, 769)`. The hidden layers use **ReLU** (`np.maximum(z, 0)`) and the output layer uses **tanh**. It returns `(N,)` scores in [−1, +1]: +1 means White is winning, −1 means Black is winning. It saves every layer's activations in `self.acts` so `train_step` can reuse them. |
| `train_step(X, y, lr=3e-3)` | 40–54 | One step of gradient descent on a batch. The loss is the **mean squared error** between output and target. The gradient goes back through tanh (`(1 - out²)`), then through each layer (`acts[i].T @ grad`), and through each ReLU (`* (acts[i] > 0)`). Then **Adam** updates every parameter: β1 = 0.9, β2 = 0.999, ε = 1e-8, with bias correction. It returns the batch's MSE as a float. |
| `save(path=MODEL)` | 56–57 | `np.savez(path, W0, W1, W2, b0, b1, b2)`. The arrays get stored as `arr_0`…`arr_5`. Adam state is **not** saved. |
| `load(path=MODEL)` (classmethod) | 59–63 | It builds a fresh `Net()` first (which uses up random numbers), then overwrites the weights with the file's 6 arrays in order. `--resume` restarts Adam from zero. |

**Parameter count:** 769×256 + 256 + 256×64 + 64 + 64×1 + 1 = **213,633**.

**The float64 bug (story beat):** `np.sqrt(2 / a)` returns a float64, which silently promoted the whole network to double precision. The fix is the `.astype(np.float32)` on line 28. The script points at line 28 for this beat.

#### `class MaterialBot`: lines 65–70
The benchmark opponent. It isn't a neural network, but it has a `forward(X)` method, so `search()` can use it in place of a `Net`.

- `weights`: a fixed 769-vector. Each White piece's 64 squares get +1/+3/+3/+5/+9/0 (P/N/B/R/Q/K), each Black piece's get the same values negated, and the side-to-move input gets 0. So `X @ weights` = White's material − Black's material.
- `forward(X)` = `tanh(material / 10)` + `uniform(0, 0.001)` noise. The noise only breaks ties between equal moves, so the bot doesn't always pick the first one.
- It doesn't know about checks, king safety, development or position. It grabs material and nothing else. Checkmates still get spotted, because `score_moves` scores them exactly (see below).

### Section 3: thinking ahead

#### `game_result(board)`: lines 73–75
Returns **+1** if White won, **−1** if Black won, and **0** for everything else: stalemate, repetition, 50-move rule, insufficient material, or game not over. Only checkmate counts as a win.

#### `score_moves(board, net)`: lines 77–90
Scores every legal move in one batched network call.
1. It plays each legal move and records `game_result()`, so checkmates and rule draws are "known for sure, no guessing".
2. Positions where the game isn't over get encoded and collected.
3. One `net.forward()` call on the whole batch replaces those scores.
4. It returns `(moves, scores)`.

At the start position that's 20 legal moves, so 20 positions go through the network in one pass.

#### `search(board, net, depth=1)`: lines 92–106
Plain **minimax**: White picks the highest score and Black picks the lowest (`sign = ±1`).
- **depth 1:** `score_moves` once, then argmax. It looks at your own move only.
- **depth > 1:** for every legal move it pushes the move, recurses with `depth - 1` (or uses `game_result` if the game ended), then pops the move. It returns `(score, best_move)`, where the score is always from White's point of view.
- There's **no alpha-beta pruning, no move ordering, no transposition table and no quiescence search**, so the cost grows by roughly ×30 per extra ply.
- Where the depths are used: self-play and the training arena use depth 1. `play` and `watch` use depth 2 by default.

### Section 4: learning

#### `self_play_game(net, explore=0.1, max_plies=200, lam=0.7)`: lines 109–123
One game of the network against itself. It returns `(X, target)` for training.
1. **Moves:** 10% of the time (`explore=0.1`) it plays a uniformly random legal move "to discover new ideas". Otherwise it plays the depth-1 `search` move.
2. It encodes every position **after** each move, so the start position isn't included.
3. It stops at game over or at **200 plies** (100 moves each).
4. **The result:**
   - checkmate gives ±1
   - a draw by the rules gives 0
   - hitting the 200-ply cap gives `tanh(material / 5)`, a small reward to whoever is ahead on material. A full queen ahead gives `tanh(9/5)` ≈ 0.95. A pawn ahead gives `tanh(1/5)` ≈ 0.20.
5. **TD(λ), "the TD-Gammon trick" (lines 117–122):** it runs one forward pass to get the network's current guesses, then walks the game backwards from the last position:
   ```
   target[t] = result
   result    = 0.7 * result + 0.3 * guess[t]
   ```
   The last position learns the real result. Each earlier one learns a blend of that result and the network's own guess one move later, so the result gets passed back down the game like a baton. With λ = 0.7, the final result's weight shrinks by ×0.7 per step back: 1, 0.7, 0.49, 0.34…

#### `arena(net, games=10, depth=1, watch=False)`: lines 125–140
The AI against `MaterialBot`. It returns `(wins, draws, losses)`.
- Colours alternate: in even-numbered games (0, 2, 4…) the AI is White, in odd ones it's Black.
- Both sides search to the same `depth`, and each game is capped at **200 plies**.
- **Win:** the game ends in checkmate with the AI not the side to move. **Draw:** anything that isn't checkmate, including hitting the 200-ply cap. **Loss:** everything else.
- `watch=True`: before each move it calls `show()`, prints `AI  plays …` / `Bot plays …`, then sleeps 0.3 s.

#### `train(generations, games_per_gen, resume)`: lines 142–158
- It starts from `Net.load()` if `--resume` and the model exists; otherwise from a fresh random `Net()`.
- **Generation numbering:** with `--resume`, `first` = number of lines in the log (the header plus N rows gives N + 1). Without it, `first = 1` and the log is **overwritten** with a new header.
- **Memory:** a rolling window of the last `5 × games_per_gen` games (250 with the defaults), so the network keeps learning from the last 5 generations.
- **Training:** mirror everything, then **2 epochs** of shuffled **batches of 256**. The logged loss is the mean over the last epoch's batches.
- **Benchmark:** `arena(net)`, which is 10 games at depth 1.
- It **saves every generation**, so stopping early still leaves a usable model.
- **Log row:** `gen,len(X),loss,w,d,l`.
- **Printed line:** `Gen   7 | loss 0.0027 | vs MaterialBot: 1W 8D 1L | 14s`

### Section 5: play against it

#### `show(board, white_at_bottom=True)`: lines 161–164
Prints the board using python-chess's `board.unicode()` with `invert_color=True` (so pieces look right on a dark terminal), `·` for empty squares, rank numbers down the left and file letters along the bottom. If `white_at_bottom=False`, the board flips (ranks 1→8 from the top, files h→a).

```
8  ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
7  ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
6  · · · · · · · ·
...
1  ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖

   a b c d e f g h
```

#### `play(depth, human_color)`: lines 166–184
- It loads the trained model and draws the board from your side.
- **Your turn:** the prompt is `Your move (e.g. e4, Nf3, e2e4) or 'quit': `. If the move is illegal or malformed, python-chess raises a `ValueError` and the game prints `Illegal move, try again.`. Typing `quit` exits.
- **AI turn:** it runs `search()` at `depth`, then prints `AI plays e6   (eval +0.01, 0.0s)`. The eval is always from White's view.
- **Game over:** it prints `Game over: 1-0 CHECKMATE` (the result plus python-chess's termination name).

#### Main block: lines 186–200
The argparse setup from §3, then it dispatches to `train`, `play` or the one-game `arena` used by `watch`.

### `plot_progress.py`
- It reads `training_log.csv` with `csv.DictReader`.
- **Score %** = (wins + ½ draws) / games × 100. It's plotted in teal `#2a9d8f` with circle markers and a grey dashed line at 50%, with the y-axis from 0 to 100.
- **Training loss** (MSE) is plotted in orange `#e76f51`.
- The figure is 12 × 4.5 in, saved at 150 dpi as `training_progress.png`.
- The x-axis is the raw `generation` column, so the two runs in the current log draw a line that jumps back to generation 1 (see §9).

---

## 6. Every number in the code

| Number | Where | Meaning |
|---|---|---|
| 769 | `encode`, `Net` | Inputs: 12 × 64 + 1 |
| 768 | `encode` | Index of the side-to-move bit |
| 256, 64, 1 | `Net(sizes=…)` | Hidden layer 1, hidden layer 2, output |
| 213,633 | (derived) | Total weights and biases |
| `sqrt(2 / a)` | `Net.__init__` | He initialisation scale |
| 3e-3 | `train_step(lr=…)` | Learning rate |
| 0.9 / 0.999 / 1e-8 | `train_step` | Adam β1, β2, ε |
| 1, 3, 3, 5, 9, 0 | `MaterialBot.weights` | P, N, B, R, Q, K values |
| /10 | `MaterialBot.forward` | MaterialBot's tanh scale |
| 1e-3 | `MaterialBot.forward` | Tie-break noise |
| 0.1 | `self_play_game(explore=…)` | 10% random moves |
| 200 | `self_play_game`, `arena` | Ply cap (100 moves each) |
| /5 | `self_play_game` | Material reward scale at the cap |
| 0.7 | `self_play_game(lam=…)` | TD(λ) λ |
| 10 | `arena(games=…)` | Benchmark games per generation |
| 0.3 s | `arena` | Delay between moves when watching |
| 30 | `--generations` | Default generations |
| 50 | `--games` | Default self-play games per generation |
| 5 × 50 = 250 | `train` | Games kept in memory |
| 2 | `train` | Epochs per generation |
| 256 | `train` | Batch size |
| 2 | `--depth` | Default search depth for play and watch |
| 1 | line 5 | OpenBLAS threads |

---

## 7. What's visible on screen

There's no GUI. Everything is text in a terminal.

| Mode | What you see |
|---|---|
| `train` | One line per generation: `Gen  12 | loss 0.0031 | vs MaterialBot: 3W 4D 3L | 15s` |
| `watch` | The Unicode board, then `AI  plays Nf3` or `Bot plays e5` every 0.3 s, then the final W/D/L line |
| `play` | The board, the move prompt, `AI plays … (eval …, …s)`, `Illegal move, try again.`, `Game over: …` |
| `plot_progress.py` | `Saved training_progress.png`, plus the two-chart PNG |

The board is drawn with Unicode chess glyphs (♔♕♖♗♘♙ ♚♛♜♝♞♟) and `·` for empty squares.

---

## 8. How python-chess and NumPy are used

**python-chess:**
- `chess.Board()`
- `board.piece_map()`, `board.turn`
- `board.legal_moves`, `board.push()`, `board.pop()`, `board.push_san()`, `board.san()`
- `board.is_checkmate()`, `board.is_game_over()`, `board.ply()`
- `board.result()`, `board.outcome().termination.name`
- `board.unicode()`
- `chess.WHITE`

That's all it does: rules and drawing. It never evaluates a position.

**NumPy:**
- matrix multiply `@`, `np.tanh`, `np.maximum`
- `np.random.randn`, `np.random.permutation`, `np.random.uniform`
- `np.savez` / `np.load`
- reshape and slicing for the mirroring

No PyTorch, no TensorFlow, no autograd.

---

## 9. The data files

### `chess_net.npz`
| Array | Shape | Is |
|---|---|---|
| `arr_0` | 769 × 256 | W0 |
| `arr_1` | 256 × 64 | W1 |
| `arr_2` | 64 × 1 | W2 |
| `arr_3` | 256 | b0 |
| `arr_4` | 64 | b1 |
| `arr_5` | 1 | b2 |

All float32. This is the model from the end of the second run (55 generations in total).

### `training_log.csv`
It holds **two runs back to back**: 25 rows, then 30 rows. The generation numbers restart at 1 on row 26. The README describes these as "55 generations, about 12 minutes on a laptop CPU".

| | Run 1 (25 gens) | Run 2 (30 gens) |
|---|---|---|
| First gen vs MaterialBot | 1W 0D 9L (score 10%) | 3W 5D 2L (55%) |
| Last gen | 1W 7D 2L (45%) | 2W 8D 0L (60%) |
| Best gen | 60% | 80% (gen 5: 6W 4D 0L) |
| Lowest loss | 0.0017 | 0.0026 |
| Positions per gen once memory is full | ~61,000 | ~58,000 |

Positions per generation fill up over the first 5 generations (≈12k, 24k, 35k, 48k, 60k) as the 5-generation memory fills, then level off. ~58,000 mirrored positions from 250 games means games average roughly **115–120 plies**.

---

## 10. Real numbers from runs (measured, not invented)

**Trained network, start position (measured on this PC):**
- Network eval of the start position: **+0.007**
- Depth 1 picks **e3** (+0.0135). Depth 2 picks **e3** in **0.04 s**. Depth 3 picks **e3** in **1.8 s**.
- All 20 opening moves scored. Top 3: e3 +0.0135, e4 +0.0121, d3 +0.0109. Bottom 3: b4 −0.0126, c4 −0.0143, a4 −0.0172. (Full list in `../assets/data/board_data.json`.)
- MaterialBot at the start position: ≈ 0.000 (equal material, plus noise).

**Piece-value probe (SNIP_02):** take one White piece off the start position and see how the eval changes.
| Remove | Change |
|---|---|
| Queen (d1) | −0.012 |
| Rook (h1) | −0.009 |
| Bishop (f1) | −0.007 |
| Knight (g1) | −0.006 |
| Pawn (e2) | **+0.011** (it thinks losing the e-pawn is *good*: it opens lines) |

So it learned the right order for the big pieces, all on a tiny scale, and it got the pawn backwards.

**Benchmarks:**
- Untrained vs a random-move bot, 10 games (SNIP_01 with `random.seed(0); np.random.seed(0)`): **(7, 3, 0)**. Another unseeded run gave (3, 7, 0).
- Untrained vs MaterialBot (older version, TXT_05): (2, 2, 16).
- Before/after training, 20 games each:
  - README run: **1W 2D 17L → 4W 11D 5L**
  - a fresh rerun: 0W 3D 17L → 6W 9D 5L (it varies run to run)

**Speed:**
- The profiler (older code, TXT_02): `forward` took 2.795 s of 3.378 s, **83%** of a self-play game.
- First version, all CPU threads (TXT_01): **47 s, 49 s, 50 s** per generation.
- After the one-thread fix (TXT_03): **9–10 s** per generation, with more than twice the games.
- Current code on this PC with an idle CPU: **13–18 s** per generation (REC_04). With 24 BLAS threads forced back on: **19–23 s** (REC_05).
- One self-play game (seed 1): 57 plies in 0.11 s, ending in checkmate.

**The "learned nothing" phase (older version, TXT_06):** losses 0.094 → 0.043 over 20 generations while results stayed at 0–1 wins out of 10. That's what motivated the TD(λ) fix.

**Deterministic footage the video uses:**
- **Cold open (seed 914, `watch` settings):** the AI gets down to three pawns and a rook (ply 68), walks its king g2 → f2 → e2 → **Kd2** (ply 75), then …Rc2+ Ke1 **Qe2#** (ply 78).
- **Seed 135:** a 200-ply draw where the AI is a rook up against a lone king and just shuffles.
- **`play` session, you as White:** 1.e4 **e6** 2.Nf3 **Bb4** 3.Bc4 **d5**. That's the French Defense, every time, because play is deterministic.

---

## 11. Story loops (emergent behaviour worth a segment)

1. **It plays e3, not e4.** Its favourite first move is the "boring" e3. Nobody taught it openings, and it picked one anyway.
2. **The French AI.** Against 1.e4 it plays …e6 and …d5: a real, named opening (the French Defense), found from self-play alone.
3. **Refusing to lose ≠ winning.** It loves draws. Up a full rook against a lone king, it shuffles until the 200-ply cap (seed 135). Its training reward for the cap is only `tanh(material/5)`, so it never learned *how* to finish a game.
4. **Gets checkmated anyway.** In the cold open (seed 914) the king walks straight into d2 and Black mates on e2. At depth 2 it can't see the mate.
5. **Piece values from nothing.** Queen > rook > bishop > knight shows up in the probe, all without being told. Then the pawn comes out *positive*.
6. **The speed bug.** NumPy's thread pool made a tiny network 5× slower. The fix is one line (line 5), plus a hidden float64 promotion (line 28).
7. **Learning nothing for 20 generations.** The loss went down but the wins didn't move. Switching to TD(λ) targets fixed it.
8. **Material Bot, the honest benchmark.** A bot that only counts pieces still crushes the untrained net 17 losses to 1. After training the AI scores 4W 11D 5L against it: more draws than losses.
9. **Mirroring = free data.** Every game is seen from both sides, which doubles the data at zero cost.
10. **Exactly 200 lines.** `chess_ai.py` is 200 lines, with blank lines, comments and docstrings included.

---

## 12. What the code does NOT do (don't claim these)

- No alpha-beta pruning, opening book, endgame tablebase, transposition table or quiescence search.
- No Stockfish, no human games, no PyTorch or any other ML framework.
- No GPU: NumPy on the CPU, one thread.
- The network never sees castling rights, en passant or repetition history.
- Draws by repetition, stalemate or the 50-move rule score 0. Only hitting the 200-ply cap gives the material reward.
- No `--seed` flag, no GUI, no frame rate.
- Adam's state isn't saved, so `--resume` restarts the optimiser.
- `watch` plays exactly **one** game, and the AI is always White.
- `train` without `--resume` **overwrites** `training_log.csv` and `chess_net.npz`.
