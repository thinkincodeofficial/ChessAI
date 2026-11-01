"""I Trained an AI to Play Chess in 200 Lines of Python.
A tiny neural network, written from scratch in NumPy, learns chess purely by playing against itself.
Usage:  python chess_ai.py train | play | watch      (add -h for options)"""
import argparse, os, random, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # tiny matrices: 1 thread is ~50x faster
import numpy as np
import chess

MODEL, LOG = "chess_net.npz", "training_log.csv"

# ---------- 1. Turn a chess board into numbers ----------
def encode(board):
    """12 piece types x 64 squares (1 = piece is there) + 1 side-to-move bit = 769 inputs."""
    x = np.zeros(769, np.float32)
    for square, piece in board.piece_map().items():
        x[(piece.piece_type - 1 + 6 * (not piece.color)) * 64 + square] = 1
    x[768] = board.turn
    return x

def add_mirrored(X, y):
    """Free extra data: flip every board and swap colours -> same position, opposite score."""
    planes = X[:, :768].reshape(-1, 2, 6, 8, 8)[:, ::-1, :, ::-1, :].reshape(-1, 768)
    return np.vstack([X, np.hstack([planes, 1 - X[:, 768:]])]), np.concatenate([y, -y])

# ---------- 2. The brain: a neural network with hand-written backprop ----------
class Net:
    def __init__(self, sizes=(769, 256, 64, 1)):
        self.W = [(np.random.randn(a, b) * np.sqrt(2 / a)).astype(np.float32) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.adam, self.t = [[np.zeros_like(p), np.zeros_like(p)] for p in self.W + self.b], 0

    def forward(self, X):
        """Returns a score in [-1, 1]: +1 = White is winning, -1 = Black is winning."""
        self.acts = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = self.acts[-1] @ W + b
            self.acts.append(np.tanh(z) if i == len(self.W) - 1 else np.maximum(z, 0))
        return self.acts[-1][:, 0]

    def train_step(self, X, y, lr=3e-3):
        out = self.forward(X)
        grad = ((out - y) * (1 - out ** 2) * 2 / len(X))[:, None]  # d(MSE)/dz through tanh
        gW, gb = [], []
        for i in reversed(range(len(self.W))):
            gW.insert(0, self.acts[i].T @ grad)
            gb.insert(0, grad.sum(0))
            if i:
                grad = (grad @ self.W[i].T) * (self.acts[i] > 0)  # back through ReLU
        self.t += 1
        for p, g, (m, v) in zip(self.W + self.b, gW + gb, self.adam):  # Adam optimiser
            m[:] = 0.9 * m + 0.1 * g
            v[:] = 0.999 * v + 0.001 * g * g
            p -= lr * (m / (1 - 0.9 ** self.t)) / (np.sqrt(v / (1 - 0.999 ** self.t)) + 1e-8)
        return float(np.mean((out - y) ** 2))

    def save(self, path=MODEL):
        np.savez(path, *self.W, *self.b)

    @classmethod
    def load(cls, path=MODEL):
        net, arrays = cls(), list(np.load(path).values())
        net.W, net.b = arrays[:len(net.W)], arrays[len(net.W):]
        return net

class MaterialBot:
    """The benchmark opponent: grabs material (pawn=1 ... queen=9) and knows nothing else."""
    weights = np.array([side * v for side in (1, -1) for v in (1, 3, 3, 5, 9, 0) for _ in range(64)] + [0], np.float32)

    def forward(self, X):
        return np.tanh(X @ self.weights / 10) + np.random.uniform(0, 1e-3, len(X))  # noise breaks ties

# ---------- 3. Thinking ahead: minimax search using the network's judgement ----------
def game_result(board):
    """+1 White won, -1 Black won, 0 draw (or game not over)."""
    return (-1.0 if board.turn == chess.WHITE else 1.0) if board.is_checkmate() else 0.0

def score_moves(board, net):
    """Score every legal move at once with a single batched pass through the network."""
    moves, scores, X, idx = list(board.legal_moves), [], [], []
    for i, move in enumerate(moves):
        board.push(move)
        scores.append(game_result(board))  # checkmates and draws are known for sure, no guessing
        if not board.is_game_over():
            X.append(encode(board))
            idx.append(i)
        board.pop()
    scores = np.array(scores, np.float32)
    if X:
        scores[idx] = net.forward(np.array(X))
    return moves, scores

def search(board, net, depth=1):
    """White picks the highest score, Black the lowest. Returns (score, best_move)."""
    sign = 1 if board.turn == chess.WHITE else -1
    if depth == 1:
        moves, scores = score_moves(board, net)
        i = int(np.argmax(sign * scores))
        return float(scores[i]), moves[i]
    best_score, best_move = -2.0, None
    for move in list(board.legal_moves):
        board.push(move)
        score = game_result(board) if board.is_game_over() else search(board, net, depth - 1)[0]
        board.pop()
        if sign * score > best_score:
            best_score, best_move = sign * score, move
    return sign * best_score, best_move

# ---------- 4. Learning: play itself, remember who won, repeat ----------
def self_play_game(net, explore=0.1, max_plies=200, lam=0.7):
    board, positions = chess.Board(), []
    while not board.is_game_over() and board.ply() < max_plies:
        explore_now = random.random() < explore  # sometimes try a random move to discover new ideas
        board.push(random.choice(list(board.legal_moves)) if explore_now else search(board, net, 1)[1])
        positions.append(encode(board))
    # Checkmate = full reward. Game too long = small reward to whoever has more material.
    result = game_result(board) if board.is_game_over() else float(np.tanh(encode(board) @ MaterialBot.weights / 5))
    # TD(lambda): each position learns from a blend of the real result and its own guess one move later
    X = np.array(positions)
    guesses, target = net.forward(X), np.zeros(len(X), np.float32)
    for t in reversed(range(len(X))):
        target[t] = result
        result = lam * result + (1 - lam) * guesses[t]
    return X, target

def arena(net, games=10, depth=1, watch=False):
    """AI vs MaterialBot, alternating colours. Returns (wins, draws, losses)."""
    wins, draws, bot = 0, 0, MaterialBot()
    for g in range(games):
        board, ai_color = chess.Board(), g % 2 == 0
        while not board.is_game_over() and board.ply() < 200:
            ai_turn = board.turn == ai_color
            move = search(board, net if ai_turn else bot, depth)[1]
            if watch:
                show(board, ai_color)
                print(f"{'AI ' if ai_turn else 'Bot'} plays {board.san(move)}")
                time.sleep(0.3)
            board.push(move)
        wins += board.is_checkmate() and board.turn != ai_color
        draws += not board.is_checkmate()
    return wins, draws, games - wins - draws

def train(generations, games_per_gen, resume):
    net = Net.load() if resume and os.path.exists(MODEL) else Net()
    first = len(open(LOG).readlines()) if resume and os.path.exists(LOG) else 1
    if first == 1:
        open(LOG, "w").write("generation,positions,loss,wins,draws,losses\n")
    memory = []
    for gen in range(first, first + generations):
        t0 = time.time()
        memory = (memory + [self_play_game(net) for _ in range(games_per_gen)])[-5 * games_per_gen:]  # last 5 gens
        X, y = add_mirrored(np.concatenate([g[0] for g in memory]), np.concatenate([g[1] for g in memory]))
        for _ in range(2):  # epochs
            order = np.random.permutation(len(X))
            losses = [net.train_step(X[order[i:i + 256]], y[order[i:i + 256]]) for i in range(0, len(X), 256)]
        w, d, l = arena(net)
        net.save()
        open(LOG, "a").write(f"{gen},{len(X)},{np.mean(losses):.4f},{w},{d},{l}\n")
        print(f"Gen {gen:3d} | loss {np.mean(losses):.4f} | vs MaterialBot: {w}W {d}D {l}L | {time.time() - t0:.0f}s")

# ---------- 5. Play against it ----------
def show(board, white_at_bottom=True):
    rows = board.unicode(invert_color=True, empty_square="·", orientation=white_at_bottom).split("\n")
    ranks, files = ("87654321", "a b c d e f g h") if white_at_bottom else ("12345678", "h g f e d c b a")
    print("\n" + "\n".join(f"{r}  {row}" for r, row in zip(ranks, rows)) + f"\n\n   {files}\n")

def play(depth, human_color):
    net, board, human = Net.load(), chess.Board(), human_color == "white"
    while not board.is_game_over():
        show(board, human)
        if board.turn == human:
            text = input("Your move (e.g. e4, Nf3, e2e4) or 'quit': ").strip()
            if text == "quit":
                return
            try:
                board.push_san(text)
            except ValueError:
                print("Illegal move, try again.")
        else:
            t0 = time.time()
            score, move = search(board, net, depth)
            print(f"AI plays {board.san(move)}   (eval {score:+.2f}, {time.time() - t0:.1f}s)")
            board.push(move)
    show(board, human)
    print("Game over:", board.result(), board.outcome().termination.name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A self-taught chess AI in 200 lines.")
    parser.add_argument("mode", choices=["train", "play", "watch"])
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--games", type=int, default=50, help="self-play games per generation")
    parser.add_argument("--resume", action="store_true", help="continue training the saved model")
    parser.add_argument("--depth", type=int, default=2, help="how many moves ahead the AI looks")
    parser.add_argument("--color", choices=["white", "black"], default="white", help="your colour")
    args = parser.parse_args()
    if args.mode == "train":
        train(args.generations, args.games, args.resume)
    elif args.mode == "play":
        play(args.depth, args.color)
    else:
        print("AI vs MaterialBot: %d wins, %d draws, %d losses" % arena(Net.load(), 1, args.depth, watch=True))
