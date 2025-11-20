"""
Ataxx Mini Self-Play Training Demo
==================================
Bu betik, kucuk bir Ataxx tahtasinda (varsayilan 5x5) birkac self-play
episodu oynayip elde ettigi verilerle sinir agini hafifce gunceller.
Her hamlede MCTS'in urettigi politika dagilimini, ziyaret sayisini ve
Q degerlerini ayrintili sekilde yazdirarak karar surecini gozlemlemenizi
saglar. Egitimden sonra guncellenmis modelle bir deneme maci oynanir.
"""

import argparse
import os
import sys
from types import SimpleNamespace
from typing import List, Sequence, Tuple

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ataxx.AtaxxGame import AtaxxGame  # noqa: E402
from ataxx.pytorch import NNet as nnet_module  # noqa: E402
from ataxx.pytorch.NNet import NNetWrapper  # noqa: E402
from MCTS import MCTS  # noqa: E402


BoardState = np.ndarray
TrainingExample = Tuple[BoardState, np.ndarray, float]


def decode_action(n: int, action: int) -> Tuple[int, int, int, int]:
    return tuple(int(v) for v in np.unravel_index(action, (n, n, n, n)))


def format_board(board: np.ndarray) -> str:
    symbols = {1: "X", -1: "O", 0: "."}
    return "\n".join(" ".join(symbols[int(cell)] for cell in row) for row in board)


def extract_mcts_stats(mcts: MCTS, canonical_board: np.ndarray) -> Tuple[dict, dict]:
    s = mcts.game.stringRepresentation(canonical_board)
    visit_counts = {}
    q_values = {}
    for (state_key, action), count in mcts.Nsa.items():
        if state_key == s:
            visit_counts[action] = count
            q_values[action] = mcts.Qsa.get((state_key, action), 0.0)
    return visit_counts, q_values


def describe_top_actions(
    pi: Sequence[float],
    visit_counts: dict,
    q_values: dict,
    game: AtaxxGame,
    top_k: int,
) -> str:
    ranked_actions = np.argsort(pi)[::-1]
    lines = []
    displayed = 0
    for action in ranked_actions:
        prob = pi[action]
        if prob <= 0:
            continue
        x, y, nx, ny = decode_action(game.n, action)
        visits = visit_counts.get(action, 0)
        q = q_values.get(action, 0.0)
        lines.append(
            f"{displayed + 1}. ({x},{y})->({nx},{ny})  "
            f"prob={prob:.3f}  visits={visits}  Q={q:+.3f}"
        )
        displayed += 1
        if displayed >= top_k:
            break
    return "\n".join(lines) if lines else "Gecerli hamle bulunamadi."


def finalize_training_examples(
    game: AtaxxGame,
    final_board: np.ndarray,
    history: List[Tuple[np.ndarray, np.ndarray, int]],
) -> List[TrainingExample]:
    examples: List[TrainingExample] = []
    for canonical, pi, player in history:
        value = game.getGameEnded(final_board, player)
        examples.append((canonical, pi, float(value)))
    return examples


def play_logged_episode(
    game: AtaxxGame,
    nnet: NNetWrapper,
    mcts_args: SimpleNamespace,
    episode_label: str,
    max_moves: int,
    temp_switch_move: int,
    top_k: int,
    collect_examples: bool = True,
) -> Tuple[List[TrainingExample], float]:
    board = game.getInitBoard()
    player = 1
    mcts = MCTS(game, nnet, mcts_args)
    history: List[Tuple[np.ndarray, np.ndarray, int]] = []

    print(f"\n===== {episode_label} =====")
    print(format_board(board))

    for move_idx in range(1, max_moves + 1):
        canonical = game.getCanonicalForm(board, player)
        temp = 1 if move_idx < temp_switch_move else 0
        pi = np.array(mcts.getActionProb(canonical, temp=temp), dtype=np.float32)

        valids = game.getValidMoves(board, player)
        pi = pi * valids
        if np.sum(pi) <= 0:
            pi = valids.astype(np.float32)
        pi = pi / np.sum(pi)

        visit_counts, q_values = extract_mcts_stats(mcts, canonical)
        print(
            f"\n[Hamle {move_idx}] Oyuncu {'X' if player == 1 else 'O'} "
            f"(temp={temp})\n{format_board(board)}"
        )
        print(describe_top_actions(pi, visit_counts, q_values, game, top_k))

        if collect_examples:
            history.append((canonical.copy(), pi.copy(), player))

        action = int(np.random.choice(len(pi), p=pi))
        move = decode_action(game.n, action)
        print(
            f"Secilen hamle: ({move[0]},{move[1]}) -> ({move[2]},{move[3]}) | "
            f"olasilik={pi[action]:.3f}"
        )

        board, player, _ = game.getNextState(board, player, action, elapsed=0.0)
        result = game.getGameEnded(board, player)
        if result != 0:
            print(f"\nOyun sonucu: {result} (oyuncu {'X' if player == 1 else 'O'} perspektifi)")
            break
    else:
        print(f"\nUyari: {max_moves} hamlede oyun tamamlanamadi.")

    final_result = game.getGameEnded(board, player)
    examples = finalize_training_examples(game, board, history) if collect_examples else []
    return examples, final_result


def run_demo(args):
    game = AtaxxGame(n=args.size, timer_limit=args.timer)
    nnet = NNetWrapper(game)

    print("Baslangic agi degerlendirmesi (bos tahta):")
    init_board = game.getInitBoard()
    pi_before, v_before = nnet.predict(game.getCanonicalForm(init_board, 1))
    print(f"- Ilk hamle v tahmini: {v_before:+.3f}")

    mcts_args = SimpleNamespace(numMCTSSims=args.mcts_sims, cpuct=args.cpuct)
    all_examples: List[TrainingExample] = []
    for episode in range(1, args.episodes + 1):
        episode_examples, _ = play_logged_episode(
            game,
            nnet,
            mcts_args,
            episode_label=f"Self-Play Episode {episode}",
            max_moves=args.max_moves,
            temp_switch_move=args.temp_switch,
            top_k=args.top_k,
            collect_examples=True,
        )
        all_examples.extend(episode_examples)

    if not all_examples:
        print("Toplanan ornek yok, egitim atlanacak.")
        return

    # Mini egitim parametrelerini ayarla
    nnet_module.args.epochs = args.train_epochs
    nnet_module.args.batch_size = max(4, len(all_examples))
    print(
        f"\nToplam {len(all_examples)} ornekle egitim baslatiliyor "
        f"(epochs={nnet_module.args.epochs}, batch_size={nnet_module.args.batch_size})"
    )
    nnet.train(all_examples)

    pi_after, v_after = nnet.predict(game.getCanonicalForm(init_board, 1))
    print(
        f"\nEgitim sonrasi bos tahta degerlendirmesi: v={v_after:+.3f} "
        f"(once {v_before:+.3f})"
    )

    # Guncellenmis modelle tek bir gosterim maci
    play_logged_episode(
        game,
        nnet,
        mcts_args,
        episode_label="Egitim Sonrasi Gosterim",
        max_moves=args.max_moves,
        temp_switch_move=args.temp_switch,
        top_k=args.top_k,
        collect_examples=False,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Ataxx mini self-play egitim demosu")
    parser.add_argument("--size", type=int, default=5, help="Tahta boyutu")
    parser.add_argument("--episodes", type=int, default=2, help="Self-play episode sayisi")
    parser.add_argument("--max-moves", type=int, default=60, help="Oyun basina maksimum hamle")
    parser.add_argument("--temp-switch", type=int, default=10, help="Hamle sayisi temp=0'a dusme esigi")
    parser.add_argument("--mcts-sims", type=int, default=6, help="Her turda MCTS simulasyon sayisi")
    parser.add_argument("--cpuct", type=float, default=1.0, help="MCTS cpuct katsayisi")
    parser.add_argument("--top-k", type=int, default=5, help="Her adimda gosterilecek hamle sayisi")
    parser.add_argument("--train-epochs", type=int, default=1, help="Mini egitim epoch sayisi")
    parser.add_argument("--timer", type=float, default=60.0, help="Oyuncu basina sure limiti")
    return parser.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())

