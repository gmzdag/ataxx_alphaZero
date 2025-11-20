"""
Ataxx NN Demo
=============
Bu betik, Ataxx oyununda sinir agi (NNetWrapper) kullanilarak
polika/v deger kestirimi ve otomatik hamle secimini gosteren hafif bir
ornek sunar. Varsayilan olarak egitimsiz (rastgele) agirliklarla calisir
ancak istege bagli olarak bir checkpoint yuklenebilir.
"""

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ataxx.AtaxxGame import AtaxxGame  # noqa: E402
from ataxx.pytorch.NNet import NNetWrapper  # noqa: E402


def decode_action(n: int, action: int) -> Tuple[int, int, int, int]:
    return tuple(int(x) for x in np.unravel_index(action, (n, n, n, n)))


def describe_board(board: np.ndarray) -> str:
    symbols = {1: "X", -1: "O", 0: "."}
    lines = [" ".join(symbols[int(val)] for val in row) for row in board]
    return "\n".join(lines)


def select_action_with_nn(
    game: AtaxxGame,
    nnet: NNetWrapper,
    board: np.ndarray,
    player: int,
    sample: bool,
) -> Tuple[Optional[int], float]:
    canon = game.getCanonicalForm(board, player)
    pi, value = nnet.predict(canon)
    valids = game.getValidMoves(board, player)

    masked = pi * valids
    s = np.sum(masked)
    if s <= 0:
        if np.sum(valids) == 0:
            return None, value
        masked = valids.astype(np.float32)
        s = np.sum(masked)
    probs = masked / s
    if sample:
        action = int(np.random.choice(len(probs), p=probs))
    else:
        action = int(np.argmax(probs))
    return action, value


def run_demo(
    size: int,
    max_moves: int,
    timer_limit: float,
    sample: bool,
    checkpoint: Optional[str],
):
    game = AtaxxGame(n=size, timer_limit=timer_limit)
    nnet = NNetWrapper(game)
    if checkpoint:
        nnet.load_checkpoint(*os.path.split(checkpoint))

    board = game.getInitBoard()
    player = 1

    start = time.time()
    print("Ataxx NN demo basliyor\n")
    print(describe_board(board))

    for move_idx in range(1, max_moves + 1):
        result = game.getGameEnded(board, player)
        if result != 0:
            break

        action, value = select_action_with_nn(game, nnet, board, player, sample)
        if action is None:
            print("Hamle bulunamadi, oyun bitiyor.")
            break

        x, y, nx, ny = decode_action(game.n, action)
        acting_player = player
        board, player, timers = game.getNextState(board, acting_player, action, elapsed=0.0)
        perspective = "+1" if acting_player == 1 else "-1"
        print(
            f"\nHamle {move_idx}: Oyuncu {perspective} -> "
            f"({x},{y}) -> ({nx},{ny}) | NN v={value:+.3f}"
        )
        print(f"Sureler | P1: {timers[1]:.2f}s | P2: {timers[-1]:.2f}s")
        print(describe_board(board))
    else:
        print(f"\nUyari: {max_moves} hamlede oyun tamamlanamadi.")

    final = game.getGameEnded(board, player)
    if final > 0:
        outcome = "P1 kazandi"
    elif final < 0:
        outcome = "P2 kazandi"
    elif np.isclose(final, 1e-4):
        outcome = "Berabere"
    else:
        outcome = "Oyun devam ediyor"

    elapsed = time.time() - start
    print("\nDemo tamamlandi")
    print(f"Sonuc: {outcome} (getGameEnded -> {final})")
    print(f"Toplam sure: {elapsed:.2f}s")


def parse_args():
    parser = argparse.ArgumentParser(description="Ataxx NN demo betigi")
    parser.add_argument("--size", type=int, default=7, help="Tahta boyutu")
    parser.add_argument("--moves", type=int, default=40, help="Maksimum hamle sayisi")
    parser.add_argument("--timer", type=float, default=30.0, help="Oyuncu basina sure")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Policy dagilimindan ornekleme yap (varsayilan argmax)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Egitimli agirlik dosyasi (folder/filename.pth.tar)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(args.size, args.moves, args.timer, args.sample, args.checkpoint)

