"""
AtaxxDisplay sınıfı, oyunun tahtasını terminalde okunabilir biçimde gösterir.
1, -1 ve 0 değerlerini sırasıyla X, O ve . sembolleriyle temsil eder.
"""
class AtaxxDisplay:
    symbols = {1: "X", -1: "O", 0: "."}

    @staticmethod
    def display(board):
        n = board.shape[0]
        hdr = "   " + " ".join([f"{i}" for i in range(n)])
        print(hdr)
        print("  " + "-" * (2*n-1))
        for i in range(n):
            row = " ".join(AtaxxDisplay.symbols[int(x)] for x in board[i])
            print(f"{i}| {row}")
