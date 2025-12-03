class LocalOracle:
    def __init__(self, board_object, taboo_moves):
        self.board = board_object
        self.N = board_object.N
        self.m = board_object.m
        self.n = board_object.n

        self.rows = [0] * self.N
        self.cols = [0] * self.N
        self.boxes = [0] * self.N

        self.taboo_set = set()
        if taboo_moves:
            for move in taboo_moves:
                r, c = move.square
                self.taboo_set.add((r, c, move.value))

        for i in range(self.N):
            for j in range(self.N):
                idx = self.N * i + j
                val = self.board.squares[idx]

                if val != 0:
                    self.mark_constraints(i, j, val)

    def get_box_index(self, r: int, c: int) -> int:
        br = r // self.m
        bc = c // self.n
        return br * self.m + bc

    def mark_constraints(self, r: int, c: int, val: int):
        mask = 1 << (val - 1)
        self.rows[r] |= mask
        self.cols[c] |= mask
        self.boxes[self.get_box_index(r, c)] |= mask

    def is_valid_sudoku_move(self, r: int, c: int, val: int) -> bool:
        mask = 1 << (val - 1)

        if self.rows[r] & mask:
            return False
        if self.cols[c] & mask:
            return False
        if self.boxes[self.get_box_index(r, c)] & mask:
            return False

        return True

    def get_legal_moves(self, allowed_squares):
        legal_moves = []

        for (r, c) in allowed_squares:
            if self.board.squares[self.N * r + c] != 0:
                continue

            for val in range(1, self.N + 1):
                if not self.is_valid_sudoku_move(r, c, val):
                    continue

                if (r, c, val) in self.taboo_set:
                    continue

                legal_moves.append((r, c, val))

        return legal_moves