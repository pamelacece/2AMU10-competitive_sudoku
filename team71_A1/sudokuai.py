#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import copy
from competitive_sudoku.sudoku import GameState, Move
import competitive_sudoku.sudokuai

from .rules import LocalOracle
from .heuristics import Heuristics

class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes the best move using Minimax with Alpha-Beta pruning
    and Iterative Deepening.
    """

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:
        """
        Use Iterative Deepening to search as deep as possible within the unknown time 
        limit.
        """

        my_player_id = game_state.current_player

        oracle = LocalOracle(game_state.board, game_state.taboo_moves)
        allowed_squares = game_state.player_squares()
        
        # Generate legal moves 
        legal_move_tuples = oracle.get_legal_moves(allowed_squares)

        if not legal_move_tuples:
            return 

        legal_moves = [Move((r, c), val) for r, c, val in legal_move_tuples]

        # sort: check scoring moves first (greedy fallback)
        legal_moves.sort(key=lambda m: self._get_immediate_points(game_state, m), reverse=True)


        # Propose the first valid move immediately as a fallback
        self.propose_move(legal_moves[0])

        # Iterative Deepening Loop
        depth = 1
        while True:
            best_move = None
            best_score = float('-inf')
            alpha = float('-inf')
            beta = float('inf')

            for move in legal_moves:
                # TODO: mplement make/undo move to avoid deepcopy
                next_state = copy.deepcopy(game_state)
                self.apply_move(next_state, move)

                score = self.minimax(next_state, depth - 1, alpha, beta, False, my_player_id)

                if score > best_score:
                    best_score = score
                    best_move = move

                alpha = max(alpha, best_score)

            if best_move:
                self.propose_move(best_move)

            depth += 1
            # print(f'AGENT depth has reached: {depth}')


    def minimax(self, game_state: GameState, depth: int, alpha: float, beta: float, is_maximizing: bool,
                my_player_id: int) -> float:

        if depth == 0:  # base case 1: reached max depth
            return self.evaluate_board(game_state, my_player_id)

        # base case 2: no moves possible (game over or boxed in)
        oracle = LocalOracle(game_state.board, game_state.taboo_moves)
        allowed = game_state.player_squares()
        
        # If no moves allowed, evaluate state immediately
        if not allowed:
            return self.evaluate_board(game_state, my_player_id)

        move_tuples = oracle.get_legal_moves(allowed)
        if not move_tuples:
            return self.evaluate_board(game_state, my_player_id)

        moves = [Move((r, c), val) for r, c, val in move_tuples]
        # optim: sort moves by immediate points to maximize a-b pruning efficiency
        moves.sort(key=lambda m: self._get_immediate_points(game_state, m), reverse=True)


        if is_maximizing:
            max_eval = float('-inf')
            for move in moves:
                next_state = copy.deepcopy(game_state)
                self.apply_move(next_state, move)
                
                eval_score = self.minimax(next_state, depth - 1, alpha, beta, False, my_player_id)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break 
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                next_state = copy.deepcopy(game_state)
                self.apply_move(next_state, move)
                
                eval_score = self.minimax(next_state, depth - 1, alpha, beta, True, my_player_id)

                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval


    def apply_move(self, game_state: GameState, move: Move):
        """
        Helper to apply a move to the game state copy updates board, territory, and scores
        """
        r, c = move.square
        val = move.value
        game_state.board.put(move.square, val)

        # update territory
        if game_state.current_player == 1:
            if game_state.occupied_squares1 is not None:
                game_state.occupied_squares1.append(move.square)
        else:
            if game_state.occupied_squares2 is not None:
                game_state.occupied_squares2.append(move.square)

        # calc points
        # since the value is already in the board, we pass is_hypothetical=False
        regions_completed = self._check_regions_completion(
            game_state.board, r, c, is_hypothetical=False
        )

        score_add = self._map_points_to_score(regions_completed)

        # update score
        player_idx = game_state.current_player - 1
        game_state.scores[player_idx] += score_add

        # switch turns
        game_state.current_player = 3 - game_state.current_player

    def _get_immediate_points(self, game_state: GameState, move: Move) -> int:
        """
        heuristic helper: calculates potential points for sorting
        """
        r, c = move.square
        # move is not on the board yet, so we pass is_hypothetical=True
        regions_completed = self._check_regions_completion(
            game_state.board, r, c, is_hypothetical=True
        )
        return self._map_points_to_score(regions_completed)

    def _check_regions_completion(self, board, r, c, is_hypothetical: bool) -> int:
        """
        shared logic to check how many regions (row, col, block) are completed by a move at (r,c)

        is_hypothetical:
            True = (r,c) is currently 0 in board.squares, but we pretend it's filled
            False = (r,c) is already filled in board.squares
        """
        N = board.N
        n = board.n
        m = board.m
        board_arr = board.squares
        points = 0

        target_idx = r * N + c

        # helper predicate to check if a cell is filled
        # if hypothetical, target_idx counts as filled even if 0
        def is_filled(idx):
            return board_arr[idx] != 0 or (is_hypothetical and idx == target_idx)

        # check row
        row_start = r * N
        if all(is_filled(i) for i in range(row_start, row_start + N)):
            points += 1

        # check col
        if all(is_filled(c + i * N) for i in range(N)):
            points += 1

        # check block
        b_row_start = (r // n) * n
        b_col_start = (c // m) * m

        block_full = True
        for bi in range(n):
            for bj in range(m):
                idx = (b_row_start + bi) * N + (b_col_start + bj)
                if not is_filled(idx):
                    block_full = False
                    break
            if not block_full: break

        if block_full:
            points += 1

        return points

    def _map_points_to_score(self, regions_count: int) -> int:
        """
        maps the scores to regions (1, 3, 7)
        """
        if regions_count == 1: return 1
        if regions_count == 2: return 3
        if regions_count == 3: return 7
        return 0

    def evaluate_board(self, game_state: GameState, my_player_id: int) -> float:
        """
        Evaluates the board state relative to the model's identity (my_player_id),
        uses the heuristics defined in Heuristics.
        """

        # 1 - calculate real score diff
        W_REAL_SCORE = 1000.0 # TODO: move to config

        my_idx = my_player_id - 1
        opp_idx = 1 - my_idx

        score_diff = (game_state.scores[my_idx] - game_state.scores[opp_idx]) * W_REAL_SCORE

        # 2 - prepare data for the heuristics class
        # note: heuristics needs to know the allowed squares for both players to calculate the threats
        # game_state.player_squares only gives squares for the current player
        # we perform a temporary flip to get both lists

        org_player = game_state.current_player

        # get p1 moves
        game_state.current_player = my_player_id
        p1_allowed = game_state.player_squares()

        # get p2 moves
        game_state.current_player = 3 - my_player_id # switch 1→2 or 2→1
        p2_allowed = game_state.player_squares()

        # restore
        game_state.current_player = org_player

        # 3 - calculate heuristic potential → p1 as us and p2 as opp → positive result means p1 has better potential
        heuristic_val = Heuristics.evaluate_board(
            board=game_state.board.squares,
            N=game_state.board.N,
            n=game_state.board.n,
            m=game_state.board.m,
            my_allowed=p1_allowed,
            opp_allowed=p2_allowed
        )

        # total eval
        return score_diff + heuristic_val