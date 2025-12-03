#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import copy
from competitive_sudoku.sudoku import GameState, Move
import competitive_sudoku.sudokuai
from .rules import LocalOracle 

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
        oracle = LocalOracle(game_state.board, game_state.taboo_moves)
        allowed_squares = game_state.player_squares()
        
        # Generate legal moves 
        legal_move_tuples = oracle.get_legal_moves(allowed_squares)

        if not legal_move_tuples:
            return 

        legal_moves = [Move((r, c), val) for r, c, val in legal_move_tuples]

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

                score = self.minimax(next_state, depth - 1, alpha, beta, False)

                if score > best_score:
                    best_score = score
                    best_move = move

                alpha = max(alpha, best_score)
            
            if best_move:
                self.propose_move(best_move)
            
            depth += 1

    def minimax(self, game_state: GameState, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        # Base case
        if depth == 0:
            return self.evaluate_board(game_state)

        oracle = LocalOracle(game_state.board, game_state.taboo_moves)
        allowed = game_state.player_squares()
        
        # If no moves allowed, evaluate state immediately
        if not allowed:
             return self.evaluate_board(game_state)

        move_tuples = oracle.get_legal_moves(allowed)
        if not move_tuples:
            return self.evaluate_board(game_state)

        moves = [Move((r, c), val) for r, c, val in move_tuples]

        if is_maximizing:
            max_eval = float('-inf')
            for move in moves:
                next_state = copy.deepcopy(game_state)
                self.apply_move(next_state, move)
                
                eval_score = self.minimax(next_state, depth - 1, alpha, beta, False)
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
                
                eval_score = self.minimax(next_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def apply_move(self, game_state: GameState, move: Move):
        """
        Helper to apply a move to the GameState copy.
        NOTE: This does NOT update scores, only board and territory.
        """
        game_state.board.put(move.square, move.value)
        
        if game_state.current_player == 1:
            if game_state.occupied_squares1 is not None:
                game_state.occupied_squares1.append(move.square)
        else:
            if game_state.occupied_squares2 is not None:
                game_state.occupied_squares2.append(move.square)

        game_state.current_player = 3 - game_state.current_player

    def evaluate_board(self, game_state: GameState) -> float:
        """
        [ATA PUT HEURISTIC HERE] 
        
        For now i just use the difference in occupied squares

        Also when you write your own heuristic, make sure it does NOT use game_state.scores
        because it is not updated during minimax.
        """
        p1 = len(game_state.occupied_squares1) if game_state.occupied_squares1 else 0
        p2 = len(game_state.occupied_squares2) if game_state.occupied_squares2 else 0
        return p1 - p2