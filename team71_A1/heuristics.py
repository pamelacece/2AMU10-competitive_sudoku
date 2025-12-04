# heuristics.py

class Heuristics:
    """
    Stateless heuristics module for our agent
    Calculates (1) board potential based on sniping, (2) mobility, and (3) the least remaining values (LRV)"""
    @staticmethod
    def get_region_status(board, N, n, m):
        """
        Scan the board and return the status of all rows, columns and blocks
        :param board: 1D array of integers representing the grid
        :param N: total size of the grid (NxN)
        :param n: number of rows in a block (block height)
        :param m: number of columns in a block (block width)
        """

        # data structures to store region properties
        rows = [{'zeros': 0, 'empty_indices': [], 'used_mask': 0} for _ in range(N)]
        cols = [{'zeros': 0, 'empty_indices': [], 'used_mask': 0} for _ in range(N)]
        blocks = [{'zeros': 0, 'empty_indices': [], 'used_mask': 0} for _ in range(N)]

        # blocks per row (how many blocks fit horizontally) grid is N, block width is m
        blocks_per_row = N // m

        for r in range(N):
            for c in range(N):
                idx = r * N + c
                val = board[idx]

                # calculate block index
                b_row = r // n
                b_col = c // m
                b_idx = b_row * blocks_per_row + b_col

                if val == 0:
                    # update row
                    rows[r]['zeros'] += 1
                    rows[r]['empty_indices'].append((r, c))

                    # update col
                    cols[c]['zeros'] += 1
                    cols[c]['empty_indices'].append((r, c))

                    # update block
                    blocks[b_idx]['zeros'] += 1
                    blocks[b_idx]['empty_indices'].append((r, c))
                else:  # update bitmasks (for heuristic 3)
                    # shift 1 by (val-1) so: 1→bit0, 2→bit1,....
                    mask = 1 << (val - 1)
                    rows[r]['used_mask'] |= mask
                    cols[c]['used_mask'] |= mask
                    blocks[b_idx]['used_mask'] |= mask

        return {'rows': rows, 'cols': cols, 'blocks': blocks}


    @staticmethod
    def calculate_sniping_score(analysis, my_allowed, opp_allowed):  # heuristic 1
        """
        Analyzes the regions with exactly 1 empty cell and returns the score.
        :param analysis: output from get_region_status()
        :param my_allowed: list tuples (r, c) of allowed squares for our agent
        :param opp_allowed: list tuples (r, c) of allowed squares for the opponent
        """
        score = 0.0

        # ---- weights ---
        # TODO: carry on these weights into a .config file s.t. we can tune them easier
        W_SNIPE = 100.0 # reward for guaranteed point for me
        W_DEFEND = 120.0 # penalty for guaranteed point for opponent -- defense is prioritized

        # we convert the lists to sets for O(1) lookup
        my_set, opp_set = set(my_allowed), set(opp_allowed)

        all_regions = analysis['rows'] + analysis['cols'] + analysis['blocks']

        for region in all_regions:
            if region['zeros'] == 1: # this is a critical region → who can reach the last cell?
                target_cell = region['empty_indices'][0] # (r, c)

                can_i_reach = target_cell in my_set
                can_opp_reach = target_cell in opp_set

                if can_i_reach and not can_opp_reach: score += W_SNIPE  # opportunity for us
                elif can_opp_reach and not can_i_reach: score -= W_DEFEND  # threat from opp
                elif can_i_reach and can_opp_reach: score += (W_SNIPE / 2)
                # ↑ contest: both can claim the cell. it is better if we take it, but it is risky.
                # slight bonus to encourage taking it before the opponent though.

        return score


    @staticmethod
    def calculate_mobility_score(board, N, my_allowed, opp_allowed): # heuristic 2
        """
        The mobility heuristic: rewards having empty squares to play in than the opponent
        checks if the allowed square is actually empty (0) on the board
        """
        W_MOBILITY = 2.0 # TODO: move to config

        # count valid empty squares for us
        my_count = 0
        for (r, c) in my_allowed:
            idx = r * N + c

            if board[idx] == 0:
                my_count += 1

        # count valid empty squares for opponent
        opp_count = 0
        for (r, c) in opp_allowed:
            idx = r * N + c

            if board[idx] == 0:
                opp_count += 1

        diff = my_count - opp_count
        return W_MOBILITY * diff


    @staticmethod
    def _count_valid_options(used_mask, N):
        """Helper: count how many bits are 0 (valid) in the mask up to N"""
        count = 0
        for i in range(N):
            if not (used_mask & (1 << i)): # check if bit i is not set
                count += 1
        return count

    @staticmethod
    def calculate_lrv_score(board, analysis, N, n, m, my_allowed):
        score = 0.0
        blocks_per_row = N // m

        for (r, c) in my_allowed:
            if board[r * N + c] != 0: continue

            b_row = r // n
            b_col = c // m
            b_idx = b_row * blocks_per_row + b_col

            r_mask = analysis['rows'][r]['used_mask']
            c_mask = analysis['cols'][c]['used_mask']
            b_mask = analysis['blocks'][b_idx]['used_mask']

            total_mask = r_mask | c_mask | b_mask
            options = Heuristics._count_valid_options(total_mask, N)

            # TODO: move to config
            if options == 1:
                score += 5.0  # single (very safe lol)
            elif options == 2:
                score += 2.0  # strong move
            elif options > 2:
                # lower returns for risky moves
                score += (1.0 / options)
            else:
                # options == 0 (impossible state)
                score -= 10.0  # bad move to be here

        return score

    @staticmethod
    def evaluate_board(board, N, n, m, my_allowed, opp_allowed):
        """
        Combines all heuristics into a single value
        """
        # scanner
        analysis = Heuristics.get_region_status(board, N, n, m)

        # components
        # Note: sniping logic calculates score based on potential moves
        # we need to calculate current score diff if we had access to game_state.scores,
        # but since we are stateless, we assume the AI handles score diff via current_score
        # here we focus on future potential.

        sniping = Heuristics.calculate_sniping_score(analysis, my_allowed, opp_allowed)
        mobility = Heuristics.calculate_mobility_score(board, N, my_allowed, opp_allowed)
        lrv = Heuristics.calculate_lrv_score(board, analysis, N, n, m, my_allowed)

        total = sniping + mobility + lrv
        return total