#include <xorshift.h>

typedef __m128 BitBoardData;

typedef struct {
    PyObject_HEAD
    BitBoardData data;
} BitBoard;

static BitBoardData put_and_flip(BitBoardData board, const Color color, const int x, const int y) {
    // TODO bit演算で。

    __m128 result = board;

    if (!BitCheck(&result, x, y, EMPTY)) return result;
    BitFSet(&result, x, y, color);

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;

            int px = x + dx;
            int py = y + dy;
            if (px < 0 || px >= 8 || py < 0 || py >= 8 || !BitCheck(&result, px, py, other(color))) continue;

            while (1) {
                px += dx;
                py += dy;

                if (px < 0 || px >= 8 || py < 0 || py >= 8) break;

                if (!BitCheck(&result, px, py, other(color))) {
                    if (BitCheck(&result, px, py, color)) {
                        while (1) {
                            px -= dx;
                            py -= dy;

                            if (px == x && py == y) break;

                            BitFSet(&result, px, py, color);
                        }
                    }
                    break;
                }
            }
        }
    }

    return result;
};

typedef struct {
    int n;
    int x[64];
    int y[64];
} Hands;

static void find_next(BitBoardData data, const Color color, Hands *hands) {
    // TODO bit演算で。
    hands->n = 0;

    for (int i = 0; i < 64; i++) {
        const int x = i / 8;
        const int y = i % 8;
        if (!BitCheck(&data, x, y, EMPTY)) continue;

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                if (dx == 0 && dy == 0) continue;

                int px = x + dx;
                int py = y + dy;
                if (px < 0 || px >= 8 || py < 0 || py >= 8 || !BitCheck(&data, px, py, other(color))) continue;

                while (1) {
                    px += dx;
                    py += dy;

                    if (px < 0 || px >= 8 || py < 0 || py >= 8) break;

                    if (BitCheck(&data, px, py, EMPTY)) break;

                    if (BitCheck(&data, px, py, color)) {
                        hands->x[hands->n] = i / 8;
                        hands->y[hands->n] = i % 8;
                        hands->n++;

                        goto next;
                    }
                }
            }
        }

        next:
        continue;
    }
};

typedef struct {
    int black;
    int white;
    int empty;
} Summary;

static Summary summarize(BitBoardData board) {
    Summary summary = (Summary) {0, 0, 0};

    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            if (BitCheck(&board, x, y, BLACK)) {
                summary.black++;
            } else if (BitCheck(&board, x, y, WHITE)) {
                summary.white++;
            } else {
                summary.empty++;
            }
        }
    }

    return summary;
};

typedef enum {
    SEARCH_MODE_WIN = 1,
    SEARCH_MODE_LOSE = 2,
    SEARCH_MODE_DRAW = 3,
} SearchMode;

static float evaluate_function_mode_win(Color start_color, Summary summary) { return start_color == BLACK ? summary.black - summary.white : summary.white - summary.black; }

static float evaluate_function_mode_lose(Color start_color, Summary summary) { return start_color == WHITE ? summary.black - summary.white : summary.white - summary.black; }

static float evaluate_function_mode_draw(Color start_color, Summary summary) { return -(summary.white - summary.black) * (summary.white - summary.black); }

static float evaluate(Color start_color, BitBoardData data, SearchMode mode) {
    switch (mode) {
        case SEARCH_MODE_WIN:
            return evaluate_function_mode_win(start_color, summarize(data));

        case SEARCH_MODE_LOSE:
            return evaluate_function_mode_lose(start_color, summarize(data));

        case SEARCH_MODE_DRAW:
        default:
            return evaluate_function_mode_draw(start_color, summarize(data));
    }
}

typedef struct {
    int x;
    int y;
    float score;
} SearchResult;

SearchResult montecalro_search(BitBoardData start_data,
                               const Color start_color,
                               const int num_branch,
                               const SearchMode mode) {
    Hands hands1;
    find_next(start_data, start_color, &hands1);

    if (hands1.n == 0) return (SearchResult) {-1, -1, -1};

    float max_min_score = -1e9f;

    int best_x = hands1.x[0];
    int best_y = hands1.y[0];

    for (int i1 = 0; i1 < hands1.n; i1++) {
#pragma omp parallel for
        for (int i_branch = 0; i_branch < num_branch; i_branch++) {
            BitBoardData current_data = put_and_flip(start_data, start_color, hands1.x[i1], hands1.y[i1]);
            int pass_count = 0;
            Color current_color = other(start_color);
            float min_score = 1e9;

            while (1) {
                Hands hands2;
                find_next(current_data, current_color, &hands2);

                if (hands2.n == 0) {
                    pass_count++;
                    if (pass_count >= 2) break;

                    current_color = other(current_color);
                    continue;

                } else {
                    pass_count = 0;

                    const int selected_hand = xor128() % hands2.n;
                    current_data = put_and_flip(current_data, current_color, hands2.x[selected_hand], hands2.y[selected_hand]);
                    current_color = other(current_color);
                }
            }

            float current_score = evaluate(start_color, current_data, mode);
            min_score = min_score < current_score ? min_score : current_score;

            if (min_score > max_min_score) {
                max_min_score = min_score;
                best_x = hands1.x[i1];
                best_y = hands1.y[i1];
            }
        }
    }

    return (SearchResult) {best_x, best_y, max_min_score};
};

SearchResult traverse_search(BitBoardData start_data,
                             const Color start_color,
                             const Color base_color,
                             const SearchMode mode,
                             int pass_count) {
    Hands hands1;
    find_next(start_data, start_color, &hands1);

    if (hands1.n == 0) {
        if (pass_count >= 1) {
            float score = evaluate(base_color, start_data, mode);
            return (SearchResult) {-1, -1, score};

        } else {
            return traverse_search(start_data, other(start_color), base_color, mode, 1);
        }
    }

    int best_x = hands1.x[0];
    int best_y = hands1.y[0];

    float *min_score = (float *) malloc(sizeof(float) * hands1.n);

#pragma omp parallel for
    for (int i1 = 0; i1 < hands1.n; i1++) {
        min_score[i1] = traverse_search(put_and_flip(start_data, start_color, hands1.x[i1], hands1.y[i1]),
                                        other(start_color), base_color,
                                        mode, 0).score;
    }

    float max_min_score = min_score[0];

    for (int i = 1; i < hands1.n; i++) {
        if (min_score[i] > max_min_score) {
            max_min_score = min_score[i];
            best_x = hands1.x[i];
            best_y = hands1.y[i];
        }
    }

    return (SearchResult) {best_x, best_y, max_min_score};
};

void get_score_prob(BitBoardData data,
                    float *prob,
                    const Color start_color,
                    const Color self_color,
                    int const num_branch) {

    for (int i = 0; i < 127; i++) prob[i] = 0;

    int num_valid_hands = 0;

    for (int i_branch = 0; i_branch < num_branch; i_branch++) {
        BitBoardData current_data = data;
        int pass_count = 0;
        Color current_color = start_color;

        while (1) {
            Hands hands;

            if (current_color == self_color) {

                //自分：ランダム
                find_next(current_data, current_color, &hands);
                if (hands.n == 0) {
                    pass_count++;
                    if (pass_count >= 2) break;

                } else {
                    pass_count = 0;

                    const int selected_hand = xor128() % num_valid_hands;
                    current_data = put_and_flip(current_data, current_color, hands.x[selected_hand], hands.y[selected_hand]);
                }

            } else {

                //相手：勝ち目標モンテカルロ
                SearchResult res = montecalro_search(current_data, current_color, 10, SEARCH_MODE_WIN);

                if (res.x == -1) {
                    pass_count++;
                    if (pass_count >= 2) break;

                } else {
                    pass_count = 0;

                    current_data = put_and_flip(current_data, current_color, res.x, res.y);
                }

            }

            current_color = other(current_color);
        }

        Summary s = summarize(current_data);
        int delta = self_color == BLACK ? s.black - s.white : s.white - s.black;
        prob[delta + 64] += 1;
    }

    for (int i = 0; i < 127; i++) prob[i] /= num_branch;
};

BitBoardData get_score_prob2(float *prob,
                             const Color self_color,
                             int const num_start_stone,
                             int const num_branch) {

    for (int i = 0; i < 127; i++) prob[i] = 0;

    int num_valid_hands = 0;
    BitBoardData current_data;
    Color current_color;

    BitBoardData BIT_BOARD_INITIAL;
    for (int i = 0; i < 4; i++) ((int *) &BIT_BOARD_INITIAL)[i] = 0xFFFFFFFF;
    BitFSet(&BIT_BOARD_INITIAL, 3, 3, WHITE);
    BitFSet(&BIT_BOARD_INITIAL, 4, 4, WHITE);
    BitFSet(&BIT_BOARD_INITIAL, 3, 4, BLACK);
    BitFSet(&BIT_BOARD_INITIAL, 4, 3, BLACK);

    while (1) {
        int pass_count = 0;
        int num_stone = 4;
        current_color = BLACK;
        current_data = BIT_BOARD_INITIAL;

        while (num_stone < num_start_stone) {
            Hands hands;
            find_next(current_data, current_color, &hands);

            if (hands.n == 0) {
                pass_count++;
                if (pass_count >= 2) break;

            } else {
                pass_count = 0;

                const int selected_hand = xor128() % num_valid_hands;
                current_data = put_and_flip(current_data, current_color, hands.x[selected_hand], hands.y[selected_hand]);
                num_stone++;
            }

            current_color = other(current_color);
        }

        if (num_stone >= num_start_stone) break;
    }

    get_score_prob(current_data, prob, current_color, self_color, num_branch);

    return current_data;
}