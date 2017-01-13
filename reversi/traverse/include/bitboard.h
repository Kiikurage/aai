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

static void find_next(BitBoardData data, const Color color, int *buf_x, int *buf_y, int *n_valid_hands) {
    // TODO bit演算で。
    *n_valid_hands = 0;

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
                        buf_x[*n_valid_hands] = i / 8;
                        buf_y[*n_valid_hands] = i % 8;
                        (*n_valid_hands)++;

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

static float evaluate_function_mode_win(Color start_color, Summary s) { return start_color == BLACK ? s.black - s.white : s.white - s.black; }

static float evaluate_function_mode_lose(Color start_color, Summary s) { return start_color == WHITE ? s.black - s.white : s.white - s.black; }

static float evaluate_function_mode_draw(Color start_color, Summary s) { return -(s.white - s.black) * (s.white - s.black); }

typedef enum {
    MONTECALRO_MODE_WIN = 1,
    MONTECALRO_MODE_LOSE = 2,
    MONTECALRO_MODE_DRAW = 3,
} MontecalroMode;

typedef struct {
    int best_x;
    int best_y;
    int best_win_count;
} MontecarloResult;

MontecarloResult montecalro(BitBoardData data,
                            const Color start_color,
                            const int num_branch,
                            const MontecalroMode mode) {

    float (*evaluate_function)(Color, Summary);
    switch (mode) {
        case MONTECALRO_MODE_WIN:
            evaluate_function = &evaluate_function_mode_win;
            break;

        case MONTECALRO_MODE_LOSE:
            evaluate_function = &evaluate_function_mode_lose;
            break;

        case MONTECALRO_MODE_DRAW:
        default:
            evaluate_function = &evaluate_function_mode_draw;
            break;
    }

    int *buf_x = (int *) malloc(sizeof(int) * 64);
    int *buf_y = (int *) malloc(sizeof(int) * 64);
    int n_valid_hands = 0;
    find_next(data, start_color, buf_x, buf_y, &n_valid_hands);

    if (n_valid_hands == 0) return (MontecarloResult) {-1, -1, 0};

    int *buf_win_count = (int *) calloc(sizeof(int), (size_t) n_valid_hands);
    int *buf_x2 = (int *) malloc(sizeof(int) * 64);
    int *buf_y2 = (int *) malloc(sizeof(int) * 64);

    for (int i_hand = 0; i_hand < n_valid_hands; i_hand++) {

        for (int i_branch = 0; i_branch < num_branch; i_branch++) {
            BitBoardData board = put_and_flip(data, start_color, buf_x[i_hand], buf_y[i_hand]);
            int pass_count = 0;
            Color current = other(start_color);
            while (1) {
                int n_valid_hands2 = 0;
                find_next(board, current, buf_x2, buf_y2, &n_valid_hands2);

                if (n_valid_hands2 == 0) {
                    pass_count++;
                    if (pass_count >= 2) break;

                    current = other(current);
                    continue;

                } else {
                    pass_count = 0;

                    const int selected_hand = xor128() % n_valid_hands2;
                    board = put_and_flip(board, current, buf_x2[selected_hand], buf_y2[selected_hand]);
                    current = other(current);
                }
            }

            Summary s = summarize(board);
            buf_win_count[i_hand] += evaluate_function(start_color, s);
        }
    }

    free(buf_x2);
    free(buf_y2);

    int best_win_count = buf_win_count[0];
    int best_x = buf_x[0];
    int best_y = buf_y[0];
    for (int i_hand = 1; i_hand < n_valid_hands; i_hand++) {
        if (best_win_count > buf_win_count[i_hand]) continue;

        best_win_count = buf_win_count[i_hand];
        best_x = buf_x[i_hand];
        best_y = buf_y[i_hand];
    }

    free(buf_x);
    free(buf_y);
    free(buf_win_count);

    return (MontecarloResult) {best_x, best_y, best_win_count};
};

void get_score_prob(BitBoardData data,
                    float *prob,
                    const Color start_color,
                    const int num_branch) {

    for (int i = 0; i < 127; i++) prob[i] = 0;

    int buf_x[64];
    int buf_y[64];
    int num_valid_hands = 0;

    find_next(data, start_color, buf_x, buf_y, &num_valid_hands);
    if (num_valid_hands == 0) return;

    int num_results = 0;

    for (int i_hand = 0; i_hand < num_valid_hands; i_hand++) {
        for (int i_branch = 0; i_branch < num_branch; i_branch++) {
            BitBoardData current_data = put_and_flip(data, start_color, buf_x[i_hand], buf_y[i_hand]);
            int pass_count = 0;
            Color current_color = other(start_color);

            while (1) {
                int buf_x2[64];
                int buf_y2[64];
                int num_valid_hands2 = 0;
                find_next(current_data, current_color, buf_x2, buf_y2, &num_valid_hands2);

                if (num_valid_hands2 == 0) {
                    pass_count++;
                    if (pass_count >= 2) break;

                    current_color = other(current_color);
                    continue;

                } else {
                    pass_count = 0;

                    const int selected_hand = xor128() % num_valid_hands2;
                    current_data = put_and_flip(current_data, current_color, buf_x2[selected_hand], buf_y2[selected_hand]);
                    current_color = other(current_color);
                }
            }

            Summary s = summarize(current_data);
            int delta = start_color == BLACK ? s.black - s.white : s.white - s.black;
            prob[delta + 64] += 1;
            num_results++;
        }
    }

    for (int i = 0; i < 127; i++) prob[i] /= num_results;
};
