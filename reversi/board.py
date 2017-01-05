import numpy as np
from .color import Color
from .traverse import traverse

Board = np.ndarray


def init() -> Board:
    """
    盤面をゲームの初期状態に初期化する
    :return: (Board) 初期化された盤面
    """

    b = np.zeros((2, 8, 8), dtype=np.bool)
    b[Color.White, 3, 3] = True
    b[Color.White, 4, 4] = True
    b[Color.Black, 3, 4] = True
    b[Color.Black, 4, 3] = True
    return b


def stringify(b: Board) -> str:
    """
    盤面を文字列化する
    :param b: (Board) 盤面
    :return: (str) 整形済み文字列
    """
    result = ''
    result += '    0   1   2   3   4   5   6   7 \n'
    result += '  ┌───┬───┬───┬───┬───┬───┬───┬───┐\n'

    for x in range(0, 8):
        result += '{} │'.format(x)
        for y in range(0, 8):
            if b[Color.Black, x, y]:
                result += ' ● '

            elif b[Color.White, x, y]:
                result += ' ○ '

            else:
                result += '   '

            result += '│'

        result += '\n'

        if x < 7:
            result += '  ├───┼───┼───┼───┼───┼───┼───┼───┤\n'

        else:
            result += '  └───┴───┴───┴───┴───┴───┴───┴───┘'

    return result


def get_reversible_count(b: Board, color: Color, x: int, y: int) -> int:
    """
    盤面bの座標(x,y)に色colorの石をおいたときに裏返せる石の数

    :param b: (Board) 盤面
    :param color: (Color) 置く石の色
    :param x: (int) 座標x
    :param y: (int) 座標y
    :return: (int) 裏返せる石の数
    """

    if b[0, x, y] or b[1, x, y]:
        return 0

    result = 0

    for dx, dy in (
            (-1, -1),
            (-1, +0),
            (-1, +1),
            (+0, -1),
            (+0, +1),
            (+1, -1),
            (+1, +0),
            (+1, +1)):

        px = x + dx
        py = y + dy
        tmp = 0

        while 0 <= px < 8 and 0 <= py < 8:
            # noinspection PyTypeChecker
            if not b[1 - color, px, py]:
                if b[color, px, py]:
                    result += tmp

                break

            tmp += 1
            px += dx
            py += dy

    return result


def is_valid(b: Board, color: Color, x: int, y: int):
    if b[0, x, y] or b[1, x, y]:
        return False

    if get_reversible_count(b, color, x, y) == 0:
        return False

    return True


def put(b: Board, c: Color, x: int, y: int):
    """
    盤面bの座標(x,y)に色colorの石を置く

    :param b: (Board) 盤面
    :param c: (Color) 置く石の色
    :param x: (int) 座標x
    :param y: (int) 座標y
    :return: (Board) 更新後の盤面
    """

    result = b.copy()

    if b[0, x, y] or b[1, x, y]:
        return result

    for dx, dy in (
            (-1, -1),
            (-1, +0),
            (-1, +1),
            (+0, -1),
            (+0, +1),
            (+1, -1),
            (+1, +0),
            (+1, +1)):

        px = x + dx
        py = y + dy

        while 0 <= px < 8 and 0 <= py < 8:
            # noinspection PyTypeChecker
            if not b[1 - c, px, py]:
                if b[c, px, py]:
                    while px != x or py != y:
                        px -= dx
                        py -= dy
                        result[c, px, py] = True
                        # noinspection PyTypeChecker
                        result[1 - c, px, py] = False

                break

            px += dx
            py += dy

    return result


def to_state(b, color, n):
    res = np.empty((6, 8, 8), dtype=np.float32)

    # 黒石/白石の位置
    res[0:2, :, :] = b.astype(np.float32)

    # 有効手かどうか, 裏返せる石の数
    for x in range(8):
        for y in range(8):
            if is_valid(b, color, x, y):
                res[2, x, y] = 1
                res[5, x, y] = get_reversible_count(b, color, x, y)

            else:
                res[2, x, y] = 0
                res[5, x, y] = 0

    # res[5, :, :] /= res[5, :, :].max()

    # ターン数
    res[3, :, :] = n / 64

    # 自分の色
    res[4, :, :] = color

    return res


class Games(object):
    def __init__(self, n):
        self.n = n
        self.t = 1
        self.board = np.zeros((n, 2, 8, 8), dtype=np.bool)
        self.multiple_init_game(n)

    def multiple_init_game(self, n):
        """
        n個の盤面をゲームの初期状態に初期化する
        :return: (Board) 初期化された盤面
        """

        self.board = np.zeros((n, 2, 8, 8), dtype=np.bool)

        print(Color.Black)
        self.board[:, Color.White, 3, 3] = True
        self.board[:, Color.White, 4, 4] = True
        self.board[:, Color.Black, 3, 4] = True
        self.board[:, Color.Black, 4, 3] = True

    def multiple_put(self, plies):
        color = Color.Black if self.t % 2 == 1 else Color.White  # type:Color
        for i in range(self.n):
            x = plies[i] // 8
            y = plies[i] % 8
            self.board[i] = put(self.board[i], color, x, y)

        # noinspection PyTypeChecker
        return to_state(self.board, 1 - color, self.t + 1)
