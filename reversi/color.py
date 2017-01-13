class Color:
    Black = 0
    White = 1

    @classmethod
    def other(cls, color):
        return Color.White if color == Color.Black else Color.Black
