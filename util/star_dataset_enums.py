from enum import Enum


class StarColor(Enum):
    BLUE_WHITE = 0
    YELLOW_WHITE = 1
    BLUE = 2
    ORANGE = 3
    WHITE = 4
    YELLOWISH = 5
    WHITE_YELLOW = 6
    RED = 7
    ORANGE_RED = 8
    PALE_YELLOW_ORANGE = 9

    # {'Blue White', 'Yellowish White', 'Blue', 'Orange', 'white', 'yellowish', 'Blue-White', 'Yellowish', 'Blue-white',
    # 'Blue white', 'yellow-white', 'White-Yellow', 'Red', 'Whitish', 'Color', 'Orange-Red', 'Pale yellow orange',
    # 'White'}


class SpectralClass(Enum):
    O = 0
    B = 1
    A = 2
    F = 3
    G = 4
    K = 5
    M = 6


class StarType(Enum):
    RED_DWARF = 0
    BROWN_DWARF = 1
    WHITE_DWARF = 2
    MAIN_SEQUENCE = 3
    SUPER_GIANTS = 4
    HYPER_GIANTS = 5
