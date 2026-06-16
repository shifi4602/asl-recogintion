from enum import Enum


class SignClass(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    H = "H"
    I = "I"
    J = "J"
    K = "K"
    L = "L"
    M = "M"
    N = "N"
    O = "O"
    P = "P"
    Q = "Q"
    R = "R"
    S = "S"
    T = "T"
    U = "U"
    V = "V"
    W = "W"
    X = "X"
    Y = "Y"
    Z = "Z"
    SPACE = "space"
    DELETE = "del"
    NOTHING = "nothing"

    @classmethod
    def label_list(cls) -> list[str]:
        return [member.value for member in cls]

    @classmethod
    def from_index(cls, index: int) -> "SignClass":
        members = list(cls)
        if index < 0 or index >= len(members):
            raise ValueError(f"Index {index} out of range for SignClass (0–{len(members) - 1})")
        return members[index]

    @classmethod
    def num_classes(cls) -> int:
        return len(cls)
