import math


def UpdateMean(OldMean: float, NewDataValue: float, n: int, A: list[float]) -> float:
    return (OldMean * n + NewDataValue) / (n + 1)


def UpdateMedian(OldMedian: float, NewDataValue: float, n: int, A: list[float]) -> float:
    if n % 2 == 0:
        if NewDataValue >= A[n // 2]:
            return A[n // 2]
        elif NewDataValue <= A[n // 2 - 1]:
            return A[n // 2 - 1]
        else:
            return NewDataValue
    else:
        if NewDataValue >= A[n // 2 + 1]:
            return (A[n // 2] + A[n // 2 + 1]) / 2
        elif NewDataValue <= A[n // 2 - 1]:
            return (A[n // 2 - 1] + A[n // 2]) / 2
        else:
            return (A[n // 2] + NewDataValue) / 2


def UpdateStd(OldMean: float, OldStd: float, NewMean: float, NewDataValue: float, n: int, A: list[float]) -> float:
    return math.sqrt(
        (n * pow(OldMean, 2) + (n - 1) * pow(OldStd, 2) + pow(NewDataValue, 2) - (n + 1) * pow(NewMean, 2)) / n)
