from typing import Callable
from math import sqrt

class LocalEvaluator:
    def __init__(self, id:int, localEvaluate:Callable[[list[float], int], float]) -> None:
        self.id = id
        self.f = localEvaluate
        self.evaluateCounter = 0

    def __call__(self, x:list[float]) -> float:
        self.evaluateCounter += 1
        return self.f(x, self.id)

def computeDistance(a:list[float], b:list[float]) -> float:
    dimension = len(a)
    return sqrt(sum([(a[d] - b[d]) ** 2 for d in range(dimension)]))