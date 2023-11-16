from typing import Callable
from math import sqrt

class LocalEvaluator:
    def __init__(self, id:int, maxEvaluate:int, localEvaluate:Callable[[list[float], int], float]) -> None:
        self.id = id
        self.f = localEvaluate
        self.evaluateCounter = 0
        self.maxEvaluate = maxEvaluate

    def __call__(self, x:list[float]) -> float:
        self.evaluateCounter += 1
        return self.f(x, self.id)
    
    def reachMaxEvaluate(self) -> bool:
        return self.evaluateCounter > self.maxEvaluate

def computeDistance(a:list[float], b:list[float]) -> float:
    dimension = len(a)
    return sqrt(sum([(a[d] - b[d]) ** 2 for d in range(dimension)]))