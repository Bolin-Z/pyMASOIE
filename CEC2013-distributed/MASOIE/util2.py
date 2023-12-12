from typing import Callable
from functools import total_ordering
from math import sqrt
from ray.actor import ActorHandle

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

@total_ordering
class Particle:
    def __init__(self, dimension:int) -> None:
        self._dimension = dimension
        self.position = [0.0 for _ in range(self._dimension)]
        self.internalVelocity = [0.0 for _ in range(self._dimension)]
        self.externalVelocity = [0.0 for _ in range(self._dimension)]
        self.fitness = float('inf')
    
    def __lt__(self, __o: "Particle") -> bool:
        return self.fitness < __o.fitness

    def __eq__(self, __o: "Particle") -> bool:
        return self.fitness == __o.fitness

class BaseMessage:
    def __init__(self, source:int, destination:int, tag:int) -> None:
        self.src = source
        self.dst = destination
        self.tag = tag

class NeighborInfo:
    def __init__(self, id:int, handle:ActorHandle, weight:float) -> None:
        self.id = id
        self.handle = handle
        self.weight = weight

def computeDistance(a:list[float], b:list[float]) -> float:
    dimension = len(a)
    return sqrt(sum([(a[d] - b[d]) ** 2 for d in range(dimension)]))