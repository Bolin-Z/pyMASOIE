from typing import Callable
from MASOIE.CEC2013 import CEC2013

class LocalEvaluator:
    def __init__(self, id:int, localEvaluate:Callable[[list[float], int], float]) -> None:
        self.id = id
        self.f = localEvaluate
    def __call__(self, x:list[float]) -> float:
        return self.f(x, self.id)

if __name__ == "__main__":
    problem = CEC2013("50D20n1d-6")