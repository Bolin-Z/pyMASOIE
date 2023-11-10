import ray
from typing import Callable
from CEC2013 import CEC2013
from MASOIE.agent import Agent

class LocalEvaluator:
    def __init__(self, id:int, localEvaluate:Callable[[list[float], int], float]) -> None:
        self.id = id
        self.f = localEvaluate
    def __call__(self, x:list[float]) -> float:
        return self.f(x, self.id)

if __name__ == "__main__":
    problem = CEC2013("50D20n1d-6")

    numberOfAgents =  problem.getGroupNum()
    dimension = problem.getDimension()
    lowerBound = problem.getMinX()
    upperBound = problem.getMaxX()
    
    localEvaluators = [LocalEvaluator(i, problem.local_eva) for i in range(numberOfAgents)]
    netWorkGraph = problem.getNetworkGraph()
    neighborsIDList = [[] for _ in range(numberOfAgents)]
    neighborsWeightList = [[] for _ in range(numberOfAgents)]
    for i in range(numberOfAgents):
        for j in range(numberOfAgents):
            if i!=j and netWorkGraph[i][j] != 0:
                neighborsIDList[i].append(j)
                neighborsWeightList[i].append(netWorkGraph[i][j])
    
    agents = [Agent.remote(
        dimension,
        lowerBound,
    ) for i in range(numberOfAgents)]