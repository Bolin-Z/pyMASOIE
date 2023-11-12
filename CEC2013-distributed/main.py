import ray, time
from typing import Callable
from CEC2013 import CEC2013
from MASOIE.agent import Agent

class LocalEvaluator:
    def __init__(self, id:int, localEvaluate:Callable[[list[float], int], float]) -> None:
        self.id = id
        self.f = localEvaluate
        self.evaluateCounter = 0

    def __call__(self, x:list[float]) -> float:
        self.evaluateCounter += 1
        return self.f(x, self.id)

if __name__ == "__main__":
    problem = CEC2013("100D20n3dheter-4")

    numberOfAgents =  problem.getGroupNum()
    dimension = problem.getDimension()
    lowerBound = problem.getMinX()
    upperBound = problem.getMaxX()

    swarmSize = 300
    learningInterval = 4
    psi = 1
    phi = 0.5
    
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
        upperBound,
        localEvaluators[i],
        swarmSize,
        learningInterval,
        psi,
        phi,
        i,
        neighborsIDList[i],
        neighborsWeightList[i]
    ) for i in range(numberOfAgents)]

    time.sleep(5)