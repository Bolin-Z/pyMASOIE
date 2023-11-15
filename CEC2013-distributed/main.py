import ray
from typing import Callable
from CEC2013 import CEC2013
from MASOIE.agent import Agent
from MASOIE.communicate import MessageBuffer
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

if __name__ == "__main__":
    ray.init()
    problemID = "100D20n3dheter-4"
    problem = CEC2013(problemID)

    numberOfAgents =  problem.getGroupNum()
    dimension = problem.getDimension()
    lowerBound = problem.getMinX()
    upperBound = problem.getMaxX()
    maxLocalEvaluate = 1000000

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

    buffers = [MessageBuffer.options(name="MessageBuffer" + str(id)).remote(id) for id in range(numberOfAgents)]
    agents = [Agent.remote(
        dimension,
        lowerBound,
        upperBound,
        maxLocalEvaluate,
        problemID,
        swarmSize,
        learningInterval,
        psi,
        phi,
        i,
        neighborsIDList[i],
        neighborsWeightList[i]
    ) for i in range(numberOfAgents)]

    meanCounter = 0
    bestFitness = float('inf')
    while meanCounter < maxLocalEvaluate:
        tasks = [agent.run.remote() for agent in agents]
        localEvaluateCounter = 0
        positions:list[list[float]] = []
        meanPositions:list[list[float]] = []
        while tasks:
            readys, tasks = ray.wait(tasks)
            for ref in readys:
                counter, position, meanPosition = ray.get(ref)
                localEvaluateCounter += counter
                positions.append(position)
                meanPositions.append(meanPosition)

        meanCounter = localEvaluateCounter / numberOfAgents
        solution = [sum([p[d] for p in positions]) / numberOfAgents for d in range(dimension)]
        meanVec = [sum([p[d] for p in meanPositions]) / numberOfAgents for d in range(dimension)]
        dvs = sum([computeDistance(meanPositions[n], meanVec) for n in range(numberOfAgents)])
        bestFitness = min(bestFitness, problem.global_eva(solution))
        print(f"meanEva: {meanCounter}\nFitness: {'{:g}'.format(bestFitness)}\ndvs:     {dvs}")