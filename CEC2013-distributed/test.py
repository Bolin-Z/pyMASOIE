import ray
from CEC2013 import CEC2013
from MASOIE import *
import matplotlib.pyplot as plt
import numpy as np

def run(funcID:str):
    ray.init()
    
    problem = CEC2013(funcID)

    numberOfAgents = problem.getGroupNum()
    dimension = problem.getDimension()
    lowerBound = problem.getMinX()
    upperBound = problem.getMaxX()
    maxLocalEvaluate = 1000000

    swarmSize = 300
    learningInterval = 4
    psi = 1
    phi = 0.5

    netWorkGraph = problem.getNetworkGraph()
    neighborsIDList = [[] for _ in range(numberOfAgents)]
    neighborsWeightList = [[] for _ in range(numberOfAgents)]
    for i in range(numberOfAgents):
        for j in range(numberOfAgents):
            if i!=j and netWorkGraph[i][j] != 0:
                neighborsIDList[i].append(j)
                neighborsWeightList[i].append(netWorkGraph[i][j])
    
    [MessageBuffer.options(name="MessageBuffer" + str(id)).remote(id) for id in range(numberOfAgents)]
    agents = [Agent.remote(
        dimension,
        lowerBound,
        upperBound,
        maxLocalEvaluate,
        funcID,
        swarmSize,
        learningInterval,
        psi,
        phi,
        i,
        neighborsIDList[i],
        neighborsWeightList[i]
    ) for i in range(numberOfAgents)]

    archive:dict[str, list[int|float]] = dict()
    archive["TotalEvaluationTimes"] = []
    archive["Fitness"] = []
    archive["Disagreement"] = []

    totalEvaluationTimes:int = 0
    bestFitness = float('inf')

    while totalEvaluationTimes < maxLocalEvaluate * numberOfAgents:
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
        
        solution = [sum([p[d] for p in positions]) / numberOfAgents for d in range(dimension)]
        meanVec = [sum([p[d] for p in meanPositions]) / numberOfAgents for d in range(dimension)]
        dvs = sum([computeDistance(meanPositions[n], meanVec) for n in range(numberOfAgents)])
        bestFitness = min(bestFitness, problem.global_eva(solution))

        meanCounter = localEvaluateCounter / numberOfAgents
        totalEvaluationTimes += localEvaluateCounter
        totalEvaluationTimes += numberOfAgents

        print(f"TotalEva: {totalEvaluationTimes} (mean: {meanCounter})")
        print(f"Fitness:  {'{:g}'.format(bestFitness)}")
        print(f"Disagree: {dvs}")

        archive["TotalEvaluationTimes"].append(totalEvaluationTimes)
        archive["Fitness"].append(bestFitness)
        archive["Disagreement"].append(dvs)

    ray.shutdown()

    return archive

if __name__ == "__main__":
    funcID = "100D20n3dheter-4"
    repeatTimes = 5

    archives = []
    for ep in range(repeatTimes):
        print(f"{'-' * 20} START[{ep}] {'-' * 20}")
        archives.append(run(funcID))
        print(f"{'-' * 20} FINISH[{ep}] {'-' * 20}")
    
    meanResult = dict()
    meanResult["TotalEvaluationTimes"] = []
    meanResult["Fitness"] = []
    meanResult["Disagreement"] = []

    sampleIndex = 0
    while True:
        counter = 0
        TotalEvaluationTimes = 0
        Fitness = 0
        Disagreement = 0
        for i in range(repeatTimes):
            if sampleIndex < len(archives[i]["TotalEvaluationTimes"]):
                counter += 1
                TotalEvaluationTimes += archives[i]["TotalEvaluationTimes"][sampleIndex]
                Fitness += archives[i]["Fitness"][sampleIndex]
                Disagreement += archives[i]["Disagreement"][sampleIndex]
        if counter == 0:
            break

        meanResult["TotalEvaluationTimes"].append(TotalEvaluationTimes / counter)
        meanResult["Fitness"].append(Fitness / counter)
        meanResult["Disagreement"].append(Disagreement / counter)

        sampleIndex += 1
    
    BestArchiveIndex = 0
    WorstArchiveIndex = 0
    for i in range(repeatTimes):
        if archives[i]["Fitness"][-1] < archives[BestArchiveIndex]["Fitness"][-1]:
            BestArchiveIndex = i
        if archives[i]["Fitness"][-1] > archives[WorstArchiveIndex]["Fitness"][-1]:
            WorstArchiveIndex = i
    
    fig, axs = plt.subplots(nrows=2, ncols=1)
    
    axs[0].set_xlabel('TotalEvaluations')
    axs[0].set_ylabel("Fitness")
    axs[0].plot(np.array(meanResult["TotalEvaluationTimes"]), np.array(meanResult["Fitness"]), '#ff9900', label="mean")
    axs[0].plot(np.array(archives[BestArchiveIndex]["TotalEvaluationTimes"]), np.array(archives[BestArchiveIndex]["Fitness"]), '#6aa84f', label="best")
    axs[0].plot(np.array(archives[WorstArchiveIndex]["TotalEvaluationTimes"]), np.array(archives[WorstArchiveIndex]["Fitness"]), '#4a86e8', label="worst")
    axs[0].legend()

    axs[1].set_xlabel('TotalEvaluations')
    axs[1].set_ylabel("Disagreement")
    axs[1].plot(np.array(meanResult["TotalEvaluationTimes"]), np.array(meanResult["Disagreement"]), '#ff9900', label="mean")
    axs[1].plot(np.array(archives[BestArchiveIndex]["TotalEvaluationTimes"]), np.array(archives[BestArchiveIndex]["Disagreement"]), '#6aa84f', label="best")
    axs[1].plot(np.array(archives[WorstArchiveIndex]["TotalEvaluationTimes"]), np.array(archives[WorstArchiveIndex]["Disagreement"]), '#4a86e8', label="worst")
    axs[1].legend()

    plt.show()