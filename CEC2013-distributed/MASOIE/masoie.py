import ray
from CEC2013 import CEC2013
from util2 import computeDistance
from agent2 import Agent

def masoie(funcID:str):
    ray.init()

    problem = CEC2013(funcID)

    # parameter setting
    numberOfAgents = problem.getGroupNum()
    dimension = problem.getDimension()
    lowerBound = problem.getMinX()
    upperBound = problem.getMaxX()
    maxLocalEvaluate = 1000000

    swarmSize = 300
    learningInterval = 4
    psi = 1
    phi = 0.5

    # fixed dllso
    fixed = True

    # topology
    netWorkGraph = problem.getNetworkGraph()
    neighborsIDList = [[] for _ in range(numberOfAgents)]
    neighborsWeightList = [[] for _ in range(numberOfAgents)]
    for i in range(numberOfAgents):
        for j in range(numberOfAgents):
            if i!=j and netWorkGraph[i][j] != 0:
                neighborsIDList[i].append(j)
                neighborsWeightList[i].append(netWorkGraph[i][j])

    # creat agents
    agents = [Agent.options(name="Agent"+str(id)).remote(
        id, 
        numberOfAgents,
        neighborsIDList[id],
        neighborsWeightList[id],
        dimension,
        lowerBound,
        upperBound,
        maxLocalEvaluate,
        funcID,
        swarmSize,
        psi,
        learningInterval,
        phi,
        fixed
    ) for id in range(numberOfAgents)]

    archive:dict[str, list[int|float]] = dict()
    archive["TotalEvaluationTimes"] = []
    archive["Fitness"] = []
    archive["Disagreement"] = []

    totalEvaluationTimes:int = 0
    globalEvaluateCounter = 0
    bestFitness = float('inf')

    # set neighbours
    getNeighborsHandler(agents)
    # creatRoutingTable
    creatRoutingTable(agents)

    # main loop
    while totalEvaluationTimes < maxLocalEvaluate * numberOfAgents:
        internalLearning(agents)
        externalLearning(agents)
        adaptiveInterval(agents)
        # record running result
        tasks = [agent.getRecord.remote() for agent in agents]
        localEvaluateCounter = 0
        positions:list[list[float]] = []
        meanPositions:list[list[float]] = []
        while tasks:
            readyRefs, tasks = ray.wait(tasks)
            for ref in readyRefs:
                counter, position, meanPosition = ray.get(ref)
                localEvaluateCounter += counter
                positions.append(position)
                meanPositions.append(meanPosition)
        
        solution = [sum([p[d] for p in positions]) / numberOfAgents for d in range(dimension)]
        meanVec = [sum([p[d] for p in meanPositions]) / numberOfAgents for d in range(dimension)]
        dvs = sum([computeDistance(meanPositions[n], meanVec) for n in range(numberOfAgents)])
        bestFitness = min(bestFitness, problem.global_eva(solution))

        meanCounter = localEvaluateCounter / numberOfAgents
        globalEvaluateCounter += numberOfAgents
        totalEvaluationTimes = localEvaluateCounter + globalEvaluateCounter

        print(f"TotalEva: {totalEvaluationTimes} (mean: {meanCounter})")
        print(f"Fitness:  {'{:g}'.format(bestFitness)}")
        print(f"Disagree: {dvs}")

        archive["TotalEvaluationTimes"].append(totalEvaluationTimes)
        archive["Fitness"].append(bestFitness)
        archive["Disagreement"].append(dvs)
    
    ray.shutdown()

    return archive

def getNeighborsHandler(agents):
    [agent.getNeighborsHandler.remote() for agent in agents]

def creatRoutingTable(agents):
    _creatRoutingTableSendMsg(agents)
    _creatRoutingTableProcessMsg(agents)
    _creatRoutingTableCompute(agents)

def internalLearning(agents):
    [agent.internalLearning.remote() for agent in agents]

def externalLearning(agents):
    _externalLearningSendMsg(agents)
    _externalLearningProcess(agents)

def adaptiveInterval(agents):
    _adaptiveIntervalSendMsg(agents)
    _adaptiveIntervalProcess(agents)

# auxiliary function
def _creatRoutingTableSendMsg(agents):
    tasks = [agent.creatRoutingTableSendMsg.remote() for agent in agents]
    while tasks:
        _, tasks = ray.wait(tasks)

def _creatRoutingTableProcessMsg(agents):
    finished = False
    while not finished:
        finished = True
        tasks = [agent.creatRoutingTableProcessMsg.remote() for agent in agents]
        while tasks:
            readyRefs, tasks = ray.wait(tasks)
            for ref in readyRefs:
                if ray.get(ref):
                    finished = False

def _creatRoutingTableCompute(agents):
    tasks = [agent.creatRoutingTableCompute.remote() for agent in agents]
    while tasks:
        _, tasks = ray.wait(tasks)

def _externalLearningSendMsg(agents):
    tasks = [agent.externalLearningSendMsg.remote() for agent in agents]
    while tasks:
        _, tasks = ray.wait(tasks)

def _externalLearningProcess(agents):
    tasks = [agent.externalLearningProcess.remote() for agent in agents]
    while tasks:
        _, tasks = ray.wait(tasks)

def _adaptiveIntervalSendMsg(agents):
    tasks = [agent.adaptiveIntervalSendMsg.remote() for agent in agents]
    while tasks:
        _, tasks = ray.wait(tasks)

def _adaptiveIntervalProcess(agents):
    tasks = [agent.adaptiveIntervalProcess.remote() for agent in agents]
    while tasks:
        _, tasks = ray.wait(tasks)
