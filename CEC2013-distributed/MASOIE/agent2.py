import ray, random
from math import exp
from copy import deepcopy, copy
from typing import Callable

from CEC2013 import CEC2013
from util2 import Particle, BaseMessage, LocalEvaluator, NeighborInfo

class RoutingMessage(BaseMessage):
    def __init__(self, source: int, destination: int, nid:int, nlist:list[int]) -> None:
        super().__init__(source, destination, 1)
        self.nid = nid
        self.nlist = copy(nlist)

class ExternalLearningMsg(BaseMessage):
    def __init__(self, source: int, destination: int, positions:list[list[float]]) -> None:
        super().__init__(source, destination, 2)
        self.positions = deepcopy(positions)

class AdaptiveMessage(BaseMessage):
    def __init__(self, source: int, destination: int, flist:list[tuple[int, float|None]]) -> None:
        super().__init__(source, destination, 3)
        self.flist = flist

@ray.remote
class Agent:
    def __init__(
            self,
            # network information
            ID:int,
            numberOfAgents:int,
            neighborsID:list[int],
            neighborsWeight:list[float],
            # problem
            dimension:int,
            lowerBound:float,
            upperBound:float,
            maxLocalEvaluate:int,
            # evaluator
            # evaluate:Callable[[list[float]], float],
            funcID:str,
            # structure
            swarmSize:int,
            # external learning
            psi:float,
            # internal learning
            learningInterval:int,
            # dllso
            phi:float,
            fixed:bool
        ) -> None:
        
        # network information
        self.ID = ID
        self.numberOfAgents = numberOfAgents
        self.neighbors:dict[int, NeighborInfo] = dict()
        for n in range(len(neighborsID)):
            self.neighbors[neighborsID[n]] = NeighborInfo(neighborsID[n], None, neighborsWeight[n])
        self.routingTable:dict[int, set[int]] = dict()
        self.communicateList:dict[int, list[int]] = dict()
        for n in self.neighbors.keys():
            self.communicateList[n] = []
        
        # communication
        self.messageBuffer = []
        
        # problem
        self.dimension = dimension
        self.upperBound = upperBound
        self.lowerBound = lowerBound
        # self.f = LocalEvaluator(self.ID, maxLocalEvaluate, evaluate)
        self.problem = CEC2013(funcID)
        self.f = LocalEvaluator(self.ID, maxLocalEvaluate, self.problem.local_eva)
        
        # structure
        self.swarmSize = swarmSize

        # external learning
        self.psi = psi

        # internal learning dllso
        self.learningInterval = learningInterval
        self.minInverval = 2
        self.c = 0
        self.threshold = 10
        self.phi = phi

        self.fixed =fixed
        self.levelPoolSize = None
        self.levelPool = None
        self.levelPerformance:list[float] = []
        self.NL = None
        self.LS = None
        self.levelIndex = None
        if self.swarmSize >= 300:
            self.levelPoolSize = 6
            self.levelPool = [4, 6, 8, 10, 20, 50]
        elif self.swarmSize >= 20:
            self.levelPoolSize = 4
            self.levelPool = [4, 6, 8, 10]
        else:
            self.levelPoolSize = 2
            self.levelPool = [2, 3]
        for _ in range(self.levelPoolSize):
            self.levelPerformance.append(1.0)
        
        # adaptive interval
        self.fitnessList:list[float|None] = [None for _ in range(self.numberOfAgents)]

        # initiate swarm
        self.swarm:list[Particle] = []
        self.swarmOrdered:list[Particle] = []
        for _ in range(self.swarmSize):
            newParticle = Particle(self.dimention)
            for i in range(self.dimention):
                newParticle.position[i] = random.uniform(self.lowerBound, self.upperBound)
            newParticle.fitness = self.f(newParticle.position)
            self.swarm.append(newParticle)
            self.swarmOrdered.append(newParticle)
        self.swarm.sort()

        self.curFitness = self.swarm[0].fitness
        self.bestFitness = float('inf')

        # creat routing table related
        self.nlists:dict[int, list[int]] = dict()
        self.nlists[self.ID] = [n for n in self.neighbors.keys()]

    def appendMsg(self, msg):
        self.messageBuffer.append(msg)

    def getNeighborsHandler(self):
        for node in self.neighbors.values():
            node.handle = ray.get_actor("Agent" + str(node.id))
            
    def creatRoutingTableSendMsg(self):
        nlist = [n for n in self.neighbors.keys()]
        for node in self.neighbors.values():
            node.handle.appendMsg.remote(RoutingMessage(
                self.ID,
                node.id,
                self.ID,
                nlist
            ))

    def creatRoutingTableProcessMsg(self) -> bool:
        sendMsg = False
        if self.messageBuffer > 0:
            if len(self.nlists) == self.numberOfAgents:
                while self.messageBuffer:
                    msg:RoutingMessage = self.messageBuffer.pop()
                    if msg.nid not in self.nlists:
                        self.nlists[msg.nid] = msg.nlist
                        [node.handle.appendMsg.remote(RoutingMessage(self.ID, node.id, msg.nid, msg.nlist)) for node in self.neighbors.values() if node.id != msg.src]
                        sendMsg = True
        return sendMsg

    def creatRoutingTableCompute(self):
        for key in self.nlists.keys():
            self.routingTable[key] = set()
        for key in self.nlists.keys():
            visited = set()
            q = [key]
            while q:
                node = q.pop(0)
                visited.add(node)
                if node == self.ID:
                    for n in self.nlists[node]:
                        if n not in visited and n not in q:
                            self.routingTable[key].add(n)
                    break
                else:
                    for n in self.nlists[node]:
                        if n not in visited and n not in q:
                            q.append(n)
        
        self._computeCommunicateList()
    
    def internalLearning(self):
        for _ in range(self.learningInterval):
            self._internalLearning()

    def externalLearningSendMsg(self):
        positions = [[p.position[d] for d in range(self.dimension)] for p in self.swarmOrdered]
        [node.handle.appendMsg.remote(self, ExternalLearningMsg(self.ID, node.id, positions)) for node in self.neighbors.values()]

        for p in self.swarmOrdered:
            for d in range(self.dimension):
                p.externalVelocity[d] *= random.random() * self.psi  

    def externalLearningProcess(self):
        while self.messageBuffer:
            msg:ExternalLearningMsg = self.messageBuffer.pop()
            src, positions = msg.src, msg.positions
            for i in range(self.swarmSize):
                for d in range(self.dimension):
                    self.swarmOrdered[i].externalVelocity[d] += self.neighbors[src].weight * (positions[i][d] - self.swarmOrdered[i].position[d])
        
        for p in self.swarmOrdered:
            for d in range(self.dimension):
                p.internalVelocity[d] = p.externalVelocity[d]
                p.position[d] += p.externalVelocity[d]
                if p.position[d] > self.upperBound:
                    p.position[d] = self.upperBound
                if p.position[d] < self.lowerBound:
                    p.position[d] = self.lowerBound
            p.fitness = self.f(p.position)
        
        self.swarm.sort()
        self.curFitness = self.swarm[0].fitness

    def adaptiveIntervalSendMsg(self):
        meanFit = 0.0
        for p in self.swarm:
            meanFit += p.fitness
        meanFit /= self.swarmSize
        self.fitnessList[self.ID] = meanFit

        for dst, items in self.communicateList.items():
            flist:list[tuple[int, float|None]] = []
            for n in items:
                flist.append((n, self.fitnessList[n]))
            self.neighbors[dst].handle.appendMsg.remote(AdaptiveMessage(self.ID, dst, flist))

    def adaptiveIntervalProcess(self):
        while self.messageBuffer:
            msg:AdaptiveMessage = self.messageBuffer.pop()
            src, flist = msg.src, msg.flist
            for u, f in flist:
                self.fitnessList[u] = f
        
        tmp = [f for f in self.fitnessList if f != None]
        totalMeanFit = sum(tmp) / len(tmp)
        if totalMeanFit < self.bestFitness:
            self.bestFitness = totalMeanFit
            self.c = 0
        else:
            self.c += 1
            if self.c > self.threshold and self.learningInterval > self.minInverval:
                self.learningInterval -= 1
                self.c = 0

    def getRecord(self):
        meanPosition = [sum([p.position[d] for p in self.swarm]) / self.swarmSize for d in range(self.dimension)]
        return self.f.evaluateCounter, self.swarm[0].position, meanPosition

    def _internalLearning(self):
        self.curFitness = self.swarm[0].fitness
        # self.levelIndex = self._selectLevel()
        self.levelIndex = self._selectLevel()
        self.NL = self.levelPool[self.levelIndex]
        self.LS = self.swarmSize // self.NL
        # (0, 1, 2, ..., NL - 1)
        for curLevel in range(self.NL - 1, 0, -1):
            numberOfParticle = self.LS
            if curLevel == self.NL - 1:
                numberOfParticle += self.swarmSize % self.NL
            for i in range(numberOfParticle):
                curIndex = curLevel * self.LS + i
                p1Index, p2Index = 0, 0
                if curLevel >= 2:
                    rl1 = random.randrange(0, curLevel)
                    rl2 = random.randrange(0, curLevel)
                    while rl1 == rl2:
                        rl2 = random.randrange(0, curLevel)
                    if rl1 > rl2:
                        rl1, rl2 = rl2, rl1
                    p1Index = random.randrange(0, self.LS) + self.LS * rl1
                    p2Index = random.randrange(0, self.LS) + self.LS * rl2
                elif curLevel == 1:
                    p1Index = random.randrange(0, self.LS)
                    p2Index = random.randrange(0, self.LS)
                    while p1Index == p2Index:
                        p2Index = random.randrange(0, self.LS)
                    if self.swarm[p2Index].fitness < self.swarm[p1Index].fitness:
                        p1Index, p2Index = p2Index, p1Index
                for d in range(self.dimention):
                    r1 = random.random()
                    r2 = random.random()
                    r3 = random.random()
                    vertical = r1 * self.swarm[curIndex].internalVelocity[d] \
                        + r2 * (self.swarm[p1Index].position[d] - self.swarm[curIndex].position[d]) \
                            + r3 * self.phi * (self.swarm[p2Index].position[d] - self.swarm[curIndex].position[d])
                    self.swarm[curIndex].position[d] = self.swarm[curIndex].position[d] + vertical
                    self.swarm[curIndex].internalVelocity[d] = vertical
                    if self.swarm[curIndex].position[d] > self.upperBound:
                        self.swarm[curIndex].position[d] = self.upperBound
                    if self.swarm[curIndex].position[d] < self.lowerBound:
                        self.swarm[curIndex].position[d] = self.lowerBound
        for p in self.swarm:
            p.fitness = self.f(p.position)
        self.swarm.sort()
        if self.curFitness > self.swarm[0].fitness:
            self.levelPerformance[self.levelIndex] = (self.curFitness - self.swarm[0].fitness) / (self.curFitness)
        else:
            self.levelPerformance[self.levelIndex] = 0
        self.curFitness = self.swarm[0].fitness

    def _selectLevel(self) -> int:
        if self.fixed:
            fixedLevelIndex = 0
            return fixedLevelIndex
        else:
            total = sum([exp(7 * p) for p in self.levelPerformance])
            cumulativePro:list[float] = [0.0 for _ in range(self.levelPoolSize + 1)]
            for i in range(self.levelPoolSize):
                cumulativePro[i + 1] = cumulativePro[i] + exp(7 * self.levelPerformance[i]) / total
            tmp = random.random()
            selected = -1
            for i in range(self.levelPoolSize):
                if tmp <= cumulativePro[i + 1]:
                    selected = i
                    break
            assert selected != -1
            return selected

    def _computeCommunicateList(self):
        for key, items in self.routingTable.items():
            for n in self.neighbors:
                if n in items:
                    self.communicateList[n].append(key)