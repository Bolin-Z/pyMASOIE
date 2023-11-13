import ray, time, random
from .communicate import send, recv
from .communicate import ANY_SRC, ANY_TAG, DEFAULT_TIMEOUT
from functools import total_ordering
from typing import Callable
from math import exp
from CEC2013 import CEC2013
from copy import deepcopy

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

class LocalEvaluator:
    def __init__(self, id:int, localEvaluate:Callable[[list[float], int], float]) -> None:
        self.id = id
        self.f = localEvaluate
        self.evaluateCounter = 0

    def __call__(self, x:list[float]) -> float:
        self.evaluateCounter += 1
        return self.f(x, self.id)
@ray.remote
class Agent:
    def __init__(
        self,
        # problem
        dimension:int,
        lowerBound:float,
        upperBound:float,
        # evaluate:Callable[[list[float]], float],
        funcID:str,
        # structure
        swarmSize:int,
        # internal learning
        learningInterval:int,
        psi:float,
        # dllso
        phi:float,
        # network
        ID:int,
        neighborsID:list[int],
        neighborsWeight:list[float]
        ) -> None:

        # problem
        problem = CEC2013(funcID)
        self.dimention = dimension
        self.upperBound = upperBound
        self.lowerBound = lowerBound
        self.f = LocalEvaluator(ID, problem.local_eva)
        # structure
        self.swarmSize = swarmSize
        # external learning
        self.psi = psi
        # internal learning dllso
        self.learningInterval = learningInterval
        self.phi = phi
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

        # network
        self.ID = ID
        self.neighborsID = neighborsID
        self.neighborsWeight:dict[int, float] = dict()
        for i in range(len(self.neighborsID)):
            self.neighborsWeight[self.neighborsID[i]] = neighborsWeight[i]
        self.routingTable:dict[int, set[int]] = dict()

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
        self.bestFitness = self.swarm[0].fitness

    def run(self) -> float:
        self._creatRoutingTable()
        print(f"[{self.ID}]: {self.neighborsID}")
        print(f"[{self.ID}] {self.routingTable}")
        # gen = 100000
        # for _ in range(gen):
        #     for _ in range(self.learningInterval):
        #         self._internalLearning()
        #     self._externalLearning()
        #     if self.ID == 0:
        #         print(f"{self.f.evaluateCounter} {self.bestFitness}")
        return self.swarm[0].position

    def _creatRoutingTable(self):
        ROUTING = 5
        FLOODING = 1
        STOP = 2
        class Message:
            def __init__(self, id:int, age:int, source:int, tag:int) -> None:
                self.id = id
                self.age = age
                self.source = source
                self.tag = tag
        
        messageCache:dict[int, Message] = dict()
        msg = Message(self.ID, 0, self.ID, FLOODING)
        send.remote(self.ID, self.neighborsID, msg, ROUTING)
        self.routingTable[self.ID] = set()
        for n in self.neighborsID:
            self.routingTable[self.ID].add(n)

        TIMEOUT = 60
        while True:
            m:Message = ray.get(recv.remote(ANY_SRC, self.ID, ROUTING, TIMEOUT))
            if not m:
                break
            if m.tag == FLOODING:
                if m.id not in messageCache:
                    messageCache[m.id] = deepcopy(m)
                    nm = Message(m.id, m.age + 1, self.ID, FLOODING)
                    send.remote(self.ID, self.neighborsID, nm, ROUTING)
                    if m.id not in self.routingTable:
                        self.routingTable[m.id] = set()
                    for n in self.neighborsID:
                        self.routingTable[m.id].add(n)
                else:
                    om = messageCache.pop(m.id)
                    if om.age > m.age:
                        messageCache[m.id] = deepcopy(m)
                        nm = Message(m.id, m.age + 1, self.ID, FLOODING)
                        send.remote(self.ID, self.neighborsID, nm, ROUTING)
                        for n in self.neighborsID:
                            self.routingTable[m.id].add(n)
                        st = Message(m.id, 0, self.ID, STOP)
                        send.remote(self.ID, om.source, st, ROUTING)
                    else:
                        st = Message(m.id, 0, self.ID, STOP)
                        send.remote(self.ID, m.source, st, ROUTING)
            elif m.tag == STOP:
                self.routingTable[m.id].discard(m.source)
            else:
                raise ValueError("Incorrect routing message type.")


    def _internalLearning(self):
        self.bestFitness = self.swarm[0].fitness
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
        if self.bestFitness > self.swarm[0].fitness:
            self.levelPerformance[self.levelIndex] = (self.bestFitness - self.swarm[0].fitness) / (self.bestFitness)
        else:
            self.levelPerformance[self.levelIndex] = 0
        self.bestFitness = self.swarm[0].fitness
    
    def _externalLearning(self):
        POSITION = 2
        positions = [[p.position[d] for d in range(self.dimention)] for p in self.swarmOrdered]
        send.remote(self.ID, self.neighborsID, positions, POSITION)
        
        for p in self.swarmOrdered:
            for d in range(self.dimention):
                p.externalVelocity[d] *= random.random() * self.psi
        
        waitingTasks = [recv.remote(n, self.ID, POSITION) for n in self.neighborsID]
        while waitingTasks:
            readyTasks, waitingTasks = ray.wait(waitingTasks)
            for task in readyTasks:
                msg = ray.get(task)
                assert msg
                src, position = msg
                for i in range(self.swarmSize):
                    for d in range(self.dimention):
                        self.swarmOrdered[i].externalVelocity[d] += self.neighborsWeight[src] * (position[i][d] - self.swarmOrdered[i].position[d])

        
        for p in self.swarmOrdered:
            for d in range(self.dimention):
                p.internalVelocity[d] = p.externalVelocity[d]
                p.position[d] += p.externalVelocity[d]
                if p.position[d] > self.upperBound:
                    p.position[d] = self.upperBound
                if p.position[d] < self.lowerBound:
                    p.position[d] = self.lowerBound
            p.fitness = self.f(p.position)
        
        self.swarm.sort()
        self.bestFitness = self.swarm[0].fitness

    def _selectLevel(self) -> int:
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
