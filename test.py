import ray, random, time
from MASOIE import *

NUM_OF_AGENT = 20
MESSAGE_LENGTH = 300
@ray.remote
class Agent:
    def __init__(self, id, nlist) -> None:
        self.id = id
        self.nlist = nlist
        self.msg = [random.randrange(0, 100) for _ in range(MESSAGE_LENGTH)]

    def run(self):
        counter = 0
        while counter < 2:
            print(f"Agent{self.id} start round{counter}: {self.msg}")
            msgTag = 2
            send.remote(self.id, self.nlist, self.msg, msgTag)
            # [send.remote(self.id, n, self.msg, msgTag) for n in self.nlist]
            tasks = [recv.remote(n, self.id, msgTag) for n in self.nlist]
            winlist = []
            while len(tasks):
                readys, tasks = ray.wait(tasks)
                for ready in readys:
                    src, msg = ray.get(ready)
                    if self.win(msg):
                        winlist.append(src)
            print(f"Agent{self.id} complete round{counter}: {winlist}")
            self.msg = [random.randrange(0, 100) for _ in range(MESSAGE_LENGTH)]
            counter += 1

    def win(self, o):
        counter = 0
        for i in range(MESSAGE_LENGTH):
            if self.msg[i] > o[i]:
                counter += 1
        return counter > MESSAGE_LENGTH / 2

ray.init()
messageBuffers = [MessageBuffer.options(name="MessageBuffer" + str(i)).remote(i) for i in range(NUM_OF_AGENT)]
agents = [Agent.remote(i, [(i+k)%NUM_OF_AGENT for k in range(-1, 2) if k != 0]) for i in range(NUM_OF_AGENT)]
time.sleep(5)
ray.get([agent.run.remote() for agent in agents])