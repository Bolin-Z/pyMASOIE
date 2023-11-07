import ray
import copy
from enum import IntEnum

class MessageType(IntEnum):
    P2P_MESSAGE = 0
    BROADCAST_MESSAGE = 1

class Message:
    def __init__(self, source:int, destination:int, tag:int, messageType, payload) -> None:
        self.src = source
        self.dst = destination
        self.tag = tag
        self.msgType = messageType
        self.payload = copy.deepcopy(payload)

@ray.remote
class MessageBuffer:
    def __init__(self, id:int) -> None:
        self.id = id
        self.buffer:dict[int, list[Message]] = dict()

    def appendMessage(self, id:int, msg:Message) -> None:
        if id not in self.buffer:
            self.buffer[id] = []
        self.buffer[id].append(msg)

    def retrieveMessage(self, id:int, tag:int) -> None | Message:
        if id in self.buffer:
            target = None
            for index, message in enumerate(self.buffer[id]):
                if message.tag == tag:
                    target = index
                    break
            if target:
                return self.buffer[id].pop(target)
        return None

@ray.remote
def send():
    """p2p block send
    """
    pass

@ray.remote
def recv():
    """p2p block receive
    """
    pass

@ray.remote
def broadcast():
    """broadcast
    """
    pass