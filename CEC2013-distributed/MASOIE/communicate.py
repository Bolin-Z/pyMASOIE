import ray, copy, time

ANY_TAG = -1
ANY_SRC = -1
DEFAULT_TIMEOUT = 100
class Message:
    def __init__(self, source:int, destination:int, payload, tag:int) -> None:
        self.src = source
        self.dst = destination
        self.payload = copy.deepcopy(payload)
        self.tag = tag
@ray.remote
class MessageBuffer:
    def __init__(self, id:int) -> None:
        self.id = id
        self.messageCount = 0
        self.buffer:dict[int, list[Message]] = dict()

    def appendMessage(self, id:int, msg:Message) -> None:
        if id not in self.buffer:
            self.buffer[id] = []
        self.buffer[id].append(msg)
        self.messageCount += 1

    def retrieveMessage(self, id:int, tag:int) -> None | tuple[int, Message]:
        if id == ANY_SRC:
            return self._retrieveAnySourceMessage(tag)
        else:
            return self._retrieveSpecificSourceMessage(id, tag)
    
    def _retrieveSpecificSourceMessage(self, id:int, tag:int) -> None | tuple[int, Message]:
        if id in self.buffer and len(self.buffer[id]) > 0:
            for index, message in enumerate(self.buffer[id]):
                if tag == message.tag or tag == ANY_TAG:
                    self.messageCount -= 1
                    return id, self.buffer[id].pop(index)
        return None
    
    def _retrieveAnySourceMessage(self, tag:int) -> None | tuple[int, Message]:
        if self.messageCount > 0:
            for id, messages in self.buffer.items():
                for index, message in enumerate(messages):
                    if tag == message.tag or tag == ANY_TAG:
                        self.messageCount -= 1
                        return id, self.buffer[id].pop(index)
        return None

@ray.remote
def send(src:int, dst:int|list[int], payload, tag:int=ANY_TAG) -> None:
    if type(dst) == int:
        _send(src, dst, payload, tag)
    else:
        [_send(src, d, payload, tag) for d in dst]

def _send(src:int, dst:int, payload, tag:int=ANY_TAG) -> None:
    handler = ray.get_actor("MessageBuffer" + str(dst))
    msg = Message(src, dst, payload, tag)
    handler.appendMessage.remote(src, msg)

@ray.remote
def recv(src:int, dst:int, tag:int=ANY_TAG, timeout:float=DEFAULT_TIMEOUT):
    handler = ray.get_actor("MessageBuffer" + str(dst))
    start = time.time()
    while True:
        res:tuple[int, Message]|None = ray.get(handler.retrieveMessage.remote(src, tag))
        if res:
            return res[0], res[1].payload
        if time.time() - start > timeout:
            break
    return None