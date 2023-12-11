import ray, copy

ANY_TAG = -1
ANY_SRC = -1
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

    def retrieveMessage(self, id:int, tag:int) -> None | Message:
        if id == ANY_SRC:
            return self._retrieveAnySourceMessage(tag)
        else:
            return self._retrieveSpecificSourceMessage(id, tag)
    
    def _retrieveSpecificSourceMessage(self, id:int, tag:int) -> None | Message:
        if id in self.buffer and len(self.buffer[id]) > 0:
            for index, message in enumerate(self.buffer[id]):
                if tag == message.tag or tag == ANY_TAG:
                    self.messageCount -= 1
                    return self.buffer[id].pop(index)
        return None
    
    def _retrieveAnySourceMessage(self, tag:int) -> None | Message:
        if self.messageCount > 0:
            for id, messages in self.buffer.items():
                for index, message in enumerate(messages):
                    if tag == message.tag or tag == ANY_TAG:
                        self.messageCount -= 1
                        return self.buffer[id].pop(index)
        return None

class Sender:
    def __init__(self) -> None:
        self.handlers = dict()

    def __call__(self, src:int, dst:int|list[int], payload, tag:int=ANY_TAG) -> None:
        if type(dst) == int:
            self._send(src, dst, payload, tag)
        else:
            [self._send(src, d, payload, tag) for d in dst]
    
    def _send(self, src:int, dst:int, payload, tag:int=ANY_TAG) -> None:
        if dst not in self.handlers:
            self.handlers[dst] = ray.get_actor("MessageBuffer" + str(dst))
        handler = self.handlers[dst]
        msg = Message(src, dst, payload, tag)
        handler.appendMessage.remote(src, msg)

class Receiver:
    def __init__(self, id) -> None:
        self.handler = ray.get_actor("MessageBuffer" + str(id))
    
    def __call__(self, src:int, dst:int, tag:int=ANY_TAG):
        return self.handler.retrieveMessage.remote(src, tag)
