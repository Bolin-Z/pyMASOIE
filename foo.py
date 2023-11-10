import random

class p:
    def __init__(self, id) -> None:
        self.id = id
        self.fitness = random.randrange(0, 100)
    
    def __lt__(self, __o: "p") -> bool:
        return self.fitness < __o.fitness
    
    def __eq__(self, __o: "p") -> bool:
        return self.fitness == __o.fitness

    def __str__(self) -> str:
        return f"{self.id}({self.fitness})"
    
    def __repr__(self) -> str:
        return f"{self.id}({self.fitness})"

TOTAL = 10
origin = [p(_) for _ in range(TOTAL)]
toBesort = []
for i in range(TOTAL):
    toBesort.append(origin[i])

# print(origin)
# print(toBesort)
# print("sort")
# toBesort.sort()
# print(origin)
# print(toBesort)
# print("touch origin")
# origin[0].fitness = random.randrange(0, 100)
# print(origin)
# print(toBesort)
# print("touch toBesort")
# toBesort[0].fitness = random.randrange(0, 100)
# print(origin)
# print(toBesort)

print([_ for _ in range(10,0,-1)])
