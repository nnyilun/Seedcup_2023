# 2023种子杯初赛报告

队伍名：我要进种子班队

## 主要思路

* 编写Bot类，储存了基本的游戏框架，负责加入游戏、运行操作等

```python
class Bot:
    """My silly bot"""
    def __init__(
            self, ...) -> None:
        """initialize game data"""

        # Game Infrastructure
        ...

        # Game Configuration
        ...


    def receive(self) -> None:
        """receive data from server"""
        ...


    def joinGame(self) -> None:
        """join game"""
        ...


    def waitToStart(self) -> None:
        """wait to start"""
        ...


    def gameOver(self) -> None:
        """print game over info"""
        ...

    
    def output(self, out:bool) -> None:
        ...

    def run(self, out:bool=False):
        ...


    def isDone(self) -> bool:
        return not self.gameStatus == GameStatus.Starting


    def step(self, action1:ActionType, action2:ActionType=ActionType.SILENT) -> None:
        """Sending data to the server, making an action"""
        ...

    def updateMap(self, map_data:dict) -> dict:
        """Update map and player positions"""
        ...

    def printMap(self) -> None:
        ...
```

* 服务器接收的数据为json格式，为了便于后续处理，将其重新格式化为二维数组。我们使用$15\times15\times7$的三维张量储存所有和位置相关的信息，将地图上每个位置的不同方块（如道具、可破坏方块等）使用独热编码的方式储存，将和位置无关的玩家信息储存到字典。

```python
def updatePlayer(self, player:dict, pos:list) -> MapBlock:
    """Parse the player data and return the current player"""
    player_data = {
        "id": player.player_id,
        ...
    }
    ...


def updateMap(self, map_data:dict) -> dict:
    """Update map and player positions"""
    self.map = np.zeros((self.map_size, self.map_size, self.block_num))
    for cell in map_data:
        x = cell.x
        y = cell.y
        
        if not cell.objs:
            ...
        else:
            for obj in cell.objs:
                if obj.type == ObjType.Player:
                    ...
                elif obj.type == ObjType.Bomb:
                    ...

                elif obj.type == ObjType.Item:
                    ...

                elif obj.type == ObjType.Block:
                    ...

                else:
                    assert False, "Unknown Block Object!"

    return self.map
```

* 使用BFS探索和规划路径。探索可破坏方块、敌方玩家位置、安全区等，探索的过程中将需要到达的地点及通往改地点的路径储存。

```python
def markSafe(self, step:int=5) -> set:
    start = self.player_info['pos']
    que = deque([(start, [])])
    ret = set()
    visited = set()

    while len(que):
        ...

    return ret

def goToTarget(self, target:tuple=None) -> list:
    """calculate the path to the target and return the path"""
    visited = set()
    queue = deque([(self.player_info['pos'], [])])  
    
    while len(queue):
        ...
    return []
```


* 因为每个回合玩家可以行动两次，所以我们在行动一次后，通过模拟函数获得这次行动后环境的变化，然后再生成第二个动作。

```python
def simulate(self, SafeZone:set) -> list:
        if self.player_info["bomb_max_num"] - self.player_info["bomb_now_num"] <= 0:
            return

        map = np.zeros((self.map_size, self.map_size))
        for (x, y) in SafeZone:
            map[x, y] = 1

        result = []
        ...
        return result
```

* 评估每条路径的价值，选择价值最高的路径进行移动。