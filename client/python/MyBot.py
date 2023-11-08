import random
from base import *
from req import *
from resp import *
from client import *
from ui import UI
from config import config
from time import sleep
import numpy as np
from collections import deque


class MapBlock:
    Null = 0
    Enemy = 1
    PlayerSelf = 2
    Wall = 3
    Barrier = 4
    BombCover = 5
    HpItem = 6
    InvincibleItem = 7
    ShieldItem = 8


class GameStatus:
    Waiting = 0
    Starting = 1
    End = 2
    Win = 3
    Fail = 4


class Bot:
    """My silly bot"""
    def __init__(self, client:Client=None, map_size:int=15, block_num:int=9, action_num:int=5, bomb_base_time:int=5, show_ui:bool=False) -> None:
        """initialize game data"""

        # Game Infrastructure
        self.client = client
        self.resp:ActionResp
        self.ui = None
        if show_ui:
            self.ui = UI()

        # Game Configuration
        self.gameStatus = GameStatus.Waiting
        self.block_num = block_num
        self.action_num = action_num
        self.bomb_base_time = bomb_base_time
        self.Num2ActionReq = [
            ActionType.SILENT,
            ActionType.MOVE_LEFT,
            ActionType.MOVE_RIGHT,
            ActionType.MOVE_UP,
            ActionType.MOVE_DOWN,
            ActionType.PLACED,
        ]
        self.DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)] # search directions 

        # Game Data
        self.round = 0
        self.bomb_timing = np.full((map_size, map_size), np.inf)
        self.map = np.zeros((map_size, map_size, block_num))
        self.map_size = map_size
        self.player_info:dict
        self.previous_score:int = 0
        self.player_id = -1
        self.enemy_info:dict
        self.enemy_id = -1


    def receive(self) -> None:
        """receive data from server"""
        self.resp = self.client.recv()

        if self.gameStatus == GameStatus.Waiting and self.resp.type == PacketType.ActionResp:
            self.gameStatus = GameStatus.Starting
            self.player_id = self.resp.data.player_id
        elif self.resp.type == PacketType.GameOver:
            self.gameStatus = GameStatus.End
            return None
        
        self.round = self.resp.data.round
        self.updateMap(self.resp.data.map)
    

    def setPlayerId(self, player_id:int=1, enemy_id:int=0) -> None:
        """for DEBUG"""
        self.player_id = player_id
        self.enemy_id = enemy_id


    def joinGame(self) -> None:
        """join game"""
        self.client.connect()
        
        initPacket = PacketReq(PacketType.InitReq, InitReq(config.get("player_name")))
        self.client.send(initPacket)


    def run(self, isTrain:bool=False, output:bool=False):
        assert self.client is not None, "Client does not exist!"

        # join the game and wait to start
        self.joinGame()
        self.receive()
        while self.gameStatus != GameStatus.Starting:
            print("Waiting to start...")
            self.receive()
            sleep(0.1)
        
        if self.ui:
            self.ui.player_id = self.player_id

        # game start
        while self.gameStatus == GameStatus.Starting:
            if self.ui:
                self.ui.refresh(self.resp.data)
                self.ui.display()

            if output:
                # os.system('clear')
                self.printMap()

            # state = self.getState()

            # first step
            action = self.choose_action()
            print(f"action:{action}")
            self.step(action)

            # seconde step
            action = self.choose_action()
            print(f"action:{action}")
            self.step(action)

            self.receive()
            self
            if isTrain:
                self.learn()

        if output:
            self.printMap()

        if self.player_id in self.resp.data.winner_ids:
            self.gameStatus = GameStatus.Win
            print("Win!")
        else:
            self.gameStatus = GameStatus.Fail
            print("Fail!")

        if isTrain:
            self.learn()

        print(f"Game over! Final scores: {self.resp.data.scores}")


    def getState(self) -> None:
        """Organize environmental data"""
        # TODO: simplified state parameters
        pass


    def choose_action(self) -> ActionType:
        """Determine the behavior based on the current state"""
        # return self.Num2ActionReq[0]
        # return self.Num2ActionReq[random.randint(0, self.actionNum)]

        player_pos = self.player_info["pos"]
        if player_pos is None:
            assert False, 'Player Id error!'
        
        # search item
        path_to_item = self.bfs_with_two_steps(player_pos, self.is_item, self.bomb_timing, max_depth=None)
        if path_to_item:
            next_step = path_to_item[1]
            return self.get_move_from_path(player_pos, next_step)

        # search enemy
        path_to_enemy = self.bfs_with_two_steps(player_pos, self.is_enemy, self.bomb_timing, max_depth=None)
        if path_to_enemy:
            next_step = path_to_enemy[1]
            return self.get_move_from_path(player_pos, next_step)

        # avoid bomb
        safe_path = self.bfs_with_two_steps(player_pos, self.is_safe_to_move, self.bomb_timing, max_depth=None)
        if safe_path:
            next_step = safe_path[0]
            return self.get_move_from_path(player_pos, next_step)

        # place bomb or keep silent
        return random.choice([ActionType.PLACED, ActionType.SILENT])


    def get_move_from_path(self, start, end):
        """decide direction"""
        x0, y0 = start
        x1, y1 = end
        if x1 < x0:
            return ActionType.MOVE_UP
        elif x1 > x0:
            return ActionType.MOVE_DOWN
        elif y1 < y0:
            return ActionType.MOVE_LEFT
        elif y1 > y0:
            return ActionType.MOVE_RIGHT
        return None


    def step(self, action:ActionType) -> None:
        """Sending data to the server, making an action"""
        action = ActionReq(self.player_id, action)
        actionPacket = PacketReq(PacketType.ActionReq, action)
        self.client.send(actionPacket)


    def learn(self) -> None:
        """Judge the value and learn the action result"""
        reward = self.player_info.scores - self.previous_score
        self.previous_score = self.player_info.scores
        pass


    def updatePlayer(self, player:dict, pos:list) -> MapBlock:
        """Parse the player data and return the current player"""
        player_data = {
            "alive": player.alive,
            "bomb_max_num": player.bomb_max_num,
            "bomb_now_num": player.bomb_now_num,
            "bomb_range": player.bomb_range,
            "hp": player.hp,
            "invincible_time": player.invincible_time,
            "score": player.score,
            "shield_time": player.shield_time,
            "speed": player.speed,
            "pos": pos
        }

        if player.player_id == self.player_id:
            self.player_info = player_data
            return MapBlock.PlayerSelf
        else:
            self.enemy_info = player_data
            return MapBlock.Enemy
        

    def getPLayersDistance(self):
        """calculates the distance between players and returns two values: 
            one is the distance including destructible blocks, 
            and the other is the distance considering all as walls."""
        pass


    def bfs(self, start, target_check, max_depth=None):
        """
        BFS is used for path planning.
        start: The starting point (x, y)
        target_check: A function that checks whether the current position is a target or not
        max_depth: The maximum depth of the search
        """
        visited = set()  # Visited collections
        queue = deque([(start, [], 0)])  # Store tuples in queue: (current position, path, current depth)

        while queue:
            (x, y), path, depth = queue.popleft()
            if (x, y) in visited:
                continue

            visited.add((x, y))

            if target_check(self.game_map[x, y, :]):  # Checks if the current location is the target
                return path + [(x, y)]

            if max_depth is not None and depth >= max_depth:  # Check the depth limit
                continue

            # Add adjacent walkable nodes to the queue
            for dx, dy in self.DIRECTIONS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.map_size and 0 <= ny < self.map_size and (nx, ny) not in visited:
                    # Make sure the next position can be moved
                    if self.game_map[nx, ny, MapBlock.Wall] == 0 and self.game_map[nx, ny, MapBlock.Barrier] == 0:
                        queue.append(((nx, ny), path + [(x, y)], depth + 1))

        return None  # can't find target
    

    def bfs_with_two_steps(self, start, target_check, bomb_timing, max_depth=None) -> list | None:
        """Consider that each turn can move two steps"""
        visited = set()
        queue = deque([(start, [], 0, 0, 0)])  # Store tuples in queue: (current position, path, current depth, time, step)

        while queue:
            print(len(queue), len(visited))
            (x, y), path, depth, time, steps = queue.popleft()
            state = (x, y)
            if state in visited:
                continue

            visited.add(state)

            if target_check(self.map[x, y, :], time):
                return path + [(x, y)]

            if max_depth is not None and depth >= max_depth:
                continue

            for dx, dy in self.DIRECTIONS:
                nx, ny = x + dx, y + dy
                next_time = time + (steps // 2)
                next_steps = (steps + 1) % 2

                # reset step and add time
                if next_steps == 0:
                    next_time += 1

                if 0 <= nx < self.map_size and 0 <= ny < self.map_size and (nx, ny, next_time, next_steps) not in visited:
                    if self.is_safe_to_move(self.map[nx, ny], next_time):
                        queue.append(((nx, ny), path + [(x, y)], depth + 1, next_time, next_steps))

        return None
    
    # Define different target checking functions
    def is_enemy(self, block:np.array, time:int) -> bool:
        return block[MapBlock.Enemy] == 1

    def is_item(self, block:np.array, time:int) -> bool:
        return np.any(block[MapBlock.HpItem : MapBlock.ShieldItem + 1])

    def is_safe_to_move(self, block:np.array, time:int) -> bool:
        """Checks if the player can safely move to a block at a given time."""
        is_bomb_active = block[MapBlock.BombCover] and time < self.bomb_timing[block[MapBlock.BombCover]]
        can_move = block[MapBlock.Wall] == 0 and block[MapBlock.Barrier] == 0
        return can_move and not is_bomb_active


    def updateMap(self, map_data:dict) -> dict:
        """Update map and player positions"""
        self.map = np.zeros((self.map_size, self.map_size, self.block_num))
        self.update_bomb_timers()
        for cell in map_data:
            x = cell.x
            y = cell.y
            
            if not cell.objs:
                self.map[x, y, MapBlock.Null] = 1
            else:
                for obj in cell.objs:
                    if obj.type == ObjType.Player:
                        self.map[x, y, self.updatePlayer(obj.property, [y, x])]

                    elif obj.type == ObjType.Bomb:
                        # The impact range of the bomb explosion
                        bomb_range = obj.property.bomb_range
                        print(obj.property)

                        for _x in range(x - bomb_range, x + bomb_range + 1):
                            if 0 <= _x < self.map_size and self.map[_x, y, MapBlock.Wall] != 1 and self.map[_x, y, MapBlock.BombCover] != 1:
                                self.map[_x, y, MapBlock.BombCover] = 1
                                self.bomb_timing[_x, y] = min(self.bomb_timing[_x, y], self.bomb_base_time)
                                if self.bomb_timing[_x, y] == 0:
                                    self.bomb_timing[_x, y] = 1

                        for _y in range(y - bomb_range, y + bomb_range + 1):
                            if 0 <= _y < self.map_size and self.map[x, _y, MapBlock.Wall] != 1 and self.map[x, _y, MapBlock.BombCover] != 1:
                                self.map[x, _y, MapBlock.BombCover] = 1
                                self.bomb_timing[x, _y] = min(self.bomb_timing[x, _y], self.bomb_base_time)
                                if self.bomb_timing[x, _y] == 0:
                                    self.bomb_timing[x, _y] = 1

                    elif obj.type == ObjType.Item:
                        # TODO: process Item
                        pass

                    elif obj.type == ObjType.Block:
                        if obj.property.removable:
                            self.map[x, y, MapBlock.Barrier] = 1
                        else:
                            self.map[x, y, MapBlock.Wall] = 1

                    else:
                        assert False, "Unknown Block Object!"

        return self.map


    def update_bomb_timers(self):
        """update bomb timers"""
        ### 注意：当两个炸弹覆盖同一块区域，计算炸弹剩余时间的代码可能会出问题
        self.bomb_timing[self.bomb_timing < np.inf] -= 1
        self.bomb_timing[self.bomb_timing < 0] = np.inf

    
    def printMap(self) -> None:
        for _ in self.map:
            for __ in _:
                print(__)
            print('')


def test():
    bot = Bot()
    with open("resp.json", 'r') as f_obj:
        resp = PacketResp().from_json(f_obj.read())

    bot.setPlayerId()
    result = bot.updateMap(resp.data.map)

    for _ in result:
        print(_)


def main():
    with Client() as client:
        bot = Bot(client=client, show_ui=False)
        bot.run(output=False)


if __name__ == "__main__":
    # test()
    main()