from base import *
from req import *
from resp import *
from client import *
from ui import UI
import random
from time import sleep
import numpy as np
from collections import deque


TEAM_ID = '1'


class MapBlock:
    Null = 0
    Enemy = 1
    PlayerSelf = 2
    Wall = 3
    Barrier = 4
    BombCover = 5
    Item = 6
    # HpItem = 6
    # InvincibleItem = 7
    # ShieldItem = 8


class GameStatus:
    Waiting = 0
    Starting = 1
    End = 2
    Win = 3
    Fail = 4


class Bot:
    """My silly bot"""
    def __init__(
            self, client:Client=None, 
            map_size:int=15, block_num:int=7, action_num:int=6, 
            player_feature_num:int=11, bomb_base_time:int=5, bomb_base_num:int=2,
            show_ui:bool=False) -> None:
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
        self.player_feature_num = player_feature_num
        self.map_size = map_size
        self.bomb_base_time = bomb_base_time
        self.bomb_base_num = bomb_base_num
        self.Num2ActionReq = [
            ActionType.SILENT,
            ActionType.MOVE_LEFT,
            ActionType.MOVE_RIGHT,
            ActionType.MOVE_UP,
            ActionType.MOVE_DOWN,
            ActionType.PLACED,
        ]
        self.ActionReq2Num = {
            ActionType.SILENT : "SILENT",
            ActionType.MOVE_LEFT : "MOVE_LEFT",
            ActionType.MOVE_RIGHT : "MOVE_RIGHT",
            ActionType.MOVE_UP : "MOVE_UP",
            ActionType.MOVE_DOWN : "MOVE_DOWN",
            ActionType.PLACED : "PLACED",
        }
        self.DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)] # search directions 

        # Game Data
        self.round = 0
        self.bomb_timing = np.full((map_size, map_size), np.inf)
        self.map = np.zeros((map_size, map_size, block_num))

        self.previous_score:int = 0
        self.bomb_now_num:int = 0
        self.player_info:dict
        self.player_info_array:np.array
        self.player_id = -1

        self.enemy_info:dict
        self.enemy_info_array:np.array
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

    
    def setCLient(self, client:Client=None) -> None:
        self.client = client


    def joinGame(self) -> None:
        """join game"""
        self.client.connect()
        
        initPacket = PacketReq(PacketType.InitReq, InitReq(TEAM_ID))
        self.client.send(initPacket)


    def waitToStart(self) -> None:
        """wait to start"""
        self.receive()
        while self.gameStatus != GameStatus.Starting:
            print("Waiting to start...")
            self.receive()
            sleep(0.1)
        
        print("Game start!")
        if self.ui:
            self.ui.player_id = self.player_id


    def gameOver(self) -> None:
        """print game over info"""
        if self.player_id in self.resp.data.winner_ids:
            self.gameStatus = GameStatus.Win
            print("Win!")
        else:
            self.gameStatus = GameStatus.Fail
            print("Fail!")
            
        print(f"Game over! Final scores: {self.resp.data.scores}")

    
    def output(self, out:bool) -> None:
        if self.ui:
            self.ui.refresh(self.resp.data)
            self.ui.display()

        if out:
            self.printMap()

    def run(self, out:bool=False):
        assert self.client is not None, "Client does not exist!"

        # join the game and wait to start
        self.joinGame()
        self.waitToStart()

        # game start
        while not self.isDone():
            self.output(out)

            action1, action2 = self.choose_action()
            print(f"action:{action1}, {action2 if action2 != None else None}")
            self.step(action1, action2)

            self.receive()

        self.output(out)

        self.gameOver()


    def isDone(self) -> bool:
        return not self.gameStatus == GameStatus.Starting


    def choose_action(self) -> (ActionType, ActionType | None):
        """Determine the behavior based on the current state"""
        safe_zone = self.markSafe()
        can_place_bomb = self.simulate(safe_zone)
        print(f'can_place_bomb:{can_place_bomb}')
        
        if len(can_place_bomb) == 0:
            return self.Num2ActionReq[0], self.Num2ActionReq[0]

        max_value_pos = max(can_place_bomb, key=lambda x:x[1])
        path = self.goToTarget(max_value_pos[0])
        print(f'path:{path}')
        return self.get_move_from_path(path[0], path[1]), ActionType.PLACED
        return self.Num2ActionReq[0], self.Num2ActionReq[0]


    def get_move_from_path(self, start:tuple, end:tuple) -> ActionType:
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
        return ActionType.SILENT


    def step(self, action1:ActionType, action2:ActionType=ActionType.SILENT) -> None:
        """Sending data to the server, making an action"""
        actionPacket = PacketReq(PacketType.ActionReq, [ActionReq(self.player_id, action1), ActionReq(self.player_id, action2)])
        self.client.send(actionPacket)


    def getReward(self, action1:ActionType, action2:ActionType, isDone:bool=False) -> int:
        if isDone:
            scores = self.resp.data.scores
            for _ in scores:
                if _['player_id'] == self.player_id:
                    reward = _['score'] - self.previous_score
        else:
            reward = self.player_info['score'] - self.previous_score
            self.previous_score = self.player_info['score'] 
        return reward


    def updatePlayer(self, player:dict, pos:list) -> MapBlock:
        """Parse the player data and return the current player"""
        player_data = {
            "id": player.player_id,
            "pos": pos,
            "alive": player.alive,
            "bomb_max_num": player.bomb_max_num,
            "bomb_now_num": player.bomb_now_num,
            "bomb_range": player.bomb_range,
            "hp": player.hp,
            "invincible_time": player.invincible_time,
            "score": player.score,
            "shield_time": player.shield_time,
            "speed": player.speed,
        }

        if player.player_id == self.player_id:
            self.player_info = player_data
            return MapBlock.PlayerSelf
        else:
            self.enemy_id = player.player_id
            self.enemy_info = player_data
            return MapBlock.Enemy
        

    def getPLayersDistance(self):
        """calculates the distance between players and returns two values: 
            one is the distance including destructible blocks, 
            and the other is the distance considering all as walls."""
        pass


    def markSafe(self, step:int=5) -> set:
        start = self.player_info['pos']
        que = deque([(start, [])])
        ret = set()
        visited = set()

        while len(que):
            (x, y), path = que.popleft()
            for dx, dy in self.DIRECTIONS:
                _x, _y = x + dx, y + dy

                if 0 <= _x < self.map_size and 0 <= _y < self.map_size and self.map[_x, _y, MapBlock.BombCover] + self.map[_x, _y, MapBlock.Wall] + self.map[_x, _y, MapBlock.Barrier] == 0:
                    if step - len(path) <= 1:
                        continue
                    que.append(((_x, _y), path + [(x, y)]))
                    ret.add((_x, _y))
                    visited.add((_x, _y))

        return ret
    

    def simulate(self, SafeZone:set) -> list:
        if self.player_info["bomb_max_num"] - self.player_info["bomb_now_num"] <= 0:
            return

        map = np.zeros((self.map_size, self.map_size))
        for (x, y) in SafeZone:
            map[x, y] = 1

        result = []
        temp_map = map.copy()
        start = self.player_info["pos"]
        bomb_range = self.player_info["bomb_range"]
        cnt = 0
        # place here
        for _x in range(start[0] - bomb_range, start[0] + bomb_range + 1):
            if 0 <= _x < self.map_size:
                temp_map[_x, start[1]] = 0
                if self.map[_x, start[1], MapBlock.Barrier] == 1:
                    cnt += 1

        for _y in range(start[1] - bomb_range, start[1] + bomb_range + 1):
            if 0 <= _y < self.map_size:
                temp_map[start[0], _y] = 0
                if self.map[start[0], _y, MapBlock.Barrier] == 1:
                    cnt += 1
        
        if np.sum(temp_map) > 0 and cnt != 0:
            coordinates = np.where(temp_map == 1)
            print(f'coor:{coordinates}, list:{list(zip(coordinates[0], coordinates[1]))}')
            result.append((start, cnt, list(zip(coordinates[0], coordinates[1]))))

        # place one step range
        for (dx, dy) in self.DIRECTIONS:
            temp_map = map.copy()
            cnt = 0
            _x, _y = start[0] + dx, start[1] + dy
            print(f'_x:{_x}, _y:{_y}')
            if 0 <= _x < self.map_size and 0 <= _y < self.map_size:
                for __x in range(_x - bomb_range, _x + bomb_range + 1):
                    if 0 <= __x < self.map_size:
                        temp_map[__x, _y] = 0
                        if self.map[__x, _y, MapBlock.Barrier] == 1:
                            cnt += 1

                for __y in range(_y - bomb_range, _y + bomb_range + 1):
                    if 0 <= __y < self.map_size:
                        temp_map[_x, __y] = 0
                        if self.map[_x, __y, MapBlock.Barrier] == 1:
                            cnt += 1
                
                if np.sum(temp_map) > 0 and cnt != 0:
                    coordinates = np.where(temp_map == 1)
                    result.append(((_x, _y), cnt, list(zip(coordinates[0], coordinates[1]))))

        return result

    
    def goToTarget(self, target:tuple=None) -> list:
        """calculate the path to the target and return the path"""
        visited = set()
        queue = deque([(self.player_info['pos'], [])])  
        
        while len(queue):
            (x, y), path = queue.popleft()
            if not target and self.map[x, y, MapBlock.BombCover] == 0:
                path.append((x, y))
                return path
            if target and (x, y) == target :
                path.append((x, y))
                return path
            
            for dx, dy in self.DIRECTIONS:
                _x = x + dx
                _y = y + dy
                if 0 <= _x < self.map_size and 0 <= _y < self.map_size and (_x, _y) not in visited \
                        and self.map[_x, _y, MapBlock.Wall] == 0 and self.map[_x, _y, MapBlock.Barrier] == 0:
                    visited.add((_x, _y))
                    queue.append(((_x, _y), path + [(x, y)]))
        return []

    def updateMap(self, map_data:dict) -> dict:
        """Update map and player positions"""
        self.map = np.zeros((self.map_size, self.map_size, self.block_num))
        for cell in map_data:
            x = cell.x
            y = cell.y
            
            if not cell.objs:
                self.map[x, y, MapBlock.Null] = 1
            else:
                for obj in cell.objs:
                    if obj.type == ObjType.Player:
                        self.map[x, y, self.updatePlayer(obj.property, (x, y))] = 1

                    elif obj.type == ObjType.Bomb:
                        # The impact range of the bomb explosion
                        bomb_range = obj.property.bomb_range

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
                        self.map[x, y, MapBlock.Item] = 1

                    elif obj.type == ObjType.Block:
                        if obj.property.removable:
                            self.map[x, y, MapBlock.Barrier] = 1
                        else:
                            self.map[x, y, MapBlock.Wall] = 1

                    else:
                        assert False, "Unknown Block Object!"

        return self.map

    
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
        bot = Bot(show_ui=True)
        bot.setCLient(client)
        bot.run(out=False)


if __name__ == "__main__":
    # test()
    main()