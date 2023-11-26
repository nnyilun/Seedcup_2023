from base import *
from req import *
from resp import *
from client import *
from ui import UI
from config import config
import random
from time import sleep
import numpy as np


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
    def __init__(self, map_size:int=15, block_num:int=9, action_num:int=6, player_feature_num:int=11, bomb_base_time:int=5, show_ui:bool=False) -> None:
        """initialize game data"""

        # Game Infrastructure
        self.client = None
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
        
        initPacket = PacketReq(PacketType.InitReq, InitReq(config.get("player_name")))
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

    def run(self, isTrain:bool=False, out:bool=False):
        assert self.client is not None, "Client does not exist!"

        # join the game and wait to start
        self.joinGame()
        self.waitToStart()

        # game start
        while not self.isDone():
            self.output(out)

            action1, action2 = self.choose_action()
            print(f"action:{action1}, {action2 if action2 else None}")
            self.step(action1, action2)

            self.receive()
            if isTrain:
                self.learn()

        self.output(out)

        if isTrain:
            self.learn()

        self.gameOver()


    def isDone(self) -> bool:
        return not self.gameStatus == GameStatus.Starting


    def choose_action(self) -> (ActionType, ActionType | None):
        """Determine the behavior based on the current state"""
        return self.Num2ActionReq[0], self.Num2ActionReq[0]
        return self.Num2ActionReq[random.randint(0, self.actionNum)]


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


    def step(self, action1:ActionType, action2:ActionType.SILENT) -> None:
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
            print(f"player_id: {self.player_id}")
            self.player_info = player_data
            return MapBlock.PlayerSelf
        else:
            self.enemy_id = player.player_id
            print(f"enemy_id: {self.enemy_id}")
            self.enemy_info = player_data
            return MapBlock.Enemy
        

    def getPLayersDistance(self):
        """calculates the distance between players and returns two values: 
            one is the distance including destructible blocks, 
            and the other is the distance considering all as walls."""
        pass


    def bfs(self, start:tuple, max_depth:int=None) -> None:
        pass


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
                        self.map[x, y, self.updatePlayer(obj.property, [x, y])] = 1

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
        bot = Bot(show_ui=False)
        bot.setCLient(client)
        bot.run(out=False)


if __name__ == "__main__":
    # test()
    main()