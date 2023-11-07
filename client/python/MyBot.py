import json
import random
import os
from base import *
from req import *
from resp import *
from client import *
from ui import UI
from config import config
from time import sleep


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
    Unknown = 9


class GameStatus:
    Waiting = 0
    Starting = 1
    End = 2
    Win = 3
    Fail = 4


class Bot:
    """My silly bot"""

    def __init__(self, client:Client=None, map_size:int=15, actionNum:int=5, showUi:bool=False) -> None:
        """initialize game data"""
        self.client = client
        self.resp:ActionResp
        if showUi:
            self.ui = UI()
        self.gameStatus = GameStatus.Waiting

        self.map = [[[MapBlock.Null] for _ in range(map_size)] for _ in range(map_size)]
        self.map_size = map_size
        self.player_info:dict
        self.previous_score:int = 0
        self.player_id = -1
        self.enemy_info:dict
        self.enemy_id = -1

        self.actionNum = actionNum
        self.Num2ActionReq = [
            ActionType.SILENT,
            ActionType.MOVE_LEFT,
            ActionType.MOVE_RIGHT,
            ActionType.MOVE_UP,
            ActionType.MOVE_DOWN,
            ActionType.PLACED,
        ]


    def receive(self) -> None:
        """receive data from server"""
        self.resp = self.client.recv()

        if self.gameStatus == GameStatus.Waiting and self.resp.type == PacketType.ActionResp:
            self.gameStatus = GameStatus.Starting
            self.player_id = self.resp.data.player_id
        elif self.resp.type == PacketType.GameOver:
            self.gameStatus = GameStatus.End

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

            state = self.getState()
            action = self.choose_action(state)
            self.step(action)
            self.receive()
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


    def getState(self) -> list:
        """Organize environmental data"""
        # TODO: simplified state parameters
        pass
        return None


    def choose_action(self, state=None) -> ActionType:
        """Determine the behavior based on the current state"""
        # TODO
        return self.Num2ActionReq[0]
        return self.Num2ActionReq[random.randint(0, self.actionNum)]


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


    def updateMap(self, map_data:dict) -> dict:
        """Update map and player positions"""
        self.map = [[[MapBlock.Null] for _ in range(self.map_size)] for _ in range(self.map_size)]
        for cell in map_data:
            x = cell.x
            y = cell.y
            
            if not cell.objs:
                cell_content = [MapBlock.Null]
            else:
                cell_content = []
                for obj in cell.objs:
                    if obj.type == ObjType.Player:
                        cell_content.append(self.updatePlayer(obj.property, [y, x]))

                    elif obj.type == ObjType.Bomb:
                        # The impact range of the bomb explosion
                        bomb_range = obj.property.bomb_range

                        for _x in range(x - bomb_range, x + bomb_range + 1):
                            if 0 <= _x < self.map_size and MapBlock.Wall not in self.map[y][_x] and MapBlock.BombCover not in self.map[y][_x]:
                                self.map[_x][y].append(MapBlock.BombCover)

                        for _y in range(y - bomb_range, y + bomb_range + 1):
                            if 0 <= _y < self.map_size and MapBlock.Wall not in self.map[_y][x] and MapBlock.BombCover not in self.map[_y][x]:
                                self.map[x][_y].append(MapBlock.BombCover)

                    elif obj.type == ObjType.Item:
                        # TODO: process Item
                        pass

                    elif obj.type == ObjType.Block:
                        if obj.property.removable:
                            cell_content.append(MapBlock.Barrier)
                        else:
                            cell_content.append(MapBlock.Wall)

                    else:
                        cell_content.append(MapBlock.Unknown)  # Unknown Obj

            # Remove redundant MapBlock.Null
            self.map[x][y] += cell_content
            if MapBlock.Null in self.map[x][y] and len(self.map[x][y]) > 1:
                self.map[x][y].remove(MapBlock.Null)
        
        return self.map

    
    def printMap(self) -> None:
        for _ in self.map:
            print(_)


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
        bot = Bot(client=client, showUi=True)
        bot.run(output=True)



if __name__ == "__main__":
    # test()
    main()