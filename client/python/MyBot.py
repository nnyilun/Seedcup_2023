from base import *
from req import *
from resp import *
from client import *
from ui import UI
from config import config
from ppo import *
import random
from time import sleep
import numpy as np
from collections import deque
from itertools import product


TEAM_ID = '1'


class MapBlock:
    Null = 0
    Wall = 1
    Barrier = 2
    BombCover = 3
    Item = 4
    PLayer = 5
    Enemy = 6


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
            map_size:int=15, block_num:int=7, action_num:int=6*6, 
            player_feature_num:int=8, bomb_base_time:int=5, bomb_base_num:int=2,
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
        action_pairs = list(product(self.Num2ActionReq, repeat=2))
        self.Num2ActionReq2 = [list(pair) for pair in action_pairs]

        self.ActionReq2Num = {
            ActionType.SILENT : "SILENT",
            ActionType.MOVE_LEFT : "MOVE_LEFT",
            ActionType.MOVE_RIGHT : "MOVE_RIGHT",
            ActionType.MOVE_UP : "MOVE_UP",
            ActionType.MOVE_DOWN : "MOVE_DOWN",
            ActionType.PLACED : "PLACED",
        }
        self.DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)] # search directions 

        # PPO agent
        self.memory = deque()
        self.num_episodes = 1000
        self.agent = PPOAgent(block_num, action_num, player_feature_num)

        # Game Data
        self.round = 0
        self.bomb_timing = np.full((map_size, map_size), np.inf)
        self.map = np.zeros((map_size, map_size, block_num))

        self.previous_score:int = 0
        self.bomb_now_num:int = 0
        self.pre_pos:list = [0, 0]
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
        self.pre_pos = self.player_info['pos']
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


    def isDone(self) -> bool:
        return not self.gameStatus == GameStatus.Starting
    

    def reset(self) -> None:
        self.gameStatus = GameStatus.Waiting
        self.round = 0
        self.map = np.zeros((self.map_size, self.map_size, self.block_num))

        self.previous_score:int = 0
        self.bomb_now_num:int = 0
        self.pre_pos:list = [0, 0]
        self.player_info:dict
        self.player_info_array:np.array
        self.player_id = -1

        self.enemy_info:dict
        self.enemy_info_array:np.array
        self.enemy_id = -1

        self.client = Client()

        self.joinGame()
        self.waitToStart()

    
    def run_ppo(self, out:bool=False) -> None:
        self.reset()

        for _ in range(self.num_episodes):
            for __ in range(self.agent.ppo_steps // 2):
                self.output(out)
                map_state = self.map
                player1_features = self.player_info_array
                player2_features = self.enemy_info_array

                action1, value1, log_prob1 = self.agent.select_action(map_state, player1_features, player2_features)
                # action2, value2, log_prob2 = self.agent.select_action(map_state, player1_features, player2_features)

                print(f"actions: {self.ActionReq2Num[self.Num2ActionReq2[action1][0]]}, {self.ActionReq2Num[self.Num2ActionReq2[action1][1]]}")

                self.step(*self.Num2ActionReq2[action1])
                self.receive()

                done = self.isDone()
                reward = self.getReward(done)
                print(f'reward:{reward}')

                full_state = (map_state, player1_features, player2_features)
                self.memory.append((full_state, action1, reward, self.map, log_prob1, value1, done))
                # self.memory.append((full_state, action2, reward, self.map, log_prob2, value2, done))

                if done:
                    print('done')
                    self.gameOver()
                    self.reset()
                    break

            self.agent.train(self.memory)
            self.memory.clear()


    def step(self, action1:ActionType, action2:ActionType=ActionType.SILENT) -> None:
        """Sending data to the server, making an action"""
        actionPacket = PacketReq(PacketType.ActionReq, [ActionReq(self.player_id, action1), ActionReq(self.player_id, action2)])
        self.client.send(actionPacket)


    def getReward(self, isDone:bool=False) -> int:
        if isDone:
            scores = self.resp.data.scores
            for _ in scores:
                if _['player_id'] == self.player_id:
                    reward = (_['score'] - self.previous_score) * 5
        else:
            reward = (self.player_info['score'] - self.previous_score) * 10

            self.previous_score = self.player_info['score'] 
            print(f'now:{self.bomb_now_num}, pla:{self.player_info["bomb_now_num"]}')
            if self.player_info["bomb_now_num"] > self.bomb_now_num:
                self.bomb_now_num = self.player_info["bomb_now_num"]
                return reward + 200
            elif self.player_info["bomb_now_num"] < self.bomb_now_num:
                self.bomb_now_num = self.player_info["bomb_now_num"]
                return reward + 50000
            elif self.pre_pos == self.player_info['pos']:
                return reward - 2000
            elif reward == 0:
                self.pre_pos = self.player_info['pos']
                return reward - 50
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

        # 位置信息已经存储在地图，玩家id也不需要储存
        values = np.array([
                            player.alive, 
                            player.hp,
                            player.bomb_max_num, player.bomb_now_num, player.bomb_range,
                            player.invincible_time, player.shield_time, player.speed,
                        ])

        if player.player_id == self.player_id:
            self.player_info = player_data
            self.player_info_array = values
            return MapBlock.PLayer
        else:
            self.enemy_info = player_data
            self.enemy_info_array = values
            return MapBlock.Enemy


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
        bot = Bot(client=client, show_ui=False)
        bot.run(out=False)


def train():
    bot = Bot()
    bot.run_ppo()


if __name__ == "__main__":
    # test()
    # main()
    train()