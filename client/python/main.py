import json
import socket
import random
from base import *
from req import *
from resp import *
from config import config
from ui import UI
import subprocess
import logging
from threading import Thread
from itertools import cycle
from time import sleep
from logger import logger

import sys
import termios
import tty

# record the context of global data
gContext = {
    "playerID": -1,
    "gameOverFlag": False,
    "prompt": (
        "Take actions!\n"
        "'w': move up\n"
        "'s': move down\n"
        "'a': move left\n"
        "'d': move right\n"
        "'blank': place bomb\n"
    ),
    "steps": ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"],
    "gameBeginFlag": False,
}

resp = ''


class Client(object):
    """Client obj that send/recv packet.
    """
    def __init__(self) -> None:
        self.config = config
        self.host = self.config.get("host")
        self.port = self.config.get("port")
        assert self.host and self.port, "host and port must be provided"
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._connected = False

    def connect(self):
        if self.socket.connect_ex((self.host, self.port)) == 0:
            logger.info(f"connect to {self.host}:{self.port}")
            self._connected = True
        else:
            logger.error(f"can not connect to {self.host}:{self.port}")
            exit(-1)
        return

    def send(self, req: PacketReq):
        msg = json.dumps(req, cls=JsonEncoder).encode("utf-8")
        length = len(msg)
        self.socket.sendall(length.to_bytes(8, sys.byteorder) + msg)
        # uncomment this will show req packet
        # logger.info(f"send PacketReq, content: {msg}")
        return

    def recv(self):
        length = int.from_bytes(self.socket.recv(8), sys.byteorder)
        result = b""
        while resp := self.socket.recv(length):
            result += resp
            length -= len(resp)
            if length <= 0:
                break

        # uncomment this will show resp packet
        # logger.info(f"recv PacketResp, content: {result}")
        packet = PacketResp().from_json(result)
        return packet

    def __enter__(self):
        return self
    
    def close(self):
        logger.info("closing socket")
        self.socket.close()
        logger.info("socket closed successfully")
        self._connected = False
    
    @property
    def connected(self):
        return self._connected

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if traceback:
            print(traceback)
            return False
        return True


def cliGetInitReq():
    """Get init request from user input."""
    # input("enter to start!")
    return InitReq(config.get("player_name"))


def recvAndRefresh(ui: UI, client: Client):
    """Recv packet and refresh ui."""
    global gContext
    global resp
    resp = client.recv()

    if resp.type == PacketType.ActionResp:
        gContext["gameBeginFlag"] = True
        gContext["playerID"] = resp.data.player_id
        ui.player_id = gContext["playerID"]

    while resp.type != PacketType.GameOver:
        # subprocess.run(["clear"])
        ui.refresh(resp.data)
        ui.display()
        resp = client.recv()

    print(f"Game Over!")

    print(f"Final scores \33[1m{resp.data.scores}\33[0m")

    if gContext["playerID"] in resp.data.winner_ids:
        print("\33[1mCongratulations! You win! \33[0m")
    else:
        print(
            "\33[1mThe goddess of victory is not on your side this time, but there is still a chance next time!\33[0m"
        )

    gContext["gameOverFlag"] = True
    print("press any key to quit")



key2ActionReq = {
    'w': ActionType.MOVE_UP,
    's': ActionType.MOVE_DOWN,
    'a': ActionType.MOVE_LEFT,
    'd': ActionType.MOVE_RIGHT,
    ' ': ActionType.PLACED,
}


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


Num2ActionReq = [
    ActionType.SILENT,
    ActionType.MOVE_LEFT,
    ActionType.MOVE_RIGHT,
    ActionType.MOVE_UP,
    ActionType.MOVE_DOWN,
    ActionType.PLACED,
]


class Bot:
    """My silly bot"""

    def __init__(self, client:Client=None, map_size:int=15) -> None:
        """initialize game data"""
        self.client = client
        self.resp:ActionResp
        self.map = [[[MapBlock.Null] for _ in range(map_size)] for _ in range(map_size)]
        self.map_size = map_size
        self.player_info:dict
        self.player_id = -1
        self.enemy_info:dict
        self.enemy_id = -1

    
    def setPlayerId(self, player_id:int=1, enemy_id:int=0) -> None:
        """DEBUG"""
        self.player_id = player_id
        self.enemy_id = enemy_id


    def run(self) -> None:
        """start game"""
        pass


    def updatePlayer(self, player:dict, pos:list) -> MapBlock:
        """Parse the player data and return the current player"""
        player_data = {
            "alive": player["alive"],
            "bomb_max_num": player["bomb_max_num"],
            "bomb_now_num": player["bomb_now_num"],
            "bomb_range": player["bomb_range"],
            "hp": player["hp"],
            "invincible_time": player["invincible_time"],
            "score": player["score"],
            "shield_time": player["shield_time"],
            "speed": player["speed"],
            "pos": pos
        }

        if player["player_id"] == self.player_id:
            self.player_info = player_data
            return MapBlock.PlayerSelf
        elif player["player_id"] == self.enemy_id:
            self.enemy_info = player_data
            return MapBlock.Enemy
        else:
            assert False, 'updatePlayer: Player Id Error!'
        

    def getPLayersDistance(self):
        """calculates the distance between players and returns two values: 
            one is the distance including destructible blocks, 
            and the other is the distance considering all as walls."""
        pass


    def updateMap(self, map_data:dict) -> dict:
        """Update map and player positions"""
        for cell in map_data:
            x = cell['x']
            y = cell['y']
            
            if not cell['objs']:
                cell_content = [MapBlock.Null]
            else:
                cell_content = []
                for obj in cell['objs']:
                    if obj['type'] == ObjType.Player:
                        cell_content.append(self.updatePlayer(obj["property"], [y, x]))

                    elif obj['type'] == ObjType.Bomb:
                        # The impact range of the bomb explosion
                        bomb_range = obj["property"]["bomb_range"]

                        for _x in range(x - bomb_range, x + bomb_range + 1):
                            if MapBlock.Wall not in self.map[y][_x] and MapBlock.BombCover not in self.map[y][_x]:
                                self.map[y][_x].append(MapBlock.BombCover)

                        for _y in range(y - bomb_range, y + bomb_range + 1):
                            if MapBlock.Wall not in self.map[_y][x] and MapBlock.BombCover not in self.map[_y][x]:
                                self.map[_y][x].append(MapBlock.BombCover)

                    elif obj['type'] == ObjType.Item:
                        # TODO: process Item
                        pass

                    elif obj['type'] == ObjType.Block:
                        if obj["property"]["removable"]:
                            cell_content.append(MapBlock.Barrier)
                        else:
                            cell_content.append(MapBlock.Wall)

                    else:
                        cell_content.append(MapBlock.Unknown)  # Unknown Obj

            self.map[y][x] += cell_content
            if MapBlock.Null in self.map[y][x] and len(self.map[y][x]) > 1:
                self.map[y][x].remove(MapBlock.Null)
        
        return self.map



def main():
    bot = Bot()
    with open("resp.json", 'r') as f_obj:
        resp = json.load(f_obj)

    bot.setPlayerId()
    result = bot.updateMap(resp['data']["map"])

    for _ in result:
        print(_)



def termPlayAPI():
    ui = UI()
    
    with Client() as client:
        client.connect()
        
        initPacket = PacketReq(PacketType.InitReq, cliGetInitReq())
        client.send(initPacket)

        # IO thread to display UI
        t = Thread(target=recvAndRefresh, args=(ui, client))
        t.start()
        
        print(gContext["prompt"])
        for c in cycle(gContext["steps"]):
            if gContext["gameBeginFlag"]:
                break
            print(
                f"\r\033[0;32m{c}\033[0m \33[1mWaiting for the other player to connect...\033[0m",
                flush=True,
                end="",
            )
            sleep(0.1)

        # starting game!!!
        while not gContext["gameOverFlag"]:
            # key = scr.getch()
            # old_settings = termios.tcgetattr(sys.stdin)
            # tty.setcbreak(sys.stdin.fileno())
            # key = sys.stdin.read(1)
            # termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            
            with open('resp.json', 'w') as f_obj:
                f_obj.write(str(resp))

            action = ActionReq(gContext["playerID"], Num2ActionReq[random.randint(0, 4)])
            
            # if key in key2ActionReq.keys():
            #     action = ActionReq(gContext["playerID"], key2ActionReq[key])
            # else:
            #     action = ActionReq(gContext["playerID"], ActionType.SILENT)
            
            if gContext["gameOverFlag"]:
                break
            
            actionPacket = PacketReq(PacketType.ActionReq, action)
            client.send(actionPacket)


if __name__ == "__main__":
    # termPlayAPI()
    main()