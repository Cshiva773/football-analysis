import sys
sys.path.append('../')
from utils import get_center_of_box,get_box_width,measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance=69

    def assign_ball_to_player(self,players,ball_box):
        ball_position=get_center_of_box(ball_box)
        min_distance=999999
        assign_player=-1
        for player_id,player in players.items():
            player_box=player["box"]
            distance_left=measure_distance((player_box[0],player_box[-1]),ball_position)
            distance_right=measure_distance((player_box[2],player_box[-1]),ball_position)
            distance=min(distance_left,distance_right)
            if distance<self.max_player_ball_distance and distance<min_distance:
                min_distance=distance
                assign_player=player_id
        return assign_player
