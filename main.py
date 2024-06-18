from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from player_ball_assigner import PlayerBallAssigner
from team_assigner import TeamAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import viewTransformer
from speed_and_distance_estimator import speedAndDistanceEstimator

def main():
    # Read video
    video_frames=read_video('input_videos/08fd33_4.mp4')


    #intialise tracker
    tracker=Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    #get object position
    tracker.add_position_to_tracks(tracks)
    
    # estimate camera movement
    camera_movement_estimator=CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame=camera_movement_estimator.get_camera_movement(video_frames,
                                                                  read_from_stub=True,
                                                                  stub_path='stubs/camera_movement_stub.pkl')

    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

    #view transformer
    view_transformer=viewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    
    #interpolate ball positions
    # tracks['ball']=tracker.interpolate_ball_positions(tracks['ball'])

    #speed and distance estimator
    speed_and_distance_estimator=speedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)


    #assign player team
    team_assigner=TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],tracks["players"][0])

    for frame_num,player_track in enumerate(tracks["players"]):
        for player_id,track in player_track.items():
            team=team_assigner.get_player_team(video_frames[frame_num],track["box"],player_id)
            tracks["players"][frame_num][player_id]["team"]=team
            tracks["players"][frame_num][player_id]["team_color"]=team_assigner.team_colors[team]
    
    #save cropped image of a player
    # for track_id,player in tracks["players"][0].items():
    #     box=player["box"]
    #     frame=video_frames[0]
    #     #crop bounding box from frame
    #     cropped_image=frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
    #     #save cropeed image
    #     cv2.imwrite(f'output_videos/cropped_image.jpg',cropped_image)
    #     break

    #assign ball aquisition
    player_assigner=PlayerBallAssigner()
    team_ball_control=[]
    for frame_num, player in enumerate(tracks["players"]):
        for track_id, ball_box in tracks["ball"][frame_num].items():
            assigned_player = player_assigner.assign_ball_to_player(player, ball_box['box'])
            if assigned_player != -1:
                tracks["players"][frame_num][assigned_player]["has_ball"] = True
                team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
            else:
                team_ball_control.append(team_ball_control[-1])
       
    team_ball_control=np.array(team_ball_control)

    #draw output
    #draw object tracks

    output_video_frames=tracker.draw_annotations(video_frames,tracks,team_ball_control)

    #draw camera movemnt
    output_video_frames=camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)
    
    #draw speed and distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)
    
    # Save video
    save_video(output_video_frames,'output_videos/output_video.avi')

if __name__ == '__main__':
    main()