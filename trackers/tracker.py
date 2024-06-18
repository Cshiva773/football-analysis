from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import sys
import numpy as np
import pandas as pd
sys.path.append('../')

from utils import get_center_of_box,get_box_width,get_foot_position

class Tracker:
    def __init__(self,model_path):
        self.model=YOLO(model_path)
        self.tracker=sv.ByteTrack()

    def add_position_to_tracks(self,tracks):
        for object,object_tracks in tracks.items():
            for frame_num,track in enumerate(object_tracks):
                for track_id,track_info in track.items():
                    box=track_info['box']
                    if object =='ball':
                        position=get_center_of_box(box)
                    else:
                        position=get_foot_position(box)

                    tracks[object][frame_num][track_id]['position']=position



    # def interpolate_ball_positions(self,ball_positions):
    #     ball_positions=[x.get(1,{}).get('box',[]) for x in ball_positions]
    #     df_ball_positions=pd.DataFrame(ball_positions,columns=["x1","y1","x2","y2"])

    #     #interpolate missing values
    #     df_ball_positions=df_ball_positions.interpolate()
    #     df_ball_positions=df_ball_positions.bfill()

    #     ball_positions = [{1: {"box":x}} for x in df_ball_positions.to_numpy().tolist()]

    #     return ball_positions

    def detect_frames(self,frames):
        batch_size=20
        detections=[]
        for i in range(0,len(frames),batch_size):
            detections_batch=self.model.predict(frames[i:i+batch_size],conf=0.1)    
            detections+=detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections=self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num,detections in enumerate(detections):
            cls_names=detections.names
            cls_names_inv={v:k for k,v in cls_names.items()}
            #convert to supervision detection format 
            detection_supervision=sv.Detections.from_ultralytics(detections)

            #convert goalkeeper to player
            for object_ind,class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id]=="goalkeeper":
                    detection_supervision.class_id[object_ind]=cls_names_inv["player"]

            #track objects
            detection_with_tracks=self.tracker.update_with_detections(detection_supervision)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                box=frame_detection[0].tolist()
                cls_id=frame_detection[3]
                track_id=frame_detection[4]

                if cls_id==cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id]={"box":box}

                if cls_id==cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id]={"box":box}

            for frame_detection in detection_supervision:
                box=frame_detection[0].tolist()
                cls_id=frame_detection[3]

                if cls_id==cls_names_inv["ball"]:
                    tracks["ball"][frame_num][track_id]={"box":box}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return  tracks
    
    def draw_ellipse(self, frame, box, color, track_id=None):
        y2 = int(box[3])
        x_center, _ = get_center_of_box(box)
        width = get_box_width(box)

        print(f"width: {width}, x_center: {x_center}, y2: {y2}")


        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )
    

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,(int(x1_rect), int(y1_rect)),(int(x2_rect), int(y2_rect)),color,cv2.FILLED)

            x1_text=x1_rect+12
            if track_id>99:
                x1_text-=10
            cv2.putText(frame,f"{track_id}",(int(x1_text),int(y1_rect+15)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
        return frame

    def draw_triangle(self,frame,box,color):
        y=int(box[1])
        x,_=get_center_of_box(box)
        triangle_points=np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
        ])
        cv2.drawContours(frame,[triangle_points],0,color,cv2.FILLED)
        cv2.drawContours(frame,[triangle_points],0,(0,0,0),2)
        return frame

   

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
    # Draw a semi-transparent rectangle at the top-left corner
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (320, 160), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, "Team Ball Control:", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, f"Team 1: {team_1*100:.2f}%", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, f"Team 2: {team_2*100:.2f}%", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return frame


    def draw_annotations(self,video_frames, tracks, team_ball_control):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame=frame.copy()

            player_dict=tracks["players"][frame_num]
            ball_dict=tracks["ball"][frame_num]
            referee_dict=tracks["referees"][frame_num]

            #draw players
            for track_id,player in player_dict.items():
                color=player.get("team_color",(0,0,255))
                frame=self.draw_ellipse(frame,player["box"],color,track_id)

                if player.get('has_ball',False):
                    frame=self.draw_triangle(frame,player["box"],(255,255,255))


            #draw referees
            for _,referee in referee_dict.items():
                frame=self.draw_ellipse(frame,referee["box"],(0,0,255))
            
            #draw ball
            for track_id,ball in ball_dict.items():
                frame=self.draw_triangle(frame,ball["box"],(255,0,0))

            #draw team ball control
            frame=self.draw_team_ball_control(frame,frame_num,team_ball_control)

            output_video_frames.append(frame)
        
        return output_video_frames
