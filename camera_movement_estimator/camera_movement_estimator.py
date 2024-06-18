import os
import pickle
import cv2
import numpy as np
import sys
sys.path.append('../')
from utils import measure_distance, measure_xy_distance


class CameraMovementEstimator():
    def __init__(self,frame):
        self.minimum_distance=5
        self.lk_parama=dict(
            winSize=(15,15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
        )
        first_frame_grayscale=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask_features=np.zeros_like(first_frame_grayscale)
        mask_features[:,0:20]=1
        mask_features[:,900:1050]=1
        self.feature=dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features,
        )

    def add_adjust_positions_to_tracks(self,tracks,camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted


    def get_camera_movement(self, frames,read_from_stub=False,stub_path=None):
        
        #read the stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                camera_movement = pickle.load(f)
            return camera_movement

        camera_movement = [(0,0)]*len(frames)

        old_gray=cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
        
        old_features=cv2.goodFeaturesToTrack(old_gray,**self.feature)

        for frame_num in range(1,len(frames)):
            frame_gray=cv2.cvtColor(frames[frame_num],cv2.COLOR_BGR2GRAY)
            new_features,_,_ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None,**self.lk_parama)
            max_dist=0
            camera_movement_x,camera_movement_y=0,0
            for i,(new,old) in enumerate(zip(new_features,old_features)):
                new_features_points=new.ravel()
                old_features_points=old.ravel()
                dist=measure_distance(new_features_points,old_features_points)
                if dist>max_dist:
                    max_dist=dist
                    camera_movement_x,camera_movement_y=measure_xy_distance(new_features_points,old_features_points)

            if max_dist>self.minimum_distance:
                camera_movement[frame_num]=[camera_movement_x,camera_movement_y]
                old_features=cv2.goodFeaturesToTrack(frame_gray,**self.feature)

            old_gray=frame_gray.copy()

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f)

        return camera_movement
    
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            # frame = frame.copy()
            overlay = frame
            
            # Create a semi-transparent rectangle at the bottom-left corner
            cv2.rectangle(overlay, (20, frame.shape[0] - 120), (300, frame.shape[0] - 20), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            x_movement, y_movement = camera_movement_per_frame[frame_num]
            
            # Draw camera movement information
            cv2.putText(frame, "Camera Movement:", (30, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(frame, f"X: {x_movement:.2f}", (30, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(frame, f"Y: {y_movement:.2f}", (30, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            output_frames.append(frame)
        return output_frames