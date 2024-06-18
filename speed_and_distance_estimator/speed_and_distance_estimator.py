import numpy as np
import cv2
import sys
sys.path.append('../')
from utils import measure_distance,get_foot_position
class speedAndDistanceEstimator():
    def __init__(self):
        self.frame_window=5
        self.frame_rate=24

    def add_speed_and_distance_to_tracks(self,tracks):
        total_distance={}

        for object,object_tracks in tracks.items():
            if object=="ball" or object=="refree":
                continue
            number_of_frames=len(object_tracks)

            for frame_num in range(0,number_of_frames,self.frame_window):
                last_frame=min(frame_num+self.frame_window,number_of_frames-1)
                for track_id,_ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position=object_tracks[frame_num][track_id]["position_transformed"]
                    end_position=object_tracks[last_frame][track_id]["position_transformed"]

                    if start_position is None or end_position is None:
                        continue

                    distance_covered=measure_distance(start_position,end_position)
                    time_elapsed=(last_frame-frame_num)/self.frame_rate

                    speed_mps=distance_covered/time_elapsed
                    speed_kmph=speed_mps*3.6
                    
                    if object not in total_distance:
                        total_distance[object]={}

                    if track_id not in total_distance[object]:
                        total_distance[object][track_id]=0
                    
                    total_distance[object][track_id]+=distance_covered

                    for frame_num_batch in range(frame_num,last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]["speed"]=speed_kmph
                        tracks[object][frame_num_batch][track_id]["distance"]=total_distance[object][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            # frame_copy = frame.copy()
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue
                for _, track_info in object_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)
                        if speed is None or distance is None:
                            continue

                        box = track_info['box']
                        position = get_foot_position(box)
                        position = list(position)
                        position[1] += 40  # Move the box above the player

                        # Create a semi-transparent rectangle overlay
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (position[0] - 80, position[1]), (position[0] + 80, position[1] + 40),
                                    (255, 255, 255), -1)
                        alpha = 0.6
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                        # Draw the speed and distance information inside the rectangle
                        cv2.putText(frame, f"{speed:.2f} km/h", (position[0] - 70, position[1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        cv2.putText(frame, f"{distance:.2f} m", (position[0] - 70, position[1] + 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            output_frames.append(frame)

        return output_frames