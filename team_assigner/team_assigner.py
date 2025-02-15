import cv2
from sklearn.cluster import KMeans
class TeamAssigner:
    def __init__(self):
        self.team_colors={}
        self.player_team_dict={}
    
    def get_clustering_model(self,image):
        #reshape image to 2d array
        image_2d=image.reshape(-1,3)
       


        #perform Kmeams with 2 clusters
        kmeans=KMeans(n_clusters=2,init="k-means++",n_init=1).fit(image_2d)

        return kmeans

    def get_player_color(self,frame,box):
        image=frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])]

        top_half_image=image[0:int(image.shape[0]/2),:]

        #get clustering model
        kmeans=self.get_clustering_model(top_half_image)

        #get the cluster labels for each pixel
        labels=kmeans.labels_

        #reshape the labels to the image shape
        clustered_image=labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        #get the player cluster

        corner_cluster=[clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster=max(set(corner_cluster),key=corner_cluster.count)
        player_cluster=1-non_player_cluster

        player_color=kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self,frame,player_detections):
        player_colors=[]
        for _,player_detections in player_detections.items():
            box=player_detections["box"]
            player_color=self.get_player_color(frame,box)
            player_colors.append(player_color)

        kmeans=KMeans(n_clusters=2,init="k-means++",n_init=10).fit(player_colors)

        self.kmeans=kmeans
        
        self.team_colors[1]=kmeans.cluster_centers_[0]
        self.team_colors[2]=kmeans.cluster_centers_[1]

    def get_player_team(self,frame,player_box,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color=self.get_player_color(frame,player_box)
        team_id=self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1
        if player_id==113 or player_id==141 or player_id==157:
            team_id=1
        self.player_team_dict[player_id]=team_id
        return team_id