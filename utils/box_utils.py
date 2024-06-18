def get_center_of_box(box):
    x1,y1,x2,y2=box
    return int((x1+x2)/2),int((y1+y2)/2) 

def get_box_width(box):
    return box[2]-box[0]

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]

def get_foot_position(box):
    x1,y1,x2,y2=box
    return int((x1+x2)/2),int(y2)