#!/usr/bin/env python
#-*- coding: utf-8 -*-
import numpy as np
import cv2
import kociemba
import roslib
import rospy
from std_msgs.msg import Float64
from std_msgs.msg import String
from dynamixel_12dof_arm.msg import Num
import time
import pdb
from plotting import load_data,load_data2
from time import sleep

def draw_edge(src):
    # create point
    """point = [[(110,100),(160,80),(240,40),(330,10),(420,45),(500,75),(560,90)],
             [(100,180),(165,165),(240,150),(330,120),(420,150),(505,170),(560,180)],
             [(100,300),(165,290),(230,280),(330,270),(420,280),(505,290),(560,300)],
             [(100,390),(155,400),(245,405),(335,410),(427,405),(510,400),(570,390)]]
             """
    for i in range(4):
        for j in range(7):
            if i < 3:
                cv2.line(src,point[i][j],point[i+1][j],(0,0,0),1)# 縦線
            if j < 6:
                cv2.line(src,point[i][j],point[i][j+1],(0,0,0),1)# 横線
def get_average(src, region):
    pt1 = region[0]
    pt2 = region[1]
    pt3 = region[2]
    pt4 = region[3]
    x_i = max(pt1[0],pt4[0])
    x_e = min(pt2[0],pt3[0])
    y_i = max(pt1[1],pt2[1])
    y_e = min(pt3[1],pt4[1])
    w = 12 #幅
    h = 25
    p = np.array([0,0,0])
    counter = 0
    for y in range(y_i+h,y_e-h):
        for x in range(x_i+w,x_e-w):
            if (src[y][x][0] > 120) and (src[y][x][1]<200):
                src[y][x][0] -= 120
            p[0] += src[y][x][0]
            p[1] += src[y][x][1]
            p[2] += src[y][x][2]
            counter += 1
    p = p / counter
    for y in range(y_i+h,y_e-h):
        for x in range(x_i+w,x_e-w):
            src[y][x][0] = p[0]
            src[y][x][1] = p[1]
            src[y][x][2] = p[2]
    return p
def filling_poly(src):
    for i in range(3):
        for j in range(6):
            pts = np.array(region[i][j])
            p = tuple(get_average(src,region[i][j]))
            cv2.fillPoly(src,pts=[pts],color=p)
def create_points(external_point):
    def split_line(p,q,r1,r2,r3):
        a = ((r2+r3)*p + r1*q)/(r1+r2+r3)
        b = (r3*p + (r1+r2)*q)/(r1+r2+r3)
        a = tuple(a.astype(int))
        b = tuple(b.astype(int))
        return a,b
    points = np.zeros(28).reshape(4,7).tolist()
    UL = points[0][0] = external_point[0];
    UM = points[0][3] = external_point[1]
    UR = points[0][6] = external_point[2]
    DL = points[3][0] = external_point[3]
    DM = points[3][3] = external_point[4]
    DR = points[3][6] = external_point[5]
    points[0][1], points[0][2] = split_line(np.array(UL),np.array(UM),1.3, 1.7, 2)#左上横線上の点
    points[0][4], points[0][5] = split_line(np.array(UM),np.array(UR),2, 1.7, 1.3)#右上横線
    points[1][0], points[2][0] = split_line(np.array(UL),np.array(DL),1.9, 2.2, 1.9)#左縦線
    points[1][3], points[2][3] = split_line(np.array(UM),np.array(DM),2.3, 3.2, 2.5)#真ん中縦線
    points[1][6], points[2][6] = split_line(np.array(UR),np.array(DR),1.9, 2.2, 1.9)#右縦線
    for i in range(4):
        points[i][1],points[i][2] = split_line(np.array(points[i][0]),np.array(points[i][3]),1.3, 1.7, 2)
        points[i][4],points[i][5] = split_line(np.array(points[i][3]),np.array(points[i][6]),2, 1.7, 1.3)
    return points
def create_regions(points):
    region = np.zeros(18).reshape(3,6).tolist()
    for i in range(3):
        for j in range(6):
            region[i][j] = [point[i][j],point[i][j+1],point[i+1][j+1],point[i+1][j]]
    return region
def get_sample(src,region):
    arr = np.array([[]])
    for r in range(1):
        for i in range(3):
            #for j in range(3):
            for j in range(3,6):
                p = get_average(src,region[i][j])
                print("{},".format(p))
                arr = np.append(arr,p)
def get_sample_left(src,region):
    arr = np.array([[]])
    print("")
    for r in range(1):
        for i in range(3):
            #for j in range(3):
            for j in range(3):
                p = get_average(src,region[i][j])
                p = np.array(p)
                print("[{},{},{}],".format(p[0],p[1],p[2]))
                arr = np.append(arr,p)
    print("")
def get_sample_right(src,region):
    arr = np.array([[]])
    print("")
    for r in range(1):
        for i in range(3):
            #for j in range(3):
            for j in range(3,6):
                p = get_average(src,region[i][j])
                p = np.array(p)
                print("[{},{},{}],".format(p[0],p[1],p[2]))
                arr = np.append(arr,p)
    print("")


def load_data():
    white = np.array([[87,14, 152],
    [ 75,  24, 141],
    [ 57,  23, 126],
    [ 71,  18, 147],
    [ 62,  31, 127],
    [ 62,  28, 121],
    [ 75,  19, 156],
    [ 66,  21, 151],
    [ 75,  24, 143],
    [120,  22, 146],
    [116,  21, 156],
    [116,  19, 162],
    [118,  27, 152],
    [110,  42, 149],
    [113,  19, 164],
    [123,  22, 156],
    [115,  23, 164],
    [119,  17, 168],])
    orange= np.array([[10, 152,108],
    [ 9 ,160 ,116],
    [ 8 ,171 ,124],
    [12 ,128 ,100],
    [11 ,136 ,108],
    [14 ,144 ,109],
    [ 9 ,136 ,108],
    [ 7 ,148 ,121],
    [ 8 ,158 ,126],
    [ 9 ,178 ,119],
    [ 10, 176, 112],
    [ 20, 153, 105],
    [ 13, 168, 113],
    [ 10, 175, 111],
    [ 12, 162, 100],
    [  8, 180, 133],
    [  9, 176, 120],
    [ 11, 153, 106],])
    green = np.array([[76, 146,  85],
    [74, 146  ,87],
    [75, 157  ,92],
    [77, 134  ,84],
    [75, 137  ,86],
    [75, 147  ,91],
    [78, 141  ,89],
    [78, 149  ,94],
    [77, 148  ,95],
    [77, 153  ,88],
    [76, 153  ,86],
    [74, 147  ,78],
    [77, 155  ,90],
    [76, 148  ,86],
    [75, 147  ,83],
    [78, 145  ,94],
    [78, 143  ,92],
    [80, 133  ,88],])
    red   = np.array([[5, 160,  90],
    [ 7, 155, 83],
    [10, 143, 77],
    [ 9, 151, 85],
    [10, 142, 79],
    [13, 135, 78],
    [15, 147, 89],
    [13, 145, 88],
    [39, 121, 88],
    [18, 126, 77],
    [14, 138, 85],
    [12, 147, 89],
    [76, 104, 79],
    [53, 112, 80],
    [32, 138, 89],
    [93,  95, 81],
    [62, 111, 84],
    [62, 132, 92],])
    blue  = np.array([[ 102,161,71],
    [101, 152, 67],
    [ 99, 142, 63],
    [ 99, 147, 67],
    [ 98, 142, 65],
    [ 96, 130, 60],
    [102, 155, 76],
    [100, 141, 70],
    [ 99, 126, 66],
    [103, 155, 75],
    [104, 166, 80],
    [103, 168, 79],
    [101, 153, 76],
    [101, 159, 80],
    [ 99, 156, 77],
    [101, 155, 80],
    [104, 171, 88],
    [104, 174, 90],])
    yellow= np.array([[23,99,125],
    [27, 112, 125],
    [24, 124, 143],
    [28,  97, 124],
    [31, 104, 125],
    [27, 113, 137],
    [23, 105, 135],
    [23, 109, 145],
    [28, 110, 136],
    [25, 105, 131],
    [25, 102, 126],
    [26,  88, 113],
    [25, 111, 135],
    [24, 105, 129],
    [25,  91, 116],
    [27, 108, 140],
    [24, 107, 135],
    [24,  88, 120],
    ])

    white = np.append(white ,np.zeros(18).reshape(18,1),axis = 1)
    orange= np.append(orange,np.ones(18).reshape(18,1),axis = 1)
    green = np.append(green ,np.full(18,2).reshape(18,1),axis = 1)
    red   = np.append(red   ,np.full(18,3).reshape(18,1),axis = 1)
    blue  = np.append(blue  ,np.full(18,4).reshape(18,1),axis = 1)
    yellow= np.append(yellow,np.full(18,5).reshape(18,1),axis = 1)
    X = np.append(white,orange,axis=0)
    X = np.append(X,green,axis=0)
    X = np.append(X,red,axis=0)
    X = np.append(X,blue,axis=0)
    X = np.append(X,yellow,axis=0)
    return X[:,:3], X[:,3]
class Knn:
    def __init__(self,k):
        self.k = k

    def fit(self,X, Y):
        self.X = X
        self.Y = Y

    def predict(self,X_new):
        dif_X = self.X - X_new
        distance_X = np.linalg.norm(dif_X, axis=1)
        indexer = distance_X.argsort()
        color = np.zeros(6)
        for i in range(self.k):
            color[self.Y[indexer[i]]] += 1
        return color.argmax()
    def color_list(self,src, region):
        color_lst = []
        color_lst2 = []
        for i in range(3):
            for j in range(3):
                color_lst.append(self.predict(get_average(src,region[i][j])))
            for k in range(3,6):
                color_lst2.append(self.predict(get_average(src,region[i][k])))
        color_lst.extend(color_lst2)
        return color_lst
def unicode_to_strlist(lst):
    lst= lst.decode()
    lst = str(lst)# make its type string
    lst = list(lst)# make its type list
    #print(type(lst))
    #print(len(lst))
    color = []
    temp = ''
    for i in  range(len(lst)):
        if lst[i]== " ":
            color.append(temp)
            temp = ''
        else:
            if lst[i]=="'":
                lst[i] = '3'
            temp = temp+lst[i]
    color.append(temp)
    return color
def lst2str(lst):
    one_str = ''
    label = ''
    for i in range(len(lst)):
        if lst[i] == 0:
            label = 'U'
        elif lst[i] == 1:#1 はカメラが読み取るラベルで、knnで出す数字。これは、plotting.pyの教師信号。
            label = 'F'#ラベルFとかは色で、kociembaが使う。数字とラベルを仲介するのが色。
        elif lst[i] == 2:
            label = 'R'
        elif lst[i] == 3:
            label = 'B'
        elif lst[i] == 4:
            label = 'L'
        else:
            label = 'D'
        one_str += label
    return one_str
def str2int(lst):#['D2', 'R3', 'D3', 'F2', 'B', 'D', 'R2', 'D2', 'R3', 'F2', 'D3', 'F2', 'U3', 'B2', 'L2', 'U2', 'D', 'R2', 'U'] -> int
    new_lst = []
    for i in range(len(lst)):
        if lst[i] == 'F':new_lst.append(11)
        elif lst[i] == 'F2':new_lst.append(12)
        elif lst[i] == 'F3':new_lst.append(13)
        elif lst[i] == 'B':new_lst.append(21)
        elif lst[i] == 'B2':new_lst.append(22)
        elif lst[i] == 'B3':new_lst.append(23)
        elif lst[i] == 'L':new_lst.append(31)
        elif lst[i] == 'L2':new_lst.append(32)
        elif lst[i] == 'L3':new_lst.append(33)
        elif lst[i] == 'R':new_lst.append(41)
        elif lst[i] == 'R2':new_lst.append(42)
        elif lst[i] == 'R3':new_lst.append(43)
        elif lst[i] == 'U':new_lst.append(51)
        elif lst[i] == 'U2':new_lst.append(52)
        elif lst[i] == 'U3':new_lst.append(53)
        elif lst[i] == 'D':new_lst.append(61)
        elif lst[i] == 'D2':new_lst.append(62)
        elif lst[i] == 'D3':new_lst.append(63)
        else:pass
    return new_lst

rospy.init_node('camera',anonymous=True)
#これは３回しか送らないので、多分 while いらなそう
#publisher, 回してという司令をおくる
pub1 = rospy.Publisher('plz_rotate', Float64)
pub2 = rospy.Publisher('solution',Num)
time.sleep(1) ##wait for starting Publisher
rotation_num = 0
num = 0
#color_state = []

def plz_rotate():
    t = 1.0 # True
    rospy.loginfo(t)
    pub1.publish(Float64(t))
    #rospy.sleep

#publisher, ルービックの解法のリストを送る
def pub_solution(sol):
    for i in range(1):
    #while not rospy.is_shutdown():
        #rospy.loginfo(sol)
        a = Num()
        a.data = sol
        pub2.publish(a)
        print("sending soluition done")
        rospy.sleep

#subscriber 、アームから、「回したよー」って合図がほしい
def callback(data):
    #global rotation_num
    #rospy.loginfo(rospy.get_name() + "+ :I heard %s" % data.data)
    print(data.data)
    print("Ready!")
    if data.data == 1.0:
        sleep(1)
        #capture =cv2.VideoCapture(1)
        ret, bgr = capture.read()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        t = knn.color_list(hsv,region)
        #print('rotation_num : ', rotation_num)
        #rotation_num += 1
        #print('color_state : ', color_state)
        color_state.extend(t)
        print(color_state)
        #publish
        if (len(color_state)) < 54:
            plz_rotate()

        if len(color_state) == 54:
            #plz_rotate()
            color_array = np.array(color_state)
            idx = [38,41,44,37,40,43,36,39,42,
                    9,10,11,12,13,14,15,16,17,
                    0,1,2,3,4,5,6,7,8,
                    27,28,29,30,31,32,33,34,35,
                    24,21,18,25,22,19,26,23,20,
                    51,48,45,52,49,46,53,50,47]
            color_array =color_array[idx]
            #flip state, for test

            #                                 |
            #                                 |
            #              |        |         |        |        |
            #オレンジ 1、緑 2
            scan1 =[1, 4, 3, 0, 1, 2, 4, 3, 2, 5, 0, 4, 5, 2, 4, 1, 0, 2]
            #青、4 黄色 5
            scan2 =[2, 4, 5, 3, 4, 1, 0, 4, 3, 1, 2, 0, 2, 5, 2, 5, 1, 3]
            #白、0 赤 3
            scan3 =[5, 5, 4, 1, 0, 0, 4, 3, 3, 1, 3, 2, 1, 3, 5, 0, 5, 0]
            scan1.extend(scan2)
            scan1.extend(scan3)
            print(scan1)
            scaned_color = np.array(scan1)
            scaned_color = scaned_color[idx]

            if np.all(color_array == scaned_color):
                print("Automatic scanning succeeded!")
            else:
                print("Automatic scanning failed!")
                color_array = scaned_color
#
            ## flip state
            #color_array = np.array([0,5,0,5,0,5,0,5,0,
            #                        2,4,2,4,2,4,2,4,2,
            #                        1,3,1,3,1,3,1,3,1,
            #                        5,0,5,0,5,0,5,0,5,
            #                        4,2,4,2,4,2,4,2,4,
            #                        3,1,3,1,3,1,3,1,3])
            print(color_array.shape)
            #color state -> 文字配列に
            one_str = lst2str(color_array.tolist())
            print(type(one_str))
            print(one_str)
            #ans = kociemba.solve('DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD').encode()
            temp_one_str = 'DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD'
            print(temp_one_str)
            ans = kociemba.solve(one_str).encode()
            print(ans)
            sol = unicode_to_strlist(ans)
            sol = str2int(sol)
            #solをpublishするだけ
            print("this is solution!",sol)
            #sol = [12,22]#test用　リスト
            pub_solution(sol)
        print("callback done!")
def get_response():
    print(1)
    print("isReady from arms!")
    #rospy.init_node('camera', anonymous=True)
    rospy.Subscriber("isReady",Float64,callback)


def talker():
    pub = rospy.Publisher('chatter', String)
    #while not rospy.is_shutdown():
    for i in range(3):
        str = "hello world %s"%rospy.get_time()
        rospy.loginfo(str)
        pub.publish(String(str))
        rospy.sleep(1.0)


external_point = [(95,126),(220,92),(412, 124),
                    (96,347),(216,391),(415,345)]##輪郭の６つの頂点をいれる。
point = create_points(external_point)
region = create_regions(point)
color_state = []
num = 0
knn = Knn(3)
#X,y = load_data()
X,y = load_data2()
knn.fit(X,y)

if __name__ == '__main__':
    global capture
    capture =cv2.VideoCapture(1)
    #cv2.namedWindow("BGR")
    #cv2.namedWindow("HSV")
    #knn = Knn(3)
    #X,y = load_data()
    #knn.fit(X,y)
    #signal
    print("main")
    rotation_num = 0
    #color_state = []
    Scanning_Ready = False
    Scanning_Done = False
    counter = 0
    print(1)
    get_response()
    #talker()
    #rospy.spin()


    while True:
    #for i in range(10):
        counter += 1
        ret, bgr = capture.read()
        cv2.imshow("original",bgr)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        """
        if Scanning_Ready == False:
            try:
                pass
            except rospy.ROSInterruptException: pass
        if counter > 10:
            Scanning_Ready = True
            try:
                get_response()
            except rospy.ROSInterruptException: pass
        """

        #print(knn.color_list(hsv,region))
        # drawing
        filling_poly(bgr)
        draw_edge(bgr)
        # averaging filter
        for i in range(3):
            for j in range(6):
                get_average(bgr,region[i][j])

        cv2.imshow("BGR",bgr)

        #get_sample_left(hsv,region);get_sample_right(hsv,region);


        c = cv2.waitKey(5)
        if c == 27:
            break
            cap.release()
            cv2.deleteAllwindows()

            lst = np


    cur_lst = kociemba.solve('DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD').encode()
    color = unicode_to_strlist(cur_lst)
    print(type(color))
