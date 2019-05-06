import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  #3Dplo


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
    [6,179,147],
[6,184,154],
[6,178,142],
[6,184,150],
[6,184,151],
[6,178,139],
[6,186,157],
[6,182,152],
[6,175,140],])
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

    white = np.append(white,np.zeros(white.shape[0]).reshape(-1,1),axis = 1)
    orange= np.append(orange,np.ones(orange.shape[0]).reshape(-1,1),axis = 1)
    green = np.append(green ,np.full(green.shape[0],2).reshape(-1,1),axis = 1)
    red   = np.append(red   ,np.full(red.shape[0],3).reshape(-1,1),axis = 1)
    blue  = np.append(blue  ,np.full(blue.shape[0],4).reshape(-1,1),axis = 1)
    yellow= np.append(yellow,np.full(yellow.shape[0],5).reshape(-1,1),axis = 1)
    X = np.append(white,orange,axis=0)
    X = np.append(X,green,axis=0)
    X = np.append(X,red,axis=0)
    X = np.append(X,blue,axis=0)
    X = np.append(X,yellow,axis=0)
    return X[:,:3], X[:,3]
def load_data2():
    #data2

    orange = np.array([
[6,215,165],
[8,219,192],
[9,218,205],
[6,220,157],
[8,222,184],
[10,218,201],
[0,0,0],
[8,218,184],
[9,215,190],
    ])
    green = np.array([
[68,176,107],
[69,178,136],
[69,180,141],
[68,174,92],
[68,183,124],
[69,186,136],
[69,175,84],
[69,183,122],
[68,176,127],
    ])
    yellow= np.array([
[24,144,221],
[25,150,215],
[26,163,187],
[25,151,207],
[26,155,201],
[26,163,179],
[29,142,193],
[25,153,199],
[25,166,177],

    ])
    blue= np.array([
[110,222,157],
[110,234,158],
[111,234,144],
[111,238,148],
[111,226,139],
[111,229,133],
[111,213,136],
[110,200,133],
[112,189,122],
    ])
    white= np.array([
[97,14,212],
[90,10,225],
[74,12,206],
[97,19,211],
[94,29,204],
[79,9,226],
[101,17,208],
[92,11,222],
[76,9,218],
])
    red= np.array([
[13,174,128],
[12,193,126],
[18,190,132],
[15,188,120],
[18,189,104],
[23,174,146],
[23,152,97],
[21,173,114],
[25,172,155],

    ])


    white = np.append(white,np.zeros(white.shape[0]).reshape(-1,1),axis = 1)
    orange= np.append(orange,np.ones(orange.shape[0]).reshape(-1,1),axis = 1)
    green = np.append(green ,np.full(green.shape[0],2).reshape(-1,1),axis = 1)
    red   = np.append(red   ,np.full(red.shape[0],3).reshape(-1,1),axis = 1)
    blue  = np.append(blue  ,np.full(blue.shape[0],4).reshape(-1,1),axis = 1)
    yellow= np.append(yellow,np.full(yellow.shape[0],5).reshape(-1,1),axis = 1)
    X = np.append(white,orange,axis=0)
    X = np.append(X,green,axis=0)
    X = np.append(X,red,axis=0)
    X = np.append(X,blue,axis=0)
    X = np.append(X,yellow,axis=0)
    #return X[:,:3], X[:,3],white,orange,green,red,blue,yellow
    return X[:,:3], X[:,3]

if __name__ == '__main__':
    #HSV graph
    X,y = load_data2()
    white = np.array([X[i] for i in range(X.shape[0]) if y[i] == 0])
    orange = np.array([X[i] for i in range(X.shape[0]) if y[i] == 1])
    green = np.array([X[i] for i in range(X.shape[0]) if y[i] == 2])
    blue = np.array([X[i] for i in range(X.shape[0]) if y[i] == 3])
    red = np.array([X[i] for i in range(X.shape[0]) if y[i] == 4])
    yellow = np.array([X[i] for i in range(X.shape[0]) if y[i] == 5])
    #black = np.array([X[i] for i in range(X.shape[0]) if y[i] == 6])
    fig1=plt.figure(figsize=(10,10))
    ax1=Axes3D(fig1)
    ax1.scatter3D(white[:,0],white[:,1],white[:,2],color = 'gray')
    ax1.scatter3D(orange[:,0],orange[:,1],orange[:,2],color = 'orange')
    ax1.scatter3D(green[:,0],green[:,1],green[:,2],color = 'green')
    ax1.scatter3D(red[:,0],red[:,1],red[:,2],color = 'red')
    ax1.scatter3D(yellow[:,0],yellow[:,1],yellow[:,2],color = 'yellow')
    ax1.scatter3D(blue[:,0],blue[:,1],blue[:,2],color = 'blue')
    #ax1.scatter3D(black[:,0],black[:,1],black[:,2],color = 'black')
    #ax1.scatter3D(test[:,0],test[:,1],test[:,2],color = 'gray')
    ax1.set_xlabel("huew")
    ax1.set_ylabel("Saturate")
    ax1.set_zlabel("value")
    """
    #BGR graph
    fig2=plt.figure(figsize=(10,10))
    ax2=Axes3D(fig2)
    ax2.scatter3D(white[:,3],white[:,4],white[:,5],color = 'black')
    ax2.scatter3D(orange[:,3],orange[:,4],orange[:,5],color = 'orange')
    ax2.scatter3D(green[:,3],green[:,4],green[:,5],color = 'green')
    ax2.scatter3D(red[:,3],red[:,4],red[:,5],color = 'red')
    ax2.scatter3D(yellow[:,3],yellow[:,4],yellow[:,5],color = 'yellow')
    ax2.scatter3D(blue[:,3],blue[:,4],blue[:,5],color = 'blue')
    #ax2.scatter3D(test[:,3],test[:,4],test[:,5],color = 'gray')
    ax2.set_xlabel("Hue")
    ax2.set_ylabel("Saturate")
    ax2.set_zlabel("value")
    """
    plt.legend()
    plt.show()
