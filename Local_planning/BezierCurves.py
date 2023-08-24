"""
局部路径规划
贝塞尔曲线法路径规划实现
@author: HowGreat
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
from celluloid import Camera  # 保存动图时用，pip install celluloid


def instanceForThreeBezier():
    """贝塞尔曲线实例，（四个控制点）3阶贝塞尔曲线"""
    P0 = np.array([0, 0])
    P1 = np.array([1, 1])
    P2 = np.array([2, 1])
    P3 = np.array([3, 0])
    fig = plt.figure(3)
    camera1 = Camera(fig)

    x_2 = []
    y_2 = []
    for t in np.arange(0, 1, 0.01):
        plt.cla()
        plt.plot([P0[0], P1[0]], [P0[1], P1[1]], 'k')
        plt.plot([P1[0], P2[0]], [P1[1], P2[1]], 'k')
        plt.plot([P2[0], P3[0]], [P2[1], P3[1]], 'k')

        # 1阶
        p11_t = (1 - t) * P0 + t * P1
        p12_t = (1 - t) * P1 + t * P2
        p13_t = (1 - t) * P2 + t * P3
        # 1阶 推 2阶
        p21_t = (1 - t) * p11_t + t * p12_t
        p22_t = (1 - t) * p12_t + t * p13_t
        # 2阶 推 3阶
        p3_t = (1 - t) * p21_t + t * p22_t

        x_2.append(p3_t[0])
        y_2.append(p3_t[1])
        plt.scatter(x_2, y_2, c='r')

        plt.plot([p11_t[0], p12_t[0]], [p11_t[1], p12_t[1]], 'b')
        plt.plot([p12_t[0], p13_t[0]], [p12_t[1], p13_t[1]], 'b')

        plt.plot([p21_t[0], p22_t[0]], [p21_t[1], p22_t[1]], 'r')
        plt.title("t=" + str(t))
        plt.pause(0.001)


    #     camera.snap()
    # animation = camera1.animate()
    # animation.save('3阶贝塞尔.gif')


## 递归的方式实现贝塞尔曲线
def bezier(Ps, n, t):
    """递归的方式实现贝塞尔曲线

    Args:
        Ps (_type_): 控制点，格式为numpy数组：array([[x1,y1],[x2,y2],...,[xn,yn]])
        n (_type_): n个控制点，即Ps的第一维度
        t (_type_): 步长t

    Returns:
        _type_: 当前t时刻的贝塞尔点
    """
    if n==1:
        return Ps[0]
    return (1-t)*bezier(Ps[0:n-1],n-1,t)+t*bezier(Ps[1:n],n-1,t)


if __name__=='__main__':
    d = 3.5  # 道路标准宽度

	# 控制点
    Ps = np.array([
        [0, -d / 2],
        [25, -d / 2],
        [25, d / 2],
        [50, d / 2]
        ])

    n = len(Ps) - 1  # 贝塞尔曲线的阶数

    path=[]  # 路径点存储
	# 贝塞尔曲线生成
    for t in np.arange(0,1.01,0.01):
        p_t = bezier(Ps,len(Ps),t)
        path.append(p_t)
    path = np.array(path)


    ## 画图
    fig = plt.figure(1)
    # plt.ylim(-4, 4)
    camera = Camera(fig)
    len_line = 50  # 车道长
    # 画灰色路面图
    GreyZone = np.array([[- 5, - d - 0.5], [- 5, d + 0.5],
                        [len_line, d + 0.5], [len_line, - d - 0.5]])
    for i in range(len(path)):
        # plt.cla()

        plt.fill(GreyZone[:, 0], GreyZone[:, 1], 'gray')
        # 画分界线
        plt.plot(np.array([- 5, len_line]), np.array([0, 0]), 'w--')

        plt.plot(np.array([- 5, len_line]), np.array([d, d]), 'w')

        plt.plot(np.array([- 5, len_line]), np.array([- d, - d]), 'w')

        plt.plot(Ps[:, 0], Ps[:, 1], 'ro') # 画控制点
        plt.plot(Ps[:, 0], Ps[:, 1], 'y') # 画控制点连线
        # 设置坐标轴显示范围
        # plt.axis('equal')
        plt.gca().set_aspect('equal')
        # 绘制路径

        plt.plot(path[0:i, 0], path[0:i, 1], 'g')  # 路径点
        plt.pause(0.001)
        plt.show()
    #     camera.snap()
    # animation = camera.animate()
    # animation.save('trajectory.gif')

