"""
轨迹跟踪与控制
PID算法
@author: HowGreat
"""

import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from celluloid import Camera  # 保存动图时用，pip install celluloid


# 位置式
class PID_pos:
    """位置式PID实现
    """
    def __init__(self, target, upper, lower, k=None):
        if k is None: k = [1., 0., 0.]
        self.kp, self.ki, self.kd = k

        self.e = 0  # error
        self.pre_e = 0  # previous error
        self.sum_e = 0  # sum of error

        self.target = target  # target
        self.upper_bound = upper    # upper bound of output
        self.lower_bound = lower    # lower bound of output

    def set_target(self, target):
        self.target = target

    def set_k(self, k):
        self.kp, self.ki, self.kd = k

    def set_bound(self, upper, lower):
        self.upper_bound = upper
        self.lower_bound = lower

    def cal_output(self, obs):   # calculate output
        self.e = self.target - obs

        u = self.e * self.kp + self.sum_e * \
            self.ki + (self.e - self.pre_e) * self.kd
        if u < self.lower_bound:
            u = self.lower_bound
        elif u > self.upper_bound:
            u = self.upper_bound

        self.pre_e = self.e
        self.sum_e += self.e
        # print(self.sum_e)
        return u

    def reset(self):
        # self.kp = 0
        # self.ki = 0
        # self.kd = 0

        self.e = 0
        self.pre_e = 0
        self.sum_e = 0
        # self.target = 0

    def set_sum_e(self, sum_e):
        self.sum_e = sum_e


# 增量式
class PID_inc:
    """增量式实现
    """
    def __init__(self, k, target, upper=1., lower=-1.):
        self.kp, self.ki, self.kd = k
        self.err = 0
        self.err_last = 0
        self.err_ll = 0
        self.target = target
        self.upper = upper
        self.lower = lower
        self.value = 0
        self.inc = 0

    def cal_output(self, state):
        self.err = self.target - state  # e_k
        self.inc = self.kp * (self.err - self.err_last) + self.ki * self.err + self.kd * (
            self.err - 2 * self.err_last + self.err_ll)
        self._update()
        return self.value

    def _update(self):
        self.err_ll = self.err_last # e_{k-2}
        self.err_last = self.err    # e_{k-1}
        self.value = self.value + self.inc
        if self.value > self.upper:
            self.value = self.upper
        elif self.value < self.lower:
            self.value = self.lower

    def set_target(self, target):
        self.target = target

    def set_k(self, k):
        self.kp, self.ki, self.kd = k

    def set_bound(self, upper, lower):
        self.upper_bound = upper
        self.lower_bound = lower


class KinematicModel_3:
  """
  假设控制量为转向角delta_f和加速度a
  """

  def __init__(self, x, y, psi, v, L, dt):
    self.x = x
    self.y = y
    self.psi = psi
    self.v = v
    self.L = L
    # 实现是离散的模型
    self.dt = dt

  def update_state(self, a, delta_f):
    self.x = self.x + self.v * math.cos(self.psi) * self.dt
    self.y = self.y + self.v * math.sin(self.psi) * self.dt
    self.psi = self.psi + self.v / self.L*math.tan(delta_f) * self.dt
    self.v = self.v + a * self.dt

  def get_state(self):
    return self.x, self.y, self.psi, self.v


def cal_target_index(robot_state, refer_path):
    """得到临近的路点

    Args:
        robot_state (_type_): 当前车辆位置
        refer_path (_type_): 参考轨迹（数组）

    Returns:
        _type_: 最近的路点的索引
    """
    dists = []
    for xy in refer_path:
        dis = np.linalg.norm(robot_state-xy)
        dists.append(dis)

    min_index = np.argmin(dists)
    return min_index


if __name__ == '__main__':
    # set reference trajectory
    refer_path = np.zeros((1000, 2))
    refer_path[:,0] = np.linspace(0, 100, 1000) # 直线
    refer_path[:,1] = 2*np.sin(refer_path[:,0]/3.0)#+2.5*np.cos(refer_path[:,0]/2.0) # 生成正弦轨迹
    refer_tree = KDTree(refer_path)  # reference trajectory

    # 假设初始状态为x=0,y=-1,偏航角=0.5rad，前后轴距离2m，速度为2m/s，时间步为0.1秒
    ugv = KinematicModel_3(0, -1, 0.5, 2, 2, 0.1)
    k, c = 0.1, 2
    x_, y_ = [], []
    fig = plt.figure(1)
    # 保存动图用
    camera = Camera(fig)
    # PID = PID_pos(target=0, upper=1., lower=-1., k=[0.3, 0.0, 0.1]) # 位置式
    PID = PID_inc(k=[1.0, 0.0, 0.0], target=0, upper=1., lower=-1.)   # 增量式
    for i in range(550):
        robot_state = np.zeros(2)
        robot_state[0] = ugv.x
        robot_state[1] = ugv.y
        distance, ind = refer_tree.query(robot_state) # 在参考轨迹上查询离robot_state最近的点
        # ind = cal_target_index(robot_state,refer_path)  # 使用简单的一个函数实现查询离robot_state最近的点，耗时比较长

        alpha = math.atan2(refer_path[ind, 1]-robot_state[1], refer_path[ind, 0]-robot_state[0])
        l_d = np.linalg.norm(refer_path[ind]-robot_state)
        # l_d = k*ugv.v+c  # 前视距离
        theta_e = alpha-ugv.psi
        e_y = -l_d*math.sin(theta_e)
        # e_y = -l_d*np.sign(math.sin(theta_e))  # 第二种误差表示
        # e_y = robot_state[1]-refer_path[ind, 1] #第三种误差表示
        # PID.set_target(refer_path[i, 1])
        # print(refer_path[i,1])
        delta_f = PID.cal_output(e_y)
        # print('e_y:{}, alpha:{}'.format(e_y, alpha))
        ugv.update_state(0, delta_f) # 加速度设为0

        x_.append(ugv.x)
        y_.append(ugv.y)

        # 显示动图
        plt.cla()
        plt.plot(refer_path[:, 0], refer_path[:, 1], '-.b', linewidth=1.0)
        plt.plot(x_, y_, "-r", label="trajectory")
        plt.plot(refer_path[ind, 0], refer_path[ind, 1], "go", label="target")
        # plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)
    #     camera.snap()
    # animation = camera.animate()
    # animation.save('trajectory.gif')

    plt.figure(2)
    plt.plot(refer_path[:, 0], refer_path[:, 1], '-.b', linewidth=1.0)
    plt.plot(x_,y_,'r')
    plt.show()
