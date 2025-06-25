from typing import List
import numpy as np
from utils import Particle

### 可以在这里写下一些你需要的变量和函数 ###
k=0.1#calculate_particle_weight中的L2系数



ALPHA = 1.05
SIGMA_P = 0.1
SIGMA_T = 0.01  
### 可以在这里写下一些你需要的变量和函数 ###


def generate_uniform_particles(walls, N):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    N: int, 采样点数量
    输出：
    particles: List[Particle], 返回在空地上均匀采样出的N个采样点的列表，每个点的权重都是1/N
    """
    all_particles: List[Particle] = []
    for _ in range(N):
        all_particles.append(Particle(1.0, 1.0, 1.0, 1./N))
    ### 你的代码 ###
    # 寻找墙壁的最大最小值
    xmin = walls[:, 0].min()
    xmax = walls[:, 0].max()
    ymin = walls[:, 1].min()
    ymax = walls[:, 1].max()
    for i in range(N):
        # 生成随机点
        all_particles[i].position[0] = np.random.uniform(xmin, xmax)
        all_particles[i].position[1] = np.random.uniform(ymin, ymax)
        all_particles[i].theta = np.random.uniform(-np.pi, np.pi)
        test = [int(all_particles[i].position[0]),int(all_particles[i].position[1])]
        if (test in walls):
            i -= 1


    ### 你的代码 ###
    return all_particles


def calculate_particle_weight(estimated, gt):
    """
    输入：
    estimated: np.array, 该采样点的距离传感器数据
    gt: np.array, Pacman实际位置的距离传感器数据
    输出：
    weight, float, 该采样点的权重
    """
    weight = 1.0
    ### 你的代码 ###
    weight=np.exp(-k*(np.abs(estimated-gt)).sum())
    ### 你的代码 ###
    return weight


def resample_particles(walls, particles: List[Particle]):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    particles: List[Particle], 上一次采样得到的粒子，注意是按权重从大到小排列的
    输出：
    particles: List[Particle], 返回重采样后的N个采样点的列表
    """
    resampled_particles: List[Particle] = []
    
    ### 你的代码 ###
    N = len(particles)
    for i in range(N):
        for j in range(int(particles[i].weight*N*ALPHA)):
            if (len(resampled_particles) >= N):
                break
            resampled_particles.append(Particle(particles[i].position[0], particles[i].position[1], particles[i].theta, 1./N))
    if (N > len(resampled_particles)):
        tmp_particles = generate_uniform_particles(walls, N-len(resampled_particles))
        for i in range(len(tmp_particles)):
            resampled_particles.append(Particle(tmp_particles[i].position[0], tmp_particles[i].position[1], tmp_particles[i].theta, 1./N))
    for i in range(len(resampled_particles)):
        resampled_particles[i].position[0] += np.random.normal(0, SIGMA_P)
        resampled_particles[i].position[1] += np.random.normal(0, SIGMA_P)
        resampled_particles[i].theta += np.random.normal(0, SIGMA_T)
        resampled_particles[i].theta = (resampled_particles[i].theta+np.pi) % (np.pi*2)-np.pi     
    ### 你的代码 ###
    return resampled_particles

def apply_state_transition(p: Particle, traveled_distance, dtheta):
    """
    输入：
    p: 采样的粒子
    traveled_distance, dtheta: ground truth的Pacman这一步相对于上一步运动方向改变了dtheta，并移动了traveled_distance的距离
    particle: 按照相同方式进行移动后的粒子
    """
    ### 你的代码 ###
    p.theta += dtheta
    p.theta = (p.theta+np.pi) % (2*np.pi)-np.pi
    dx = traveled_distance*np.cos(p.theta)
    dy = traveled_distance*np.sin(p.theta)
    p.position[0] += dx
    p.position[1] += dy
    ### 你的代码 ###
    return p

def get_estimate_result(particles: List[Particle]):
    """
    输入：
    particles: List[Particle], 全部采样粒子
    输出：
    final_result: Particle, 最终的猜测结果
    """
    final_result = Particle()
    ### 你的代码 ###
    K=1
    particles.sort(key=lambda x: x.weight, reverse=True)
    # 寻找权重最大的粒子
    # final_result = particles[0]
    # 寻找权重最大的前k个粒子，计算平均
    final_result = Particle()
    for i in range(K):
        final_result.position[0] += particles[i].position[0] / K
        final_result.position[1] += particles[i].position[1] / K
        final_result.theta += particles[i].theta/K
    ### 你的代码 ###
    return final_result