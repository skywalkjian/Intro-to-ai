import numpy as np
from typing import List
from utils import TreeNode
from simuScene import PlanningMap


### 定义一些你需要的变量和函数 ###
STEP_DISTANCE = 3

TARGET_THRESHOLD = 0.23
WILL_EAT = 0.10
ONE_TARGET_MOVE_NUM = 30


def edge(walls):
    """得到墙的边界"""
    x_min = np.min(walls, axis=0)[0]
    y_min = np.min(walls, axis=0)[1]
    x_max = np.max(walls, axis=0)[0]
    y_max = np.max(walls, axis=0)[1]
    return x_min, x_max, y_min, y_max


def distance(current_position, target):
    """计算两点之间的距离"""
    diff = current_position - target
    return np.sqrt(diff @ diff.T)


### 定义一些你需要的变量和函数 ###


class RRT:
    def __init__(self, walls) -> None:
        """
        输入包括地图信息，你需要按顺序吃掉的一列事物位置 
        注意：只有按顺序吃掉上一个食物之后才能吃下一个食物，在吃掉上一个食物之前Pacman经过之后的食物也不会被吃掉
        """
        self.map = PlanningMap(walls)
        self.walls = walls
        
        # 其他需要的变量
        ### 你的代码 ###      
        self.current_target_index = 1
        self.current_target_move_count = 0
        self.next_food_position = np.array([0.0, 0.0])
        self.last_position = None
        ### 你的代码 ###
        
        # 如有必要，此行可删除
        self.path = None
        
        
    def find_path(self, current_position, next_food):
        """
        在程序初始化时，以及每当 pacman 吃到一个食物时，主程序会调用此函数
        current_position: pacman 当前的仿真位置
        next_food: 下一个食物的位置
        
        本函数的默认实现是调用 build_tree，并记录生成的 path 信息。你可以在此函数增加其他需要的功能
        """
        ### 你的代码 ### 
        self.current_target_index = 1
        self.current_target_move_count = 0
        ### 你的代码 ###
        
        # 如有必要，此行可删除
        self.path = self.build_tree(current_position, next_food)
        
        
    def get_target(self, current_position, current_velocity):
        """
        主程序将在每个仿真步内调用此函数，并用返回的位置计算 PD 控制力来驱动 pacman 移动
        current_position: pacman 当前的仿真位置
        current_velocity: pacman 当前的仿真速度
        一种可能的实现策略是，仅作为参考：
        （1）记录该函数的调用次数
        （2）假设当前 path 中每个节点需要作为目标 n 次
        （3）记录目前已经执行到当前 path 的第几个节点，以及它的执行次数，如果超过 n，则将目标改为下一节点
        
        你也可以考虑根据当前位置与 path 中节点位置的距离来决定如何选择 target
        
        同时需要注意，仿真中的 pacman 并不能准确到达 path 中的节点。你可能需要考虑在什么情况下重新规划 path
        """
        ### 你的代码 ###
        final_goal = self.path[-1]
        
        if not self.map.checkline(current_position.tolist(), final_goal.tolist())[0]:
            return final_goal
        if distance(current_position, final_goal) < 0.1:
            return final_goal
        
        current_target = self.path[self.current_target_index]
        if self.current_target_move_count < ONE_TARGET_MOVE_NUM:
            self.current_target_move_count += 1
            return current_target - 0.1 * current_velocity
        
        velocity_small = abs(current_velocity[0]) < 1e-3 and abs(current_velocity[1]) < 1e-3
        has_wall = self.map.checkline(current_position.tolist(), current_target.tolist())[0]
        if velocity_small and has_wall:
            self.find_path(current_position, final_goal)
            current_target = self.path[self.current_target_index]
            self.current_target_move_count += 1
            return current_target - 0.1 * current_velocity

        self.current_target_index += 1
        self.current_target_move_count = 0
        if self.current_target_index >= len(self.path):
            self.find_path(current_position, final_goal)
            current_target = self.path[self.current_target_index]
            self.current_target_move_count += 1
            return current_target - 0.1 * current_velocity
            
        current_target = self.path[self.current_target_index]
        return current_target - 0.1 * current_velocity
        ### 你的代码 ###
        
    ### 以下是RRT中一些可能用到的函数框架，全部可以修改，当然你也可以自己实现 ###
    def build_tree(self, start, goal):
        """
        实现你的快速探索搜索树，输入为当前目标食物的编号，规划从 start 位置食物到 goal 位置的路径
        返回一个包含坐标的列表，为这条路径上的pd targets
        你可以调用find_nearest_point和connect_a_to_b两个函数
        另外self.map的checkoccupy和checkline也可能会需要，可以参考simuScene.py中的PlanningMap类查看它们的用法
        """
        graph = [TreeNode(-1, start[0], start[1])]
        
        ### 你的代码 ###
        current_node = graph[0]
        while distance(current_node.pos, goal) > TARGET_THRESHOLD:
            bounds = edge(self.walls)
            if np.random.randint(0, 100) >= 80:
                random_point = goal
            else:
                x = np.random.uniform(bounds[0], bounds[1])
                y = np.random.uniform(bounds[2], bounds[3])
                random_point = np.array([x, y])
            
            nearest_index, nearest_distance = self.find_nearest_point(random_point, graph)
            nearest_node = graph[nearest_index]
            
            if nearest_distance > STEP_DISTANCE:
                can_connect, new_position = self.connect_a_to_b(nearest_node.pos, random_point)
                if can_connect:
                    new_node = TreeNode(nearest_index, new_position[0], new_position[1])
                    graph.append(new_node)
                    current_node = new_node
            else:
                # 距离很近，直接连接
                if not self.map.checkline(nearest_node.pos.tolist(), random_point.tolist())[0]:
                    new_node = TreeNode(nearest_index, random_point[0], random_point[1])
                    graph.append(new_node)
                    current_node = new_node
        path = []
        if not np.array_equal(goal, current_node.pos):
            path.append(goal)
        
        node = current_node
        path.append(node.pos)
        while node.parent_idx != -1:
            node = graph[node.parent_idx]
            path.append(node.pos)
        path.reverse()
        if not path:
            return [start, goal]
        
        final_path = [start]
        i = 0
        while not np.array_equal(final_path[-1], goal):
            for j in range(len(path) - 1, i, -1):
                start_point = path[i].tolist()
                end_point = path[j].tolist()
                if not self.map.checkline(start_point, end_point)[0]:
                    final_path.append(path[j])
                    i = j
                    break
        
        return final_path
        ### 你的代码 ###

    @staticmethod
    def find_nearest_point(point, graph):
        """
        找到图中离目标位置最近的节点，返回该节点的编号和到目标位置距离、
        输入：
        point：维度为(2,)的np.array, 目标位置
        graph: List[TreeNode]节点列表
        输出：
        nearest_idx, nearest_distance 离目标位置距离最近节点的编号和距离
        """
        nearest_idx = -1
        nearest_distance = 999999
        
        ### 你的代码 ###
        for i in range(len(graph)):
            dist = distance(point, graph[i].pos)
            if dist < nearest_distance:
                nearest_distance = dist
                nearest_idx = i
        ### 你的代码 ###
        
        return nearest_idx, nearest_distance
    
    def connect_a_to_b(self, point_a, point_b):
        """
        以A点为起点，沿着A到B的方向前进STEP_DISTANCE的距离，并用self.map.checkline函数检查这段路程是否可以通过
        输入：
        point_a, point_b: 维度为(2,)的np.array，A点和B点位置，注意是从A向B方向前进
        输出：
        is_empty: bool，True表示从A出发前进STEP_DISTANCE这段距离上没有障碍物
        newpoint: 从A点出发向B点方向前进STEP_DISTANCE距离后的新位置，如果is_empty为真，之后的代码需要把这个新位置添加到图中
        """
        ### 你的代码 ###
        direction = point_b - point_a
        direction_length = np.sqrt(direction @ direction.T)
        
        if direction_length == 0:
            return False, point_a
        new_point = point_a + STEP_DISTANCE * direction / direction_length
        point_a_list = point_a.tolist()
        new_point_list = new_point.tolist()
        
        has_obstacle = self.map.checkline(point_a_list, new_point_list)[0]
        is_occupied = self.map.checkoccupy(new_point_list)
        
        is_empty = not (has_obstacle or is_occupied)
        ### 你的代码 ###
        
        return is_empty, new_point