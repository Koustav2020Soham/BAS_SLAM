import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math
import random

avoidance_mode = False
obstacle_count = 0
last_slam_point_index = -1

# PID Controller parameters (will be optimized by BAS)
class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0
        
    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output
    
    def reset(self):
        self.previous_error = 0
        self.integral = 0

# Beetle Antenna Search Algorithm
class BeetleAntennaSearch:
    def __init__(self, dim=3, max_iter=50, step_size=0.1):
        self.dim = dim  # 3 parameters: Kp, Ki, Kd
        self.max_iter = max_iter
        self.step_size = step_size
        self.best_position = np.random.uniform(0.1, 2.0, dim)
        self.best_fitness = float('inf')
        
    def fitness_function(self, params):
        """Fitness function based on collision avoidance performance"""
        # This would be called during actual navigation
        # For now, return a placeholder - will be updated during runtime
        return np.random.random()
    
    def optimize(self, fitness_func):
        """Run BAS optimization"""
        beetle_pos = self.best_position.copy()
        
        for iteration in range(self.max_iter):
            # Generate random direction
            direction = np.random.randn(self.dim)
            direction = direction / np.linalg.norm(direction)
            
            # Beetle antenna positions
            antenna_distance = 0.1
            left_antenna = beetle_pos + antenna_distance * direction
            right_antenna = beetle_pos - antenna_distance * direction
            
            # Evaluate fitness at antenna positions
            left_fitness = fitness_func(left_antenna)
            right_fitness = fitness_func(right_antenna)
            
            # Move beetle towards better antenna
            if left_fitness < right_fitness:
                beetle_pos += self.step_size * direction
            else:
                beetle_pos -= self.step_size * direction
            
            # Ensure parameters stay within bounds
            beetle_pos = np.clip(beetle_pos, 0.01, 5.0)
            
            # Update best position
            current_fitness = fitness_func(beetle_pos)
            if current_fitness < self.best_fitness:
                self.best_fitness = current_fitness
                self.best_position = beetle_pos.copy()
            
            # Adaptive step size
            self.step_size *= 0.95
        
        return self.best_position

# Global PID controller and BAS optimizer
pid_controller = PIDController()
bas_optimizer = BeetleAntennaSearch()

slam_points = [  # Improved SLAM checkpoint positions
    np.array([-3.72363, 0.20007]),
    np.array([-3.37363, 2.5707]),
    np.array([-1.67363, 4.00007]),
    np.array([-0.72363, 2.80007]),
    np.array([1.35137, 2.92507]),
    np.array([2.40137, 2.12507]),
    np.array([2.97637, 0.67507]),
    np.array([2.92637, -1.04993]),
    np.array([2.42637, -3.07493]),
    np.array([0.47637, -2.42493]),
    np.array([0.62337, -0.02493]),
    np.array([-0.92363, -0.17493]),
    np.array([-2.69863, -2.19993]),
    np.array([-3.62363, -1.14993])
]

def set_movement(bot_wheels, FB_vel, LR_vel, rot):
    sim.setJointTargetVelocity(bot_wheels[0], -FB_vel - LR_vel - rot)
    sim.setJointTargetVelocity(bot_wheels[1], -FB_vel + LR_vel - rot)
    sim.setJointTargetVelocity(bot_wheels[2], -FB_vel - LR_vel + rot)
    sim.setJointTargetVelocity(bot_wheels[3], -FB_vel + LR_vel + rot)

def line_follower(vision_sensor_handle, bot_wheels):
    if avoidance_mode:
        return
    img, [resX, resY] = sim.getVisionSensorImg(vision_sensor_handle)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
    img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    M = cv2.moments(binary)
    if M['m00'] > 0:
        cx = int(M['m10'] / M['m00'])
        cv2.circle(img, (cx, resY//2), 5, (0, 0, 255), -1)
        error = cx - resX // 2
        Kp = 0.05
        base_speed = 8
        set_movement(bot_wheels, base_speed / (1 + 0.05 * abs(error)), 0, Kp * error)
    else:
        set_movement(bot_wheels, 0, 0, 0)

    cv2.imshow('Vision Sensor', img)

def detect_imminent_collision(points, threshold=0.7):
    points = np.array(points, dtype=np.float32).reshape(-1, 3)
    for p in points:
        x, y = p[:2]
        distance = math.hypot(x, y)
        angle = math.atan2(y, x)
        if abs(angle) < np.pi / 4 and distance < threshold:
            return True
    return False

def get_closest_obstacle_distance(points):
    """Get distance to closest obstacle in front"""
    points = np.array(points, dtype=np.float32).reshape(-1, 3)
    min_distance = float('inf')
    
    for p in points:
        x, y = p[:2]
        distance = math.hypot(x, y)
        angle = math.atan2(y, x)
        if abs(angle) < np.pi / 3:  # Front sector
            min_distance = min(min_distance, distance)
    
    return min_distance if min_distance != float('inf') else 2.0

def pid_obstacle_avoidance(points, desired_distance=1.0):
    """Use PID controller for obstacle avoidance"""
    global pid_controller
    
    current_distance = get_closest_obstacle_distance(points)
    error = desired_distance - current_distance
    
    # PID control
    dt = 0.1  # Time step
    control_output = pid_controller.update(error, dt)
    
    # Convert control output to movement
    if error > 0:  # Too close to obstacle
        # Move away from obstacle
        avoidance_speed = min(abs(control_output), 5.0)
        set_movement(bot_wheels, -avoidance_speed, 0, control_output * 0.5)
    else:  # Safe distance
        # Normal movement
        set_movement(bot_wheels, 5, 0, control_output * 0.1)
    
    return abs(error)  # Return error for fitness evaluation

def execute_short_avoidance():
    duration = 1.0
    t_start = time.time()
    while time.time() - t_start < duration:
        set_movement(bot_wheels, -2, 0, 2)
        time.sleep(0.05)
    set_movement(bot_wheels, 0, 0, 0)

def world_to_grid(x, y):
    j = int((origin_y - y) / cell_size)
    i = int((x + origin_x) / cell_size)
    return i, j

def is_path_clear(start_pos, end_pos, grid, safety_margin=2):
    """Check if path between two points is clear in occupancy grid"""
    start_i, start_j = world_to_grid(start_pos[0], start_pos[1])
    end_i, end_j = world_to_grid(end_pos[0], end_pos[1])
    
    # Get line cells between start and end
    line_cells = get_line((start_i, start_j), (end_i, end_j))
    
    for cell in line_cells:
        i, j = cell
        if not (0 <= i < cols and 0 <= j < rows):
            return False
        
        # Check cell and surrounding area for obstacles
        for di in range(-safety_margin, safety_margin + 1):
            for dj in range(-safety_margin, safety_margin + 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < cols and 0 <= nj < rows:
                    if grid[nj, ni] == 2:  # Occupied cell
                        return False
    
    return True

def update_grid(robot_pose, lidar_points, occupied_radius_cells=1):
    x_robot, y_robot, theta = robot_pose
    robot_i, robot_j = world_to_grid(x_robot, y_robot)
    theta -= np.pi / 2
    
    for x_rel, y_rel in lidar_points:
        x_world = x_robot + x_rel * np.cos(theta) - y_rel * np.sin(theta)
        y_world = y_robot + x_rel * np.sin(theta) + y_rel * np.cos(theta)
        end_i, end_j = world_to_grid(x_world, y_world)
        
        # Ray casting for free space
        line_cells = get_line((robot_i, robot_j), (end_i, end_j))
        for cell in line_cells[:-1]:
            if 0 <= cell[1] < rows and 0 <= cell[0] < cols:
                grid[cell[1], cell[0]] = 1  # Free space
        
        # Mark occupied cells with radius
        for di in range(-occupied_radius_cells, occupied_radius_cells + 1):
            for dj in range(-occupied_radius_cells, occupied_radius_cells + 1):
                ni, nj = end_i + di, end_j + dj
                if di**2 + dj**2 <= occupied_radius_cells**2:
                    if 0 <= ni < cols and 0 <= nj < rows:
                        grid[nj, ni] = 2  # Occupied

def get_line(start, end):
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    is_steep = abs(dy) > abs(dx)
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1
    dy = y2 - y1
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
    if swapped:
        points.reverse()
    return points

def show_grid(grid):
    img = np.full(grid.shape, 128, dtype=np.uint8)
    img[grid == 1] = 255
    img[grid == 2] = 0
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    i, j = world_to_grid(target[0], target[1])
    cv2.circle(img_color, (i, j), radius=3, color=(0, 0, 255), thickness=-1)
    cv2.imshow('Occupancy Grid', img_color)
    cv2.waitKey(1)

def world_to_bot_frame(global_point, bot_position, bot_theta):
    dx = global_point[0] - bot_position[0]
    dy = global_point[1] - bot_position[1]
    x_local = np.cos(-bot_theta) * dx - np.sin(-bot_theta) * dy
    y_local = np.sin(-bot_theta) * dx + np.cos(-bot_theta) * dy
    return np.array([x_local, y_local])

def angle_between_vectors(v1, v2):
    """Calculate angle between two vectors"""
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norms == 0:
        return 0
    cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
    return np.arccos(cos_angle)

def get_forward_slam_points(robot_pose, cone_angle=150):
    """Get SLAM points within a forward cone of ±150 degrees"""
    current_pos = np.array(robot_pose[:2])
    current_theta = robot_pose[2]
    
    # Forward direction vector
    forward_vec = np.array([np.cos(current_theta), np.sin(current_theta)])
    
    forward_points = []
    max_cone_angle = np.radians(cone_angle)
    
    for i, point in enumerate(slam_points):
        # Vector from robot to SLAM point
        to_point = point - current_pos
        
        # Skip if point is too close or at same position
        if np.linalg.norm(to_point) < 0.1:
            continue
            
        # Calculate angle between forward direction and point direction
        angle = angle_between_vectors(forward_vec, to_point)
        
        # Check if point is within the cone and path is clear
        if angle <= max_cone_angle and is_path_clear(current_pos, point, grid):
            forward_points.append((i, point, np.linalg.norm(to_point)))
    
    # Sort by distance (closest first)
    forward_points.sort(key=lambda x: x[2])
    return forward_points

def nav_to_slam_point_safe(target_point, target_index, lidar_points):
    """Navigate to SLAM point with PID-based obstacle avoidance"""
    global avoidance_mode, last_slam_point_index, pid_controller
    avoidance_mode = True
    pid_controller.reset()
    
    print(f"Navigating safely to SLAM point {target_index}: {target_point}")
    
    while True:
        # Get current robot pose
        robo_pose = sim.getFloatArrayProperty(sim.handle_scene, "signal.robo_pose")
        local = world_to_bot_frame(target_point, robo_pose[:2], robo_pose[2])
        dist = np.linalg.norm(local)
        
        # Check if reached target
        if dist < 0.3:
            print(f"Reached SLAM point {target_index} - Restarting line following immediately")
            last_slam_point_index = target_index
            break
        
        # Get fresh lidar data
        myData = sim.getBufferProperty(sim.handle_scene, "customData.lidar_points", {'noError': True})
        current_points = sim.unpackTable(myData)
        
        # Use PID controller for obstacle avoidance during navigation
        obstacle_distance = get_closest_obstacle_distance(current_points)
        
        if obstacle_distance < 0.8:  # Too close to obstacle
            # Use PID for obstacle avoidance
            pid_obstacle_avoidance(current_points, desired_distance=1.0)
        else:
            # Normal navigation towards target
            curvature = 2 * local[1] / (local[0] ** 2 + local[1] ** 2) if local[0] != 0 else 0
            angular_velocity = 10 * curvature
            set_movement(bot_wheels, 8, 0, -angular_velocity * 2)
        
        time.sleep(0.1)
    
    # Immediately restart line following
    avoidance_mode = False
    print("Line following resumed with highest priority")

def optimize_pid_parameters():
    """Optimize PID parameters using Beetle Antenna Search"""
    global pid_controller
    
    def fitness_function(params):
        # Test PID parameters
        pid_controller.kp, pid_controller.ki, pid_controller.kd = params
        
        # Simulate obstacle avoidance performance
        # This is a simplified fitness - in practice, you'd run actual navigation
        stability_score = 1.0 / (1.0 + params[0])  # Penalize high Kp
        response_score = params[0] + params[1] * 0.1  # Reward good response
        damping_score = params[2] * 0.5  # Reward some damping
        
        return abs(stability_score - response_score - damping_score)
    
    # Optimize parameters
    optimal_params = bas_optimizer.optimize(fitness_function)
    pid_controller.kp, pid_controller.ki, pid_controller.kd = optimal_params
    
    print(f"Optimized PID parameters: Kp={optimal_params[0]:.3f}, Ki={optimal_params[1]:.3f}, Kd={optimal_params[2]:.3f}")

def handle_obstacle_avoidance(lidar_points):
    """Handle obstacle avoidance with smart SLAM point selection"""
    global obstacle_count
    
    obstacle_count += 1
    print(f"Obstacle detected (count: {obstacle_count}) — rerouting to SLAM point")
    
    # Stop the robot
    set_movement(bot_wheels, 0, 0, 0)
    
    # Execute short avoidance maneuver
    execute_short_avoidance()
    
    # Get robot pose
    robo_pose = sim.getFloatArrayProperty(sim.handle_scene, "signal.robo_pose")
    
    # Get forward SLAM points within the cone
    forward_points = get_forward_slam_points(robo_pose)
    
    if not forward_points:
        print("No forward SLAM points found, using nearest clear point")
        current_pos = np.array(robo_pose[:2])
        # Find nearest point with clear path
        best_idx = None
        best_distance = float('inf')
        
        for i, point in enumerate(slam_points):
            distance = np.linalg.norm(current_pos - point)
            if distance < best_distance and is_path_clear(current_pos, point, grid):
                best_distance = distance
                best_idx = i
        
        if best_idx is not None:
            nav_to_slam_point_safe(slam_points[best_idx], best_idx, lidar_points)
        return
    
    # Select target point based on obstacle count
    if obstacle_count <= 1:
        # First obstacle: go to nearest forward point
        target_idx, target_point, _ = forward_points[0]
        nav_to_slam_point_safe(target_point, target_idx, lidar_points)
    else:
        # After 1 obstacle: try next forward point in the cone
        if len(forward_points) > 1:
            target_idx, target_point, _ = forward_points[1]
            nav_to_slam_point_safe(target_point, target_idx, lidar_points)
        else:
            target_idx, target_point, _ = forward_points[0]
            nav_to_slam_point_safe(target_point, target_idx, lidar_points)
        
        # Reset obstacle count after trying next point
        obstacle_count = 0

# Initialization
k = 1
grid_width = 20 * k
grid_height = 20 * k
cell_size = 0.05 * k
cols = int(grid_width / cell_size)
rows = int(grid_height / cell_size)
grid = np.zeros((rows, cols), dtype=np.uint8)
origin_x = grid_width // 2
origin_y = grid_height // 2

client = RemoteAPIClient()
sim = client.require('sim')
vision_sensor_handle = sim.getObject('/youBot/visionSensor')
bot_wheels = [
    sim.getObject('/youBot/rollingJoint_fl'),
    sim.getObject('/youBot/rollingJoint_rl'),
    sim.getObject('/youBot/rollingJoint_rr'),
    sim.getObject('/youBot/rollingJoint_fr')
]
target = np.array([0, 0])

# Optimize PID parameters at startup
print("Optimizing PID parameters using Beetle Antenna Search...")
optimize_pid_parameters()

sim.startSimulation()
time.sleep(0.5)

# Main loop
try:
    while sim.getSimulationState() != sim.simulation_stopped:
        myData = sim.getBufferProperty(sim.handle_scene, "customData.lidar_points", {'noError': True})
        points = sim.unpackTable(myData)
        points = np.array(points, dtype=np.float32).reshape(-1, 3)
        robo_pose = sim.getFloatArrayProperty(sim.handle_scene, "signal.robo_pose")
        
        # Update occupancy grid
        update_grid(robo_pose, points[:, :2])
        show_grid(grid)

        if detect_imminent_collision(points):
            handle_obstacle_avoidance(points)
            continue

        line_follower(vision_sensor_handle, bot_wheels)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    set_movement(bot_wheels, 0, 0, 0)
    time.sleep(0.5)
    sim.stopSimulation()
    cv2.destroyAllWindows()