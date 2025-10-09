#!/usr/bin/env python3
import heapq
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Set, List, Iterable, Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
import sys

Coord = Tuple[float, float]
Index = Tuple[int, int]
DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

# ---------------- Grid graph ----------------
@dataclass
class GridGraph:
    """
    Lightweight grid graph whose nodes are (row, col) indices mapped
    to metric (x, y) world coordinates. Edges connect 4-neighborhood cells.
    """
    H: int
    W: int
    positions: Dict[Index, Coord]   # (r,c) -> (x,y)
    coord2idx: Dict[Coord, Index]   # (x,y) -> (r,c)
    adj: Dict[Index, Set[Index]]    # adjacency

    @classmethod
    def from_node_graph(cls, node_graph: List[List[List[float]]]):
        """
        Build a GridGraph from a 2D list of XY coordinates.

        Args:
            node_graph: HxW list; node_graph[r][c] is [x, y] for grid cell (r, c).

        Returns:
            GridGraph: A fully connected 4-neighborhood graph with
            (r,c)<->(x,y) mappings initialized.
        """
        H, W = len(node_graph), len(node_graph[0])
        positions: Dict[Index, Coord] = {}
        coord2idx: Dict[Coord, Index] = {}
        for r in range(H):
            for c in range(W):
                xy = tuple(float(v) for v in node_graph[r][c])
                positions[(r, c)] = xy
                coord2idx[xy] = (r, c)
        adj: Dict[Index, Set[Index]] = {(r, c): set() for r in range(H) for c in range(W)}
        for r in range(H):
            for c in range(W):
                for dr, dc in DIRS.values():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        adj[(r, c)].add((nr, nc))
                        adj[(nr, nc)].add((r, c))
        return cls(H, W, positions, coord2idx, adj)

    def index_of(self, coord: Coord, eps: float = 1e-9) -> Index:
        """
        Look up the grid index for a given (x, y) coordinate.

        Args:
            coord: (x, y) world coordinate.
            eps: absolute tolerance for fallback floating-point comparison.

        Returns:
            (r, c) integer grid index.

        Raises:
            KeyError: if the coordinate is not found (within tolerance).
        """
        x, y = float(coord[0]), float(coord[1])
        key = (x, y)
        if key in self.coord2idx:
            return self.coord2idx[key]
        for idx, (px, py) in self.positions.items():
            if math.isclose(px, x, abs_tol=eps) and math.isclose(py, y, abs_tol=eps):
                return idx
        raise KeyError(f"Coordinate {coord} not found.")

    def coord_of(self, idx: Index) -> Coord:
        """
        Get the (x, y) coordinate for a given grid index.

        Args:
            idx: (r, c) integer grid index.

        Returns:
            (x, y) world coordinate.
        """
        return self.positions[idx]

    def has_edge(self, a: Coord, b: Coord) -> bool:
        """
        Check whether the edge between two coordinates exists.

        Args:
            a: (x, y) of first node.
            b: (x, y) of second node.

        Returns:
            True if a<->b exists, else False.
        """
        ia, ib = self.index_of(a), self.index_of(b)
        return (ib in self.adj[ia]) and (ia in self.adj[ib])

    def remove_edge(self, a: Coord, b: Coord):
        """
        Remove the edge between two coordinates if the edge exists

        Args:
            a: (x, y) of first node.
            b: (x, y) of second node.
        """
        ia, ib = self.index_of(a), self.index_of(b)
        self.adj[ia].discard(ib)
        self.adj[ib].discard(ia)

    def edges(self) -> Iterable[Tuple[Coord, Coord]]:
        """
        Iterate all unique edges in coordinate space.

        Yields:
            Tuples (coord_a, coord_b) for each edge once.
        """
        seen = set()
        for a_idx, nbrs in self.adj.items():
            for b_idx in nbrs:
                key = tuple(sorted([a_idx, b_idx]))
                if key not in seen:
                    seen.add(key)
                    yield (self.coord_of(a_idx), self.coord_of(b_idx))

    def _compute_normalizer(self, extra_points: Optional[List[Coord]] = None,
                            img_size=700, margin=50):
        """
        Prepare a coordinate->pixel mapping for visualization.

        Args:
            extra_points: optional extra (x, y) points to include in bounds.
            img_size: output square image size in pixels.
            margin: padding in pixels.

        Returns:
            (to_px, img_size, margin) where to_px(x, y)->(u, v) pixel function.
        """
        xs = [p[0] for p in self.positions.values()]
        ys = [p[1] for p in self.positions.values()]
        if extra_points:
            xs.extend([p[0] for p in extra_points])
            ys.extend([p[1] for p in extra_points])
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        pad_x = 0.05 * (xmax - xmin if xmax > xmin else 1.0)
        pad_y = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
        xmin -= pad_x; xmax += pad_x
        ymin -= pad_y; ymax += pad_y
        xr = xmax - xmin if xmax > xmin else 1.0
        yr = ymax - ymin if ymax > ymin else 1.0
        S = img_size - 2 * margin
        def to_px(x, y):
            nx = (x - xmin) / xr
            ny = (y - ymin) / yr
            nx = 0.0 if nx < 0.0 else (1.0 if nx > 1.0 else nx)
            ny = 0.0 if ny < 0.0 else (1.0 if ny > 1.0 else ny)
            u = margin + int(nx * S)
            v = margin + int((1.0 - ny) * S)
            return (u, v)
        return to_px, img_size, margin

    def base_image(self, extra_points: Optional[List[Coord]] = None,
                   img_size=700, margin=50) -> Tuple[np.ndarray, callable]:
        """
        Create a blank white image and a to_px mapping for drawing the graph.

        Args:
            extra_points: optional (x, y) points to include in normalization.
            img_size: output image size in pixels.
            margin: border margin in pixels.

        Returns:
            (img, to_px) where img is an (H x H x 3) uint8 canvas,
            and to_px is a function mapping (x, y)->(u, v).
        """
        to_px, H, _ = self._compute_normalizer(extra_points, img_size, margin)
        img = np.ones((H, H, 3), dtype=np.uint8) * 255
        return img, to_px

# -------- path util ----------
def nearest_point_on_polyline(path: List[Coord], p: Coord) -> Tuple[float, int, Coord]:
    """
    Project a point onto a polyline and find the closest point/segment.

    Args:
        path: list of waypoints [(x, y), ...].
        p: query point (x, y).

    Returns:
        (s, seg_idx, proj) where:
          - s is arclength from path[0] to the projection point (meters),
          - seg_idx is the index of the segment [path[i], path[i+1]] containing proj,
          - proj is the projected (x, y) point on the path.
    """
    if len(path) == 1:
        return 0.0, 0, path[0]
    def dot(a, b): return a[0]*b[0] + a[1]*b[1]
    def sub(a, b): return (a[0]-b[0], a[1]-b[1])
    def add(a, b): return (a[0]+b[0], a[1]+b[1])
    cum = [0.0]
    for i in range(1, len(path)):
        cum.append(cum[-1] + math.dist(path[i-1], path[i]))
    best_s = 0.0; best_pt = path[0]; best_seg = 0
    for i in range(1, len(path)):
        a = path[i-1]; b = path[i]
        ab = sub(b, a); ap = sub(p, a)
        ab2 = dot(ab, ab) if ab != (0.0, 0.0) else 1e-9
        t = max(0.0, min(1.0, dot(ap, ab) / ab2))
        proj = add(a, (ab[0]*t, ab[1]*t))
        s_here = cum[i-1] + math.dist(a, proj)
        if i == 1 or math.dist(p, proj) < math.dist(p, best_pt):
            best_s = s_here; best_pt = proj; best_seg = i - 1
    return best_s, best_seg, best_pt

def lookahead_point(path: List[Coord], p: Coord, Ld: float) -> Coord:
    """
    Find the point at arclength along a polyline from the projection of p.

    Args:
        path: list of waypoints [(x, y), ...].
        p: current position (x, y).
        Ld: lookahead distance in meters.

    Returns:
        (x, y) of the lookahead point clamped to the end of the path.
    """
    if not path: return p
    if len(path) == 1: return path[0]
    s, _, _ = nearest_point_on_polyline(path, p)
    cum = [0.0]
    for i in range(1, len(path)):
        cum.append(cum[-1] + math.dist(path[i-1], path[i]))
    target_s = s + Ld
    if target_s >= cum[-1]: return path[-1]
    for i in range(1, len(path)):
        if cum[i] >= target_s:
            seg_s = target_s - cum[i-1]
            a = path[i-1]; b = path[i]
            seg_len = math.dist(a, b)
            t = seg_s / seg_len
            return (a[0] + (b[0]-a[0]) * t, a[1] + (b[1]-a[1]) * t)
    return path[-1]

def angle_wrap(a: float) -> float:
    """
    Wrap an angle to the range [-pi, pi].

    Args:
        a: angle in radians.

    Returns:
        Wrapped angle in radians.
    """
    while a > math.pi: a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a

def yaw_from_quat(x, y, z, w) -> float:
    """
    Extract yaw (heading about +Z) from a quaternion.

    Args:
        x, y, z, w: quaternion components.

    Returns:
        Yaw in radians in the odometry frame
    """
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    return math.atan2(siny_cosp, cosy_cosp)

# --------- Simple PID ----------
@dataclass
class PID:
    """
    Minimal PID controller with anti-windup (integrator clamping).
    Call update(e, dt) each cycle to get the control effort.
    """
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    i: float = 0.0
    prev_e: Optional[float] = None
    i_limit: float = 1.0

    def reset(self):
        """
        Reset internal integrator and derivative memory.
        """
        self.i = 0.0
        self.prev_e = None

    def update(self, e: float, dt: float) -> float:
        """
        Compute PID output for a given error and timestep.

        Args:
            e: control error (setpoint - measurement).
            dt: time step in seconds.

        Returns:
            Control effort (float).
        """
        if dt <= 0.0:
            dterm = 0.0
        else:
            dterm = 0.0 if self.prev_e is None else (e - self.prev_e) / dt
        self.i += e * dt
        self.i = float(np.clip(self.i, -self.i_limit, self.i_limit))
        self.prev_e = e
        return self.kp * e + self.ki * self.i + self.kd * dterm

# --------------- Node ---------------
class GridPathFollowerNode(Node):
    """
    ROS2 node that:
    - Plans on a grid using A* between start/goal nodes,
    - Follows the path with a suitable control mechanism,
    - Backtracks and replans if an obstacle is not passable
    """
    def __init__(self):
        """
        Initialize parameters, build graph, subscribe/publish ROS topics,
        and compute the initial path.
        """
        super().__init__("grid_path_follower")

        # --- Motion and geometry parameters ---
        self.goal_tol = 0.05
        self.node_tol = 0.04
        # TODO: YOUR CODE HERE: ~3 lines: set the lookahead distance for pure-pursuit based control, and the maximum allowed linear and angular velocities 
        self.lookahead_dist = 0.8
        self.v_max = 3.0
        self.w_max = 2.0
        # ...
        # TODO: YOUR CODE HERE: ~1-2 lines: set your PID controller/s for linear/angular motion
        self.pid_angular = PID(kp=0.8, ki=0.1, kd=0.05, i_limit=1.0)

        # --- Helpers to halt turtlebot ---
        self.dwell_s = 0.5
        self.dwell_until: Optional[float] = None

        # --- Start and goal positions. TODO: OPTIONAL: modify for testing purposes as needed ---
        self.start_pos = (0, 0)
        self.goal_pos  = (7.120, -5.696)

        # --- State ---
        self.pose_ready = False                    # to check if odometry values are being received
        self.x = 0.0; self.y = 0.0; self.yaw = 0.0 # turtlebot pose 
        self.scan: Optional[LaserScan] = None      # stores the latest scan
        self.mode = 'None'                         # operation mode 
        self.last_t = self._now()

        # --- Grid (5x5) ---
        self.node_graph = [
            [[0.712, 0], [2.136, 0], [3.560, 0], [4.984, 0], [6.408, 0]],
            [[0.712, -1.424], [2.136, -1.424], [3.560, -1.424], [4.984, -1.424], [6.408, -1.424]],
            [[0.712, -2.848], [2.136, -2.848], [3.560, -2.848], [4.984, -2.848], [6.408, -2.848]],
            [[0.712, -4.272], [2.136, -4.272], [3.560, -4.272], [4.984, -4.272], [6.408, -4.272]],
            [[0.712, -5.696], [2.136, -5.696], [3.560, -5.696], [4.984, -5.696], [6.408, -5.696]],
        ]
        # Graph and initial plan
        self.graph = GridGraph.from_node_graph(self.node_graph)
        """
        GridGraph.from_node_graph returns:
                 GridGraph(
                    H=5, 
                    W=5, 
                    positions={(0, 0): (0.712, 0.0), (0, 1): (2.136, 0.0), ....}
                    coord2idx={(0.712, 0.0): (0, 0), (2.136, 0.0): (0, 1), ....}
                    adj={(0, 0): {(1, 0), (0, 1)}, (0, 1): {(0, 2), (1, 1), (0, 0)}, (0, 2): {(0, 1), (1, 2), (0, 3)}, ........., (4, 4): {(3, 4), (4, 3)}}
                   )
        """
        self.full_path:  List[Coord] = []
        self.prev_node : Optional[Coord] = None
        self.block_edge: Optional[Tuple[Coord, Coord]] = None
        #TODO: OPTIONAL: you can manually remove edges here using 'self.graph.remove_edge([..., ...], [..., ...]))

        # --- ROS I/O ---
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 20)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_cb, 20)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_cb, 20)
        self.timer = self.create_timer(1.0/20.0, self.control_step)

        # --- Modes of operation ---
        """
            1. FOLLOW: when in default mode of following the A* generated path  
            2. BACKTRACK: when an unpassable obstacle is encountered (go back to immediately previous node) 
            3. REPLAN: recalculate A* to get new path to the goal from the current node
            4. DONE: when goal has been reached
        """
        self._set_mode('FOLLOW') 

        # --- Vizualization ---
        self.win = "Grid A* Path Follower"
        cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)

        # --- Initial plan ----
        self.path_plan(self.start_pos)
        self.get_logger().info(f"Planned {len(self.full_path)} waypoints.")

    # --- OPTIONAL helper: time ---
    def _now(self) -> float:
        """
        Get ROS2 clock time in seconds (float).

        Returns:
            Current time in seconds as a float.
        """
        return self.get_clock().now().nanoseconds * 1e-9

    # --- Helper to set the mode of operation ---
    def _set_mode(self, new_mode: str):
        """
        Switch high-level mode and enforce a short dwell (pause).

        Args:
            new_mode: one of {'FOLLOW', 'BACKTRACK', 'REPLAN', 'DONE'}.
        """
        if self.mode != new_mode:
            self.mode = new_mode
            self.publish_stop()
            self.dwell_until = self._now() + self.dwell_s
            self.get_logger().info(f"Mode -> {self.mode}, pausing {self.dwell_s}s")

    # --- planning ---
    # --- A* implementation ---
    def astar_path(self, graph: GridGraph, start: Coord, goal: Coord) -> List[Coord]:
        """
        Run A* on the GridGraph between two coordinates.

        Args:
            graph: GridGraph instance.
            start: (x, y) start coordinate
            goal:  (x, y) goal coordinate 

        Returns:
            List of (x, y) coordinates along the path from start node to goal node,
            inclusive. Empty list if no path.

        Hint: use the relevant GriGraph functions)
        """
        start_idx = graph.index_of(start)
        goal_idx = graph.index_of(goal)
        def h(i: Index) -> float:
            x1, y1 = graph.coord_of(i); x2, y2 = graph.coord_of(goal_idx)
            return math.hypot(x2 - x1, y2 - y1)
        def c(i: Index, j: Index) -> float:
            x1, y1 = graph.coord_of(i); x2, y2 = graph.coord_of(j)
            return math.hypot(x2 - x1, y2 - y1)
        open_heap = [(h(start_idx), start_idx)]
        came: Dict[Index, Index] = {}
        g = {start_idx: 0.0}
        in_open = {start_idx}
        while open_heap:
            _, cur = heapq.heappop(open_heap)
            in_open.discard(cur)
            if cur == goal_idx:
                path_idx = [cur]
                while cur in came:
                    cur = came[cur]
                    path_idx.append(cur)
                path_idx.reverse()
                return [graph.coord_of(i) for i in path_idx]
            for nbr in graph.adj[cur]:
                t = g[cur] + c(cur, nbr)
                if t < g.get(nbr, float('inf')):
                    came[nbr] = cur
                    g[nbr] = t
                    f = t + h(nbr)
                    if nbr not in in_open:
                        heapq.heappush(open_heap, (f, nbr))
                        in_open.add(nbr)
        return []

    def nearest_graph_coord(self, p: Coord) -> Coord:
        """
        Find the nearest graph node (in metric space) to a given point.

        Args:
            p: (x, y) query position.

        Returns:
            (x, y) of the closest node in the GridGraph.
        """
        best = None; best_d = float('inf')
        for xy in self.graph.coord2idx.keys():
            d = math.dist(p, xy)
            if d < best_d: best_d = d; best = xy
        return best

    # ---  Invoke A* to return the nodes that define path ---
    def plan_from(self, start_pose: Coord) -> List[Coord]:
        """
        Build a waypoint list: start_pose -> nearest start node -> A* nodes
        -> nearest goal node -> goal.

        Args:
            start_pose: (x, y) where to start planning from.

        Returns:
            List of (x, y) waypoints including start_pose and goal_pos.
        """
        
        # 1. Get the closest graph node to the robot’s current starting position (ns)
        ns = self.nearest_graph_coord(start_pose)
        
        # 2. Get the closest graph node to the goal position (ng)
        ng = self.nearest_graph_coord(self.goal_pos)
        
        # 3. Run A* to get a path (sequence of graph nodes) from ns → ng
        grid_path = self.astar_path(self.graph, ns, ng)

        path: List[Coord] = [start_pose]

        # 5. If the start node is different from the current robot position, append it to 'path'
        if ns != start_pose:
            path.append(ns)
        
        # 6. Append all waypoints from the A* grid path (avoiding duplicate points)
        for p in grid_path:
            if p != path[-1]:
                path.append(p)
                
        # 7. Ensure the final node (ng) is included
        if ng != path[-1]:
            path.append(ng)
            
        # 8. Ensure the actual goal position is included as the last waypoint
        if self.goal_pos != path[-1]:
            path.append(self.goal_pos)

        # Return the full planned path as a list of coordinates
        return path

    def path_plan(self, start_from: Optional[Coord] = None):
        """
        (Re)plan a global path and store it in self.full_path.

        Args:
            start_from: optional (x, y) to start planning from; if None,
                        uses current odom pose (if available) else start_pos.
        """
        if start_from is None:
            start_from = (self.x, self.y) if self.pose_ready else self.start_pos
        self.full_path = self.plan_from(start_from)
        self.get_logger().info(f"Path planned from {start_from} with {len(self.full_path)} waypoints.")

    # --- callbacks ---
    def odom_cb(self, msg: Odometry):
        """
        Odometry callback. Updates pose estimate (x, y, yaw).

        Args:
            msg: nav_msgs/Odometry message.
        """
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        self.pose_ready = True

    def scan_cb(self, msg: LaserScan):
        """
        LaserScan callback. Stores the latest scan.

        Args:
            msg: sensor_msgs/LaserScan message.
        """
        self.scan = msg

    # --- TODO: YOUR CODE HERE: Implement the function to determine passability  ---
    def is_passable(self, seg_dir_yaw: float) -> bool:
        """
        Decide if the edge between two nodes is passable or not

        Args:
            seg_dir_yaw: segment heading in world/odom frame (radians).

        Returns:
            True if passable; False otherwise.
        """
        if self.scan is None:
            self.get_logger().warn("No LIDAR data yet — assuming path is passable.")
            return True

        # --- Parameters ---
        dist_thresh = 0.4          # distance (m) to consider a hit (obstacle)
        min_corridor = 0.20        # minimum gap width robot can fit through (m)
        robot_width = 0.178         # TurtleBot3 width (m)
        sector_half_angle = math.radians(30)  # angular window (±45° around path direction)

        # --- Extract LIDAR data ---
        angle_min = self.scan.angle_min
        angle_max = self.scan.angle_max
        angle_inc = self.scan.angle_increment
        ranges = np.array(self.scan.ranges)
        n = len(ranges)

        # --- Compute relative yaw ---
        # LIDAR frame is robot-relative; seg_dir_yaw is world-relative.
        rel_yaw = angle_wrap(seg_dir_yaw - self.yaw)
        center_angle = rel_yaw
        left_bound = center_angle - sector_half_angle
        right_bound = center_angle + sector_half_angle

        # Convert angles to indices
        left_idx = int(max(0, (left_bound - angle_min) / angle_inc))
        right_idx = int(min(n - 1, (right_bound - angle_min) / angle_inc))
        if left_idx >= right_idx:
            self.get_logger().warn("Invalid LIDAR sector bounds — assuming safe.")
            return True

        sector_ranges = ranges[left_idx:right_idx]
        sector_angles = np.linspace(left_bound, right_bound, len(sector_ranges))

        # --- Detect obstacle hits ---
        hits = np.where(sector_ranges < dist_thresh)[0]
        if len(hits) == 0:
            self.get_logger().debug("No obstacles detected in sector — passable.")
            return True

        mid_idx = len(sector_ranges) // 2
        left_hits = hits[hits < mid_idx]
        right_hits = hits[hits > mid_idx]

        if len(left_hits) == 0 or len(right_hits) == 0:
            self.get_logger().debug("Open space on one side — passable.")
            return True

        # --- Last hit on the left, first hit on the right ---
        left_hit_idx = left_hits[-1]
        right_hit_idx = right_hits[0]
        d1 = sector_ranges[left_hit_idx]
        d2 = sector_ranges[right_hit_idx]
        d_angle = abs(sector_angles[right_hit_idx] - sector_angles[left_hit_idx])

        # --- Compute corridor width (Law of Cosines) ---
        width = math.sqrt(max(0.0, d1**2 + d2**2 - 2 * d1 * d2 * math.cos(d_angle)))

        # --- Decision ---
        passable = width > max(min_corridor, robot_width)

        # --- Logging ---
        self.get_logger().info(
            f"[is_passable] Sector ±{math.degrees(sector_half_angle):.1f}°, "
            f"width={width:.3f} m, "
            f"d1={d1:.3f}, d2={d2:.3f}, Δθ={math.degrees(d_angle):.1f}°, "
            f"passable={passable}"
        )

        return passable

    
    # --- TODO: YOUR CODE HERE: Implement the function to generate the lookahead point away from obstacles  ---
    # def obstacle_avoided_target(self, robot_pos: Coord, seg_dir_yaw: float) -> Coord:
    #     """
    #     Generate a target point that accounts for obstacle avoidance:
    #     - Pure-pursuit base target at lookahead distance along the path.
    #     - Lateral shift inside the road corridor to increase clearance.

    #     Args:
    #         robot_pos: current turtlebot position (x, y) in odom/world frame.
    #         seg_dir_yaw: segment heading in world/odom frame (radians).

    #     Returns:
    #         Target (x, y) in world/odom frame.
    #     """
    #     # front = msg.ranges[0]      # Front (0°)
    #     # left = msg.ranges[90]      # Left (90°)
    #     # right = msg.ranges[270]    # Right (270°)
    #     # back = msg.ranges[180]     # Back (180°)

    #     #If no scan data is available, just fall back to pure-pursuit behavior
    #     if self.scan is None:
    #         return lookahead_point(self.full_path, robot_pos, self.lookahead_dist)

    #     base_lookahead = self.lookahead_dist
    #     close_lookahead = 0.19  # distanza fissa più vicina in caso di ostacoli
    #     max_shift = 0.6
    #     safety_threshold = 0.2
    #     alpha = 0.6

    #     ranges = self.scan.ranges

    #     left_idx = range(15,75)
    #     right_idx = range(285, 345)
    #     front_idx = list(range(0,360))#list(range(345, 360)) + list(range(0, 15))

    #     def avg_range(idxs):
    #         vals = [ranges[i] for i in idxs if math.isfinite(ranges[i])]
    #         return sum(vals) / len(vals) if vals else float("inf")
    #     def front_score(idxs):
    #         vals = [ranges[i] for i in idxs if math.isfinite(ranges[i])]
    #         return  0.5 * min(vals) + 0.5 * sum(vals)/len(vals) if vals else float("inf")

    #     avg_left = front_score(left_idx)
    #     avg_right = front_score(right_idx)
    #     avg_front = front_score(front_idx)

    #     # --- ADAPT LOOKAHEAD DISTANCE ---
    #     if avg_front < safety_threshold or avg_left < safety_threshold or avg_right < safety_threshold:
    #         target_lookahead = close_lookahead
    #     else:
    #         target_lookahead = base_lookahead

    #     # --- COMPLEMENTARY FILTER (lookahead smoothing) ---
    #     if not hasattr(self, "_smooth_lookahead"):
    #         self._smooth_lookahead = base_lookahead
    #     alpha_dist = 0.4  # più basso = più smorzato
    #     self._smooth_lookahead = (
    #         alpha_dist * target_lookahead + (1 - alpha_dist) * self._smooth_lookahead
    #     )

    #     # --- BASE TARGET CON LOOKAHEAD FILTRATO ---
    #     base_target = lookahead_point(self.full_path, robot_pos, self._smooth_lookahead)


    #     # --- COMPUTE LATERAL SHIFT ---
    #     if avg_front < safety_threshold or avg_left < safety_threshold or avg_right < safety_threshold:
    #         diff = avg_left - avg_right
    #         diff_norm = max(-1.0, min(1.0, diff / (avg_left + avg_right)))
    #         target_shift = diff_norm * max_shift
    #     else:
    #         target_shift = 0.0

    #     # --- SMOOTH LATERAL SHIFT ---
    #     if not hasattr(self, "_smooth_shift"):
    #         self._smooth_shift = 0.0
    #     beta = 0.4
    #     self._smooth_shift = beta * target_shift + (1 - beta) * self._smooth_shift

    #     # --- APPLY SHIFT IN WORLD FRAME ---
    #     perp_yaw = seg_dir_yaw + math.pi / 2.0
    #     target = (
    #         base_target[0] + self._smooth_shift* math.cos(perp_yaw),
    #         base_target[1] + self._smooth_shift * math.sin(perp_yaw)
    #     )
    #     return target





    def obstacle_avoided_target(self, robot_pos: tuple, seg_dir_yaw: float) -> tuple:
        """
        Generate a lateral-only avoidance target:
        - If no close obstacles in front, return standard lookahead
        - Otherwise sample points along the line perpendicular to seg_dir_yaw
        and pick the one that maximizes lateral clearance from front obstacles
        while minimizing lateral displacement.
        """
        radius = 0.1           # max lateral shift (m)
        num_samples = 10       # samples per side (total samples = 2*num_samples + 1)
        safety_dist = 0.3      # consider as "obstacle" only ranges < safety_dist
        lateral_weight = 0.8   # penalty weight for lateral displacement (tune this)
        best_point = None
        best_score = -float('inf')

        # front LIDAR indices (±15 degrees around 0°)
        front_idx = list(range(345, 360)) + list(range(0, 15))
        # keep only indices that actually exist in this scan
        front_idx = [i for i in front_idx if i < len(self.scan.ranges)]

        # quick check: any close obstacle in front?
        obstacle_close = any(
            (self.scan.ranges[i] < safety_dist) for i in front_idx
            if self.scan.ranges[i] > 0.01
        )
        if not obstacle_close:
            # no front obstacles close -> normal lookahead
            return lookahead_point(self.full_path, robot_pos, self.lookahead_dist)

        # unit vectors: segment direction and its perpendicular (left)
        seg_dx = math.cos(seg_dir_yaw)
        seg_dy = math.sin(seg_dir_yaw)
        perp_dx = -seg_dy   # perpendicular left
        perp_dy = seg_dx

        # sample candidates along the perpendicular line: alpha in [-radius, +radius]
        for i in range(-num_samples, num_samples + 1):
            alpha = radius * i / num_samples
            candidate = (
                robot_pos[0] + alpha * perp_dx,
                robot_pos[1] + alpha * perp_dy
            )

            # compute minimum perpendicular distance (relative to segment direction)
            # between the candidate and any front obstacle (only obstacles < safety_dist)
            min_perp_dist = float('inf')
            for idx in front_idx:
                r = self.scan.ranges[idx]
                if r < 0.01 or r > safety_dist:
                    continue
                angle_rad = self.scan.angle_min + idx * self.scan.angle_increment
                # If scan angles are in robot frame, and you have robot yaw (self.yaw),
                # you may need: angle_world = angle_rad + self.yaw
                angle_world = angle_rad  # adjust if necessary
                obs_x = robot_pos[0] + r * math.cos(angle_world)
                obs_y = robot_pos[1] + r * math.sin(angle_world)

                # vector from candidate to obstacle
                v_x = obs_x - candidate[0]
                v_y = obs_y - candidate[1]

                # perpendicular distance to segment direction = |v x seg_unit|
                # cross product magnitude (since seg unit length = 1)
                perp_dist = abs(v_x * seg_dy - v_y * seg_dx)
                min_perp_dist = min(min_perp_dist, perp_dist)

            # if no obstacle considered for this candidate, assume full clearance = safety_dist
            if min_perp_dist == float('inf'):
                min_perp_dist = safety_dist

            # score: prefer large lateral clearance, penalize large lateral shifts
            score = min_perp_dist - lateral_weight * abs(alpha)

            if score > best_score:
                best_score = score
                best_point = candidate

        # fallback
        if best_point is None:
            best_point = lookahead_point(self.full_path, robot_pos, self.lookahead_dist)

        return best_point



    # --- TODO: YOUR CODE HERE: controller used for TurtleBot ---
    def turtlebot_control(self, robot_pos: Coord, robot_h: float, target: Coord, dt: float, Ld: float) -> Tuple[float, float]:
        """
        Compute (v, w) commands that blend pure-pursuit with PID:

        Args:
            robot_pos: current pose (x, y) in odom/world frame.
            robot_h: current heading (yaw angle) in the odom/world frame.
            target: target point (x, y) to track.
            dt: timestep (s).
            Ld: pure-pursuit lookahead distance (m).

        Returns:
            (v_cmd, w_cmd): linear and angular velocity commands.
        """
        # Calculate desired heading toward target
        dx = target[0] - robot_pos[0]
        dy = target[1] - robot_pos[1]
        desired_heading = math.atan2(dy, dx)
        
        # Calculate heading error
        heading_error = angle_wrap(desired_heading - robot_h)
        
        # Angular control using PID
        w_cmd = self.pid_angular.update(heading_error, dt)
        w_cmd = np.clip(w_cmd, -self.w_max, self.w_max)
        
        # Linear control - reduce speed when turning sharply
        distance_to_target = math.dist(robot_pos, target)
        v_base = min(self.v_max, distance_to_target * 0.5)  # Base speed
        
        # Slow down when making sharp turns
        turn_factor = 1.0 - min(1.0, abs(heading_error) / (math.pi/2))
        v_cmd = v_base * turn_factor
        
        # Further reduce speed if very close to target
        if distance_to_target < 0.2:
            v_cmd *= distance_to_target / 0.1
        
        v_cmd = np.clip(v_cmd, 0.0, self.v_max)
        
        return v_cmd, w_cmd

    # --- Publisher helpers ---
    def publish_cmd(self, v: float, w: float):
        """
        Publish a geometry_msgs/Twist command.

        Args:
            v: linear velocity (m/s), applied to msg.linear.x.
            w: angular velocity (rad/s), applied to msg.angular.z.
        """
        msg = Twist(); msg.linear.x = float(v); msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def publish_stop(self):
        """
        Convenience: publish zero velocities to stop the turtlebot.
        """
        self.publish_cmd(0.0, 0.0)

    # --- Vizualization helper ---
    def draw_scene(self, robot: Optional[Coord] = None, target: Optional[Coord] = None):
        """
        Render the grid, nodes, path, turtlebot and target for debugging.

        Args:
            robot: optional (x, y) turtlebot position to plot.
            target: optional (x, y) target point to plot.
        """
       
        extra = [self.start_pos, self.goal_pos]
        if robot: extra.append(robot)
        if target: extra.append(target)
        img, to_px = self.graph.base_image(extra_points=extra, img_size=800, margin=60)
        # edges (grey)
        for a, b in self.graph.edges():
            cv2.line(img, to_px(*a), to_px(*b), (220, 220, 220), 2)
        # nodes (purple)
        for xy in self.graph.coord2idx.keys():
            cv2.circle(img, to_px(*xy), 6, (240, 32, 160), -1)
        # path (black)
        if self.full_path and len(self.full_path) >= 2:
            pts = np.array([to_px(*p) for p in self.full_path], dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=False, color=(30, 30, 30), thickness=3)

        # start (red) & goal (green)
        cv2.circle(img, to_px(*self.start_pos), 9, (0, 0, 220), -1)
        cv2.circle(img, to_px(*self.goal_pos), 9, (0, 220, 0), -1)
        # turtlebot
        if robot is not None:
            pr = to_px(*robot)
            cv2.circle(img, pr, 7, (0, 165, 255), -1)
            hx = robot[0] + math.cos(self.yaw) * 0.3
            hy = robot[1] + math.sin(self.yaw) * 0.3
            cv2.arrowedLine(img, pr, to_px(hx, hy), (0, 165, 255), 2, tipLength=0.3)
        # lookahead target (blue)
        if target is not None:
            cv2.circle(img, to_px(*target), 7, (200, 0, 0), -1) 
        # HUD
        cv2.putText(img, f"Mode: {self.mode}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imshow(self.win, img); cv2.waitKey(1)
    
    # --- Main control loop ----
    def control_step(self):
        """
        Main control loop
        """

        # --- Stamping current time ---
        now = self._now()
        dt = max(1e-3, now - self.last_t)
        self.last_t = now

        # --- No odometry values, no calculated A* path, or path is only one coordinate ---
        if not self.pose_ready or self.full_path is None or len(self.full_path) < 2:
            self.draw_scene()
            return

        # --- Current turtlebot position ---
        robot_position = (self.x, self.y)
        robot_heading  = self.yaw

        # --- Check if goal reached ---
        if math.dist(robot_position, self.goal_pos) < self.goal_tol:
            self.publish_stop()
            self._set_mode('DONE')
            self.get_logger().info(f'GOAL REACHED!!!')
            self.draw_scene(robot=robot_position)
            return

        # --- (Optional) Pause motion ---
        if self.dwell_until is not None and now < self.dwell_until:
            self.publish_stop()
            self.draw_scene(robot=robot_position)
            return
        else:
            self.dwell_until = None

        # --- Find the current path segment and its properties ---
        _, seg_idx, _ = nearest_point_on_polyline(self.full_path, robot_position) # which path segment (between two waypoints) the turtlebot is currently on
        ga = self.full_path[seg_idx]     # segment’s start point
        gb = self.full_path[seg_idx + 1] # segment’s end point
        seg_vec = (gb[0] - ga[0], gb[1] - ga[1]) # vector pointing from a → b (i.e. local forward direction)
        seg_dir_yaw = math.atan2(seg_vec[1], seg_vec[0]) # heading (yaw angle) of this segment in the world frame
        dist_to_b = math.dist(robot_position, gb) # robot’s straight-line distance to the next waypoint (b)

        # --- Check arrival at the node 'b' and (Optional) pause motion ---
        if dist_to_b <= self.node_tol:
            if seg_idx + 1 < len(self.full_path):
                self.full_path = self.full_path[seg_idx + 1:]
            self.publish_stop()
            self.dwell_until = self._now() + self.dwell_s
            self.draw_scene(robot=robot_position, target=gb)
            return

        # --- MODE: REPLAN ---
        if self.mode == 'REPLAN':
            self.publish_stop()
            # TODO: YOUR CODE HERE:
            # Remove the edge if the edge is blocked 
            if self.block_edge is not None:
                self.graph.remove_edge(self.block_edge[0], self.block_edge[1])
            
            # Set start_from to the previous node
            start_from = self.prev_node
            
            # Replan the global path from new start point and store it in self.full_path.
            self.full_path = self.plan_from(start_from)
            
            # Set the mode back to FOLLOW
            self._set_mode('FOLLOW')
            self.draw_scene(robot=robot_position)
            return

        # --- MODE: BACKTRACK ---
        if self.mode == 'BACKTRACK':
            # TODO: YOUR CODE HERE
            # ~1 line: set a temporary goal as the previous node
            temp_goal = self.prev_node if self.prev_node is not None else self.full_path[0]

            back_path = [robot_position, temp_goal]
            target = lookahead_point(back_path, robot_position, self.lookahead_dist)
            v_cmd, w_cmd = self.turtlebot_control(robot_position, robot_heading, target, dt, self.lookahead_dist) # Get linear and angular velocity values from your control block
            self.publish_cmd(v_cmd, w_cmd) # publish the velocities
            self.draw_scene(robot=robot_position, target=target)

            # TODO: YOUR CODE HERE: ~3 lines
            # If the distance between the current turtlebot position ('robot') and temp_goal is less than the tolerance ('self.node_tol'), stop the turtlebot immediately and set the mode to REPLAN
            if math.dist(robot_position, temp_goal) < self.node_tol:
                self.publish_stop()
                self._set_mode('REPLAN')
            return

        # --- MODE: FOLLOW ---
        # React if an obstacle has become non-passable
        if not self.is_passable(seg_dir_yaw):
            # TODO: YOUR CODE HERE: ~4 lines
            # 1. Stop the turtlebot immediately
            self.publish_stop()
            # 2. set 'prev_node' to 'ga'
            self.prev_node = ga
            # 3. set 'block_edge' to '(ga, gb)'
            self.block_edge = (ga, gb)
            # 4. set mode to 'BACKTRACK'
            self._set_mode('BACKTRACK')
            self.draw_scene(robot=robot_position)
            return
            
        target = self.obstacle_avoided_target(robot_position, seg_dir_yaw) # Get obstacle avoided lookahead target point
        v_cmd, w_cmd = self.turtlebot_control(robot_position, robot_heading, target, dt, self.lookahead_dist) # Get linear and angular velocity values from your control block
        self.publish_cmd(v_cmd, w_cmd) # publish the velocities
        self.draw_scene(robot=robot_position, target=target)


# --------------- entry ---------------
def main():
    rclpy.init()
    node = GridPathFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main()