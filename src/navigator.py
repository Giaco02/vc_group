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
        self.lookahead_dist = 0.55
        self.v_max = 5.0
        self.w_max = 5.0
        # ...
        # TODO: YOUR CODE HERE: ~1-2 lines: set your PID controller/s for linear/angular motion
        self.pid_angular = PID(kp=1.0, ki=0.0, kd=0.05, i_limit=0.5)
        self.pid_linear = PID(kp=0.5, ki=0.0, kd=0.0, i_limit=0.2)

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
        self.alignment_target_yaw: Optional[float] = None  # For ALIGN mode
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
    # Conservative navigation functions for maze-like environments

    # ============================================================================
    # IMPROVED is_passable - More conservative passage detection
    # ============================================================================
    def is_passable(self, seg_dir_yaw: float) -> bool:
        """
        Decide if the edge between two nodes is passable or not
        Args:
            seg_dir_yaw: segment heading in world/odom frame (radians).
        Returns:
            True if passable; False otherwise.
        """
        if self.scan is None:
<<<<<<< HEAD
            return True  # Assume passable if no scan data.

        # Parameters for obstacle detection
        cone_angle = math.radians(30)  # Cone to check ahead
        critical_dist = 0.45  # INCREASED - detect obstacles further away
        safety_dist = 0.30  
        min_passage_width = 0.22
        robot_diameter = 0.158

        # Calculate the relative yaw of the path segment
        rel_yaw = angle_wrap(seg_dir_yaw - self.yaw)

        # Collect ALL readings in the forward cone with their positions
        obstacle_points = []
        angle = self.scan.angle_min

        for r in self.scan.ranges:
            ang_diff = angle_wrap(angle - rel_yaw)
            if abs(ang_diff) <= cone_angle:
                if self.scan.range_min < r < self.scan.range_max:
                    # Convert to robot frame
                    x_r = r * math.cos(angle)
                    y_r = r * math.sin(angle)

                    # Rotate to path-aligned frame
                    x_seg = math.cos(rel_yaw) * x_r + math.sin(rel_yaw) * y_r
                    y_seg = -math.sin(rel_yaw) * x_r + math.cos(rel_yaw) * y_r

                    obstacle_points.append((x_seg, y_seg, r))

            angle += self.scan.angle_increment

        if not obstacle_points:
            return True  # No obstacles detected

        # Check for any close obstacle in path
        min_dist = min(pt[2] for pt in obstacle_points)

        # CRITICAL: If anything is close, be more careful
        if min_dist < critical_dist:
            # Separate by sides
            left_points = [pt for pt in obstacle_points if pt[1] > 0.03]
            right_points = [pt for pt in obstacle_points if pt[1] < -0.03]
            center_points = [pt for pt in obstacle_points if abs(pt[1]) <= 0.03]

            # Check for center blockage (like a ball directly in path)
            if center_points:
                center_min_dist = min(pt[2] for pt in center_points)
                if center_min_dist < safety_dist:
                    # Something directly blocking the path
                    self.get_logger().info(f"Center blockage detected at {center_min_dist:.2f}m")
                    return False

            # Check if it's a narrow corridor
            if left_points and right_points:
                left_min = min(abs(pt[1]) for pt in left_points)
                right_min = min(abs(pt[1]) for pt in right_points)
                corridor_width = left_min + right_min

                required_width = robot_diameter + min_passage_width

                if corridor_width < required_width:
                    self.get_logger().info(f"Corridor too narrow: {corridor_width:.2f}m < {required_width:.2f}m")
                    return False

                # Wide enough corridor
                return True

            # Obstacle on one side only
            if left_points or right_points:
                side_min = min_dist
                if side_min < safety_dist:
                    # Check if there's enough room on the other side
                    if left_points and not right_points:
                        # Obstacle on left, check if we can pass on right
                        return True  # Assume passable
                    elif right_points and not left_points:
                        # Obstacle on right, check if we can pass on left
                        return True

            # Default: if something close and unclear, be cautious
            if min_dist < safety_dist * 0.8:
                return False

        return True  # Far enough, passable


    # == ==========================================================================
    # IMPROVED obstacle_avoided_target - With alignment awareness
    # ============================================================================
=======
            self.get_logger().warn("No LIDAR data yet — assuming path is passable.")
            return True

                # --- Parameters ---
        dist_thresh = 0.3
        min_corridor = 0.2
        robot_width = 0.178
        sector_half_angle = math.radians(30)

        # --- LIDAR geometry ---
        angle_min = self.scan.angle_min       # 0.0
        angle_max = self.scan.angle_max       # 6.28
        angle_inc = self.scan.angle_increment # 0.0175
        ranges = np.array(self.scan.ranges)
        n = len(ranges)

        # --- Compute relative yaw in lidar frame (wrap to [0, 2π)) ---
        rel_yaw = (seg_dir_yaw - self.yaw) % (2 * math.pi)
        center_angle = rel_yaw

        # Compute angular window and wrap within [0, 2π)
        left_bound = (center_angle - sector_half_angle) % (2 * math.pi)
        right_bound = (center_angle + sector_half_angle) % (2 * math.pi)

        # --- Handle wrap-around case (window crosses 0 radians) ---
        if left_bound < right_bound:
            indices = np.arange(int((left_bound - angle_min) / angle_inc),
                                int((right_bound - angle_min) / angle_inc))
        else:
            # e.g., left=350°, right=10°: combine two slices
            indices = np.concatenate((
                np.arange(int((left_bound - angle_min) / angle_inc), n),
                np.arange(0, int((right_bound - angle_min) / angle_inc))
            ))

        # Safety clamp
        indices = np.clip(indices, 0, n - 1).astype(int)
        sector_ranges = ranges[indices]
        sector_angles = (angle_min + indices * angle_inc) % (2 * math.pi)

        # --- Obstacle detection ---
        hits = np.where(sector_ranges < dist_thresh)[0]
        if len(hits) == 0:
            self.get_logger().info("No obstacles detected in sector — passable.")
            return True

        mid_idx = len(sector_ranges) // 2
        left_hits = hits[hits < mid_idx]
        right_hits = hits[hits > mid_idx]

        if len(left_hits) == 0 or len(right_hits) == 0:
            self.get_logger().info("Open space on one side — passable.")
            return True

        # --- Last hit on left, first hit on right ---
        li = left_hits[-1]
        ri = right_hits[0]
        d1 = sector_ranges[li]
        d2 = sector_ranges[ri]
        d_angle = abs(sector_angles[ri] - sector_angles[li])

        # --- Law of Cosines for corridor width ---
        width = math.sqrt(max(0.0, d1**2 + d2**2 - 2*d1*d2*math.cos(d_angle)))
        passable = width > max(min_corridor, robot_width)

        # --- Detailed Logging ---
        self.get_logger().info(
            f"[is_passable] rel_yaw={rel_yaw:.2f} rad "
            f"({math.degrees(rel_yaw):.1f}°), left_idx={indices[0]}, right_idx={indices[-1]}, "
            f"hits_left={len(left_hits)}, hits_right={len(right_hits)}, "
            f"width={width:.3f} m, passable={passable}"
        )

        return passable

    
    # --- TODO: YOUR CODE HERE: Implement the function to generate the lookahead point away from obstacles  ---
>>>>>>> test_mufi
    def obstacle_avoided_target(self, robot_pos: Coord, seg_dir_yaw: float) -> Coord:
        """
        Generate a target point that accounts for obstacle avoidance:
        - Pure-pursuit base target at lookahead distance along the path.
        - Gentle lateral shift to center in corridors.

        NOTE: This function now assumes the robot is already aligned with seg_dir_yaw
        at the start of each segment (handled by control_step alignment mode).

        Args:
            robot_pos: current turtlebot position (x, y) in odom/world frame.
            seg_dir_yaw: segment heading in world/odom frame (radians).
        Returns:
            Target (x, y) in world/odom frame.
        """
<<<<<<< HEAD
        # 1) Pure-pursuit base target
        base_target = lookahead_point(self.full_path, robot_pos, self.lookahead_dist)

        if self.scan is None:
            return base_target

        # 2) Segment direction relative to robot heading
        seg_dir_rel = angle_wrap(seg_dir_yaw - self.yaw)

        # Parameters
        front_cone = math.radians(35)
        x_min_ahead = 0.08
        x_max_ahead = 1.0 * self.lookahead_dist
        safety_threshold = 0.28
        max_shift = 0.04  # Even gentler

=======
        # front = msg.ranges[0]      # Front (0°)
        # left = msg.ranges[90]      # Left (90°)
        # right = msg.ranges[270]    # Right (270°)
        # back = msg.ranges[180]     # Back (180°)

        # If no scan data is available, just fall back to pure-pursuit behavior
        if self.scan is None:
            return lookahead_point(self.full_path, robot_pos, self.lookahead_dist)

        # --- PARAMETERS ---
        base_lookahead = self.lookahead_dist         # default lookahead distance
        min_lookahead = 0.05 * base_lookahead         # shortest lookahead when obstacles are close
        max_shift = 0.3                             # maximum lateral shift (m)
        safety_threshold = 0.5                       # distance where avoidance starts (m)
>>>>>>> test_mufi
        angle_min = self.scan.angle_min
        angle_inc = self.scan.angle_increment
        ranges = self.scan.ranges

<<<<<<< HEAD
        left_vals = []
        right_vals = []

        # 3) Project scan points
        for i, r in enumerate(ranges):
            if not (self.scan.range_min < r < self.scan.range_max):
                continue
            
            beam_angle = angle_min + i * angle_inc
            ang_diff = angle_wrap(beam_angle - seg_dir_rel)

            if abs(ang_diff) > front_cone:
                continue
            
            # Point in robot frame
            x_r = r * math.cos(beam_angle)
            y_r = r * math.sin(beam_angle)

            # Rotate into segment-aligned frame
            x_seg = math.cos(seg_dir_rel) * x_r + math.sin(seg_dir_rel) * y_r
            y_seg = -math.sin(seg_dir_rel) * x_r + math.cos(seg_dir_rel) * y_r

            if x_min_ahead <= x_seg <= x_max_ahead:
                if y_seg > 0:
                    left_vals.append(y_seg)
                elif y_seg < 0:
                    right_vals.append(-y_seg)

        # 4) Calculate clearances
        left_clear = min(left_vals) if left_vals else float("inf")
        right_clear = min(right_vals) if right_vals else float("inf")

        # 5) Gentle centering
        lateral_shift = 0.0

        if left_clear < safety_threshold or right_clear < safety_threshold:
            diff = left_clear - right_clear
            diff_norm = max(-1.0, min(1.0, diff / (safety_threshold * 1.5)))
            lateral_shift = diff_norm * max_shift

        # 6) Apply shift
=======
        # --- FRONT-SECTOR READINGS ---
        # These index ranges depend on LiDAR configuration (assumes ~360 readings per revolution)
        # Adjust if your sensor has a different resolution or angle coverage.
        left_idx = range(30, 45)     # ~+30° to +45°
        right_idx = range(315, 330)    # ~-45° to -30°
        front_idx = list(range(350, 360)) + list(range(0, 11))  # ~center (front zone)

        def avg_range(idxs):
            """Compute average distance for valid (finite) range values."""
            vals = [ranges[i] for i in idxs if math.isfinite(ranges[i])]
            return sum(vals) / len(vals) if vals else float('inf')

        avg_left = avg_range(left_idx)
        avg_right = avg_range(right_idx)
        avg_front = avg_range(front_idx)

        # --- ADAPT LOOKAHEAD DISTANCE ---
        # If there is an obstacle in front, reduce lookahead proportionally.
        if avg_front < safety_threshold:
            factor = avg_front / safety_threshold
            lookahead = min_lookahead + factor * (base_lookahead - min_lookahead)
        else:
            lookahead = base_lookahead

        # --- COMPUTE BASE TARGET ALONG PATH ---
        base_target = lookahead_point(self.full_path, robot_pos, lookahead)

        # --- COMPUTE LATERAL SHIFT ---
        # Shift away from the closer side (toward the side with more free space)
        lateral_shift = 0.0  # default: no shift

        if avg_front < safety_threshold:
            # Shift away from the closer side (toward the side with more free space)
            diff = avg_left - avg_right
            diff_norm = max(-1.0, min(1.0, diff / safety_threshold))
            lateral_shift = diff_norm * max_shift
        # --- APPLY SHIFT IN WORLD FRAME ---
        # Perpendicular to segment direction
>>>>>>> test_mufi
        perp_yaw = seg_dir_yaw + math.pi / 2.0
        target = (
            base_target[0] + lateral_shift * math.cos(perp_yaw),
            base_target[1] + lateral_shift * math.sin(perp_yaw)
        )

<<<<<<< HEAD
        return target


    # ============================================================================
    # IMPROVED turtlebot_control - Supports alignment mode
    # ============================================================================
    def turtlebot_control(self, robot_pos: Coord, robot_h: float, target: Coord, 
                          dt: float, Ld: float) -> Tuple[float, float]:
=======
        return target       
    # --- TODO: YOUR CODE HERE: controller used for TurtleBot ---
    def turtlebot_control(self, robot_pos: Coord, robot_h: float, target: Coord, dt: float, Ld: float) -> Tuple[float, float]:
>>>>>>> test_mufi
        """
        Compute (v, w) commands with conservative speed control.

        When in alignment mode, allow slow forward motion to avoid getting stuck.
        """
        # Calculate desired heading toward target
        dx = target[0] - robot_pos[0]
        dy = target[1] - robot_pos[1]
        desired_heading = math.atan2(dy, dx)

        # Calculate heading error
        heading_error = angle_wrap(desired_heading - robot_h)

        # === ANGULAR CONTROL ===
        w_cmd = self.pid_angular.update(heading_error, dt)
        w_cmd = np.clip(w_cmd, -self.w_max, self.w_max)

        # === LINEAR CONTROL ===
        distance_to_target = math.dist(robot_pos, target)
        aligning = hasattr(self, 'mode') and 'ALIGN' in self.mode

        # Base forward speed (allow small creep in ALIGN to avoid getting stuck)
        v_base = min(
            self.v_max * (0.18 if aligning else 0.35),
            max(0.0, distance_to_target) * (1.0 if aligning else 1.5)
        )

        # Slow down when turning; never drop below 10% of v_base
        turn_factor = max(0.1, 1.0 - abs(heading_error) / (math.pi / 2.0))
        v_cmd = v_base * turn_factor

        # Slow down near the target
        if distance_to_target < 0.25:
            v_cmd *= distance_to_target / 0.25

        # Only hard-stop if almost facing the opposite direction
        if abs(heading_error) > math.radians(140):
            v_cmd = 0.0

        v_cmd = float(np.clip(v_cmd, 0.0, self.v_max))

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

    def is_aligned_with_segment(self, seg_dir_yaw: float, tolerance: float = math.radians(15)) -> bool:
        """
        Check if the robot is aligned with the desired segment direction.
        
        Args:
            seg_dir_yaw: desired heading in world/odom frame (radians).
            tolerance: angular tolerance in radians (default 15 degrees).
        Returns:
            True if aligned within tolerance, False otherwise.
        """
        heading_error = abs(angle_wrap(seg_dir_yaw - self.yaw))
        return heading_error < tolerance
    
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

        # --- MODE: ALIGN (add this BEFORE FOLLOW mode) ---
        if self.mode == 'ALIGN':
            # Rotate in place to face the segment direction
            target_yaw = self.alignment_target_yaw if self.alignment_target_yaw is not None else seg_dir_yaw

            # Create a virtual target point in the desired direction
            align_dist = 0.5  # meters ahead
            target = (
                robot_position[0] + align_dist * math.cos(target_yaw),
                robot_position[1] + align_dist * math.sin(target_yaw)
            )

            # Get control commands (will be rotation only due to ALIGN mode check)
            v_cmd, w_cmd = self.turtlebot_control(robot_position, robot_heading, target, dt, self.lookahead_dist)
            self.publish_cmd(v_cmd, w_cmd)
            self.draw_scene(robot=robot_position, target=target)

            # Check if aligned now
            if self.is_aligned_with_segment(target_yaw, tolerance=math.radians(8)):
                self.get_logger().info("Alignment complete, resuming FOLLOW")
                self._set_mode('FOLLOW')

            return

    # --- MODE: FOLLOW (modified) ---
        if self.mode == 'FOLLOW':
            # First check: Are we aligned with the segment direction?
            if not self.is_aligned_with_segment(seg_dir_yaw, tolerance=math.radians(12)):
                # NOT ALIGNED - Enter alignment mode
                self.publish_stop()
                self._set_mode('ALIGN')
                self.alignment_target_yaw = seg_dir_yaw
                self.get_logger().info(f"Entering ALIGN mode, target: {math.degrees(seg_dir_yaw):.1f}°, current: {math.degrees(self.yaw):.1f}°")
                self.draw_scene(robot=robot_position)
                return

            # Second check: Is the path ahead passable?
            if not self.is_passable(seg_dir_yaw):
                self.publish_stop()
                self.prev_node = ga
                self.block_edge = (ga, gb)
                self._set_mode('BACKTRACK')
                self.draw_scene(robot=robot_position)
                return

            # All clear - proceed with normal following
            target = self.obstacle_avoided_target(robot_position, seg_dir_yaw)
            v_cmd, w_cmd = self.turtlebot_control(robot_position, robot_heading, target, dt, self.lookahead_dist)
            self.publish_cmd(v_cmd, w_cmd)
            self.draw_scene(robot=robot_position, target=target)
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