# MEC-E5012 Vehicle Mechatronics: Control – TurtleBot Maze Navigation

## 🧠 Project Overview

This repository contains the ROS2-based implementation for the final project in **MEC-E5012: Vehicle Mechatronics - Control** at Aalto University.

The task is to navigate a TurtleBot through a custom 5x5 maze track in Gazebo, avoiding obstacles and finding a valid path to the goal using autonomous control strategies.

> **Latest exercise update:** September 15, 2025

---

## Project Objectives

- Implement a Python ROS2 node (`navigator.py`) to control the TurtleBot.
- Apply perception, control, and path planning strategies to navigate a challenging maze environment.
- Handle dynamic path planning using the A* algorithm when blocked.
- Avoid hardcoding: the solution must generalize to other mazes.
- Achieve robust and repeatable navigation performance.

---

## Environment Setup

### Required Files from MyCourses

1. `project_track.zip` – Maze track with obstacles  
2. `project_track_no_obs.zip` – Maze track without obstacles  
3. `vmc2_project.zip` – ROS2 package containing simulation setup and template node

---

## 🧭 Navigation Framework

Implementation should be structured around the following modules:

### 1. **Perception**

* Detects whether a path between two nodes is blocked or passable using sensor data.

### 2. **Control**

* Controls the robot’s motion through the maze using PID or other strategies.
* Avoids collisions with maze walls and obstacles.

### 3. **Path Planning**

* Computes the best route to the goal using A* algorithm.
* Recomputes when a path becomes blocked.

---

## Robot Navigation States

The TurtleBot uses a finite state machine with four key states:

* `FORWARD` – Moves forward and checks passability
* `BACKTRACK` – Returns to the previous node if blocked
* `REPLAN` – Recomputes the path using A* if the current path is invalid
* `DONE` – Goal reached

---

## Development guide

1. Start with the **obstacle-free** map for debugging and tuning.
2. Avoid hardcoded paths or positions.
3. Use built-in or custom PID controllers as needed.
4. Rotate the TurtleBot at nodes before moving to the next segment.
5. For testing, use teleportation commands to jump between checkpoints (see below).

---

## Testing and Teleportation

Use these commands to teleport the robot (after stopping `navigator.py`):

<details>
<summary>Checkpoint 1</summary>

```bash
ros2 service call /set_entity_state gazebo_msgs/srv/SetEntityState "{
  state: {
    name: 'burger',
    pose: {
      position: {x: 3.560, y: -4.272, z: 0.0},
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
    },
    reference_frame: 'world'
  }
}"
```

</details>

<details>
<summary>Checkpoint 2</summary>

```bash
ros2 service call /set_entity_state gazebo_msgs/srv/SetEntityState "{
  state: {
    name: 'burger',
    pose: {
      position: {x: 4.984, y: -4.272, z: 0.0},
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
    },
    reference_frame: 'world'
  }
}"
```

</details>

<details>
<summary>Checkpoint 3</summary>

```bash
ros2 service call /set_entity_state gazebo_msgs/srv/SetEntityState "{
  state: {
    name: 'burger',
    pose: {
      position: {x: 7.120, y: -5.696, z: 0.0},
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
    },
    reference_frame: 'world'
  }
}"
```

</details>

---

## Grading Criteria (Total: 75 points)

| Criteria                         | Points                                     |
| -------------------------------- | ------------------------------------------ |
| Complete maze without collisions | 60                                         |
| Finish within 2.5 minutes        | 10                                         |
| Presentation at Gala             | 5                                          |
| Each checkpoint reached          | 20 each (max 60, partial credit available) |

> **Passing Threshold:** 30 points

---

## 📁 Repository Structure

```
vmc2_project/
├── launch/
│   └── project.launch.py
├── models/
│   └── (Gazebo model files)
├── src/
│   └── navigator.py
├── world/
│   └── (Custom world files)
└── README.md
```

---

## Notes and Best Practices

* Don't hardcode the maze layout or coordinates.
* Avoid modifying or deleting the provided structure unless necessary.
* Use the **PID class** only if your strategy needs it.
* Be consistent across test runs to ensure reliability.
* Start simple, debug in the no-obstacle map first.

---

## 📄 License

This project is part of the academic course MEC-E5012 at Aalto University. Not intended for commercial or production use.

---
