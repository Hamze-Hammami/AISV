# (multi?) path planning and exploration

## Potential Fields

Potential fields are a method used in robotics to guide a robot towards a goal while avoiding obstacles. This approach uses artificial potential fields to simulate attraction and repulsion forces:

- **Attractive Potential:** This force draws the robot towards its goal.

### Visualizing Attractive Force

<img src="https://github.com/user-attachments/assets/ba5902a8-c187-41fd-aabc-36a7c2866102" alt="Attractive Force" width="400"/>
*Figure: Visualization of Attractive Force*

- **Repulsive Potential:** This force pushes the robot away from obstacles.
### Visualizing Repulsive Force

<img src="https://github.com/user-attachments/assets/fedced02-d6d9-4d43-807b-234764da8e12" alt="Repulsive Force" width="400"/>


## Frontier-Based Exploration

Frontier-based exploration is a strategy used to explore unknown environments by identifying and navigating towards the boundaries between known and unknown areas:

- **Frontier Detection:** Frontiers are detected by analyzing the occupancy grid, where unknown areas are adjacent to explored areas. These frontiers represent potential areas for further exploration.

### Visualizing Frontier Detection

<img src="https://github.com/user-attachments/assets/d79ec676-f8cd-4dcf-b989-3b6fe5dbfa09" alt="Frontier Detection" width="400"/>

