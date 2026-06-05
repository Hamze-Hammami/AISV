# S.U.S.H.I 🍣
![System Block Diagram](https://github.com/user-attachments/assets/0342a10b-6043-4a94-81f2-bc1c477583ba)
**Surface Utilization Surveillance and Hazard Identification**

## Goal of S.U.S.H.I 🍣 
S.U.S.H.I aims to create a vision system for detecting floating surface trash with optimal accuracy, discarded objects, and similar objects that are outside the water area.

## Seeing the Water (U-Net)

A U-Net built from scratch in PyTorch, trained via knowledge distillation from SAM2. Rather than using standard transfer learning, SAM2 acts as an offline teacher — its soft logits are precomputed and stored in `distill.h5`, with no gradient propagation through SAM2 during training. The student U-Net learns from both binary ground truth masks and the teacher's logits simultaneously:

$$\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{hard} + (1 - \alpha) \cdot \mathcal{L}_{soft}$$

$$\mathcal{L}_{hard} = \text{BCE}(\hat{y}, y) \qquad \mathcal{L}_{soft} = \text{MSE}(\hat{y}, z_{SAM2}) \qquad \alpha = 0.5$$

This guides the student toward SAM2-like boundary precision while remaining lightweight enough for edge deployment. The architecture uses 4 encoders, 1 backbone encoder, and 4 decoders with skip connections preserving fine boundary details. The final layer uses sigmoid activation to produce binary water masks. The output is used directly in [Vision Fusion](#vision-fusion) for object validity checks and exploration.

![U-Net Architecture](https://github.com/user-attachments/assets/770f9530-a64d-4441-902a-4cc9c5c65d58)

## Vision Fusion
Three models run in parallel:
- **YOLO Detection** — trained on simulation + pool data (2 classes: goal, obstacle). Uses CSRT-tracked video annotations and mosaic augmentation. No pretrained weights.
- **[Water Segmentation (U-Net)](#water-segmentation-u-net)** — knowledge distillation from SAM2 logits. Combines hard loss (BCE) and soft loss (MSE against teacher logits). Requires only 500–550 frames from 20 video sequences.
- **Depth Estimation** — [Depth-Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), used over stereo cameras as it outperforms depth with water reflections.![Uploading Unet.png…]()


![Vision Fusion](https://github.com/user-attachments/assets/cdf9b3f4-d676-4aee-9db8-c9f3fbd3f493)

## Multi-Field Synthesis (MFS)
MFS solves the local minima problem in classical APF by blending:
- **Local APF forces** — reactive obstacle repulsion and goal attraction
- **Global wavefront flow field** — computed periodically from goal cell outward
- **Clearance field** — to find narrow corridors

Where an adaptive blending coefficient γ ∈ [0,1] shifts between APF and wavefront, increasing wavefront influence in narrow corridors, escaping local minima after repeated stuck steps using wavefront stagnation, and switching to APF dominance for precise final approach based on goal proximity. Dynamic repulsion scaling (β) reduces repulsive forces in traversable corridors. Oscillation is broken via angular perturbations. Hermite cubic interpolation smooths the final path.

![MFS Components](https://github.com/user-attachments/assets/384f7f53-78e9-4bec-b6cc-cd93042e73f5)

MFS uses SUSHI's Vision Fusion to construct the obstacle and goal map:

![MFS Vision Grid](https://github.com/user-attachments/assets/35aa0ff0-2a6a-48be-9a6d-daa41a178b1b)

## Visual Exploration
When no explicit goal is detected, a line-of-sight inspired exploration samples the water mask for curious navigation:
- Density map via 15×15 Gaussian kernel convolution
- Distance transform for shore clearance
- Multi-scale structural kernels detect narrow channels and dead-ends
- Sector-based scoring (16 angular wedges) combining density, reactivity, and structural features
- Three-tier fallback: global max → hierarchical grid → uniform grid

![Exploration Pipeline](https://github.com/user-attachments/assets/bf04413b-2eae-45e7-a803-981747cbaa13)

This allows the robot to navigate safely around obstacle-filled areas, find entrances to disconnected water regions and avoid wall collision during exploration — only through water mask sampling:

![Exploration Examples](https://github.com/user-attachments/assets/03e8f2da-6fc9-4798-a7c8-38d16e1c5d32)

## Decision Control
Behavior hierarchy (lowest to highest priority):

1. **Fuzzy Path Following** — follows waypoints using 4-input triangular membership functions (distance to lookahead, heading error, cross-track error, obstacle proximity) with centroid defuzzification:

$$\mu_A(x) = \max\left(0,\ 1 - \frac{|x - c|}{a}\right)$$

2. **DWA Obstacle Avoidance** — triggered by sonar or horizon depth readings, constructs a dynamic window of velocities and scores each candidate trajectory:

$$W = \{(v, \omega) \mid v \in [v_{min}, v_{max}] \cap [v - a_{max}\Delta t,\ v + a_{max}\Delta t],\ \omega \in [\omega_{min}, \omega_{max}] \cap [\omega - \alpha_{max}\Delta t,\ \omega + \alpha_{max}\Delta t]\}$$

$$R(v, \omega) = -w_h|\theta_{goal} - \theta_N| - w_o d_{min}^{-2} + w_s v$$

3. **Turn-in-Place** — mini-DWA with $v_{min} = v_{max} = 0$, restricting motion to pure rotation when all trajectories collide or the water mask becomes unreliable near shore.

![Global Trajectory](https://github.com/user-attachments/assets/ad5b539d-17d2-43fd-a930-c14eee9fa418)

