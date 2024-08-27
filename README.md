## Overview and Motivation

### Traditional Autonomous Driving Systems
- Typically use a modular pipeline where tasks like perception, prediction, and planning are handled in separate stages.
- The output of one module serves as the input to the next, often leading to suboptimal feature representation and compounded errors across stages.

### End-to-End Vision-Based Approach
- Proposes a unified model (ST-P3) that directly takes raw sensor data (e.g., camera images) as input and outputs control signals or planned routes.
- The goal is to optimize feature representations simultaneously across perception, prediction, and planning tasks within a single network.

---

## Key Innovations in ST-P3

### Egocentric-Aligned Accumulation
**Purpose**: To preserve and enhance geometric information in 3D space before converting to Bird’s Eye View (BEV).

**Method**:
- Aligns past frames with the current frame using ego-motion data.
- Accumulates and aggregates features from multi-view camera inputs in 3D space, preserving spatial coherence and enhancing the robustness of the BEV representation.

### Dual Pathway Modeling
**Purpose**: To improve future predictions by considering both the uncertainty of future events and the dynamics of past motion.

**Method**:
- **First Pathway**: Uses historical features (from past frames) to capture motion continuity.
- **Second Pathway**: Incorporates a probabilistic model to account for uncertainty in future predictions, producing stronger and more reliable scene representations.

### Prior-Knowledge Refinement
**Purpose**: To integrate high-level commands and vision-based elements (such as traffic lights) into the final trajectory planning without relying on HD maps.

**Method**:
- Uses features from the early stages of the network along with high-level driving commands.
- Refines the selected trajectory by incorporating vision-based information, particularly focusing on elements like traffic lights, using a lightweight GRU (Gated Recurrent Unit) network.

---

## Detailed Framework of ST-P3

### Perception
- **Input**: Multi-view camera images capturing different perspectives around the vehicle.
- **Process**:
  - Features are extracted and depth is estimated for each frame.
  - All features are aligned in 3D space relative to the vehicle’s current position.
  - Aggregated features are then transformed into a BEV representation for further processing.

### Prediction
- **Task**: Future instance segmentation and trajectory prediction.
- **Process**:
  - Leverages the dual pathway model to enhance prediction accuracy.
  - Predicts future states by combining insights from past dynamics and uncertainty modeling, producing a more reliable semantic map of future scenarios.

### Planning
- **Task**: Generate the safest and most efficient trajectory for the vehicle.
- **Process**:
  - Samples a diverse set of possible trajectories based on the predicted BEV features.
  - Selects the optimal trajectory using a learned cost function that balances safety, comfort, and progress.
  - Refines the selected trajectory using prior knowledge and real-time visual information (e.g., from the front camera).

---
## Perception: Egocentric Aligned Accumulation

**Objective:** 
The goal here is to create a spatiotemporal Bird's Eye View (BEV) feature from multi-view camera inputs over several time steps.

**Challenges Addressed:**
- **Alignment Issues:** Direct concatenation methods suffer from alignment problems.
- **Height Information Loss:** Some existing methods like FIERY lose important height information during the process.

**Proposed Method:** 
The framework introduces an accumulative ego-centric alignment method, which includes two key steps:

1. **Spatial Fusion:**
   - **Transformation:** Multi-view images are transformed into a common 3D frame using depth predictions.
   - **Feature Extraction:** Features from each camera image are lifted into 3D space based on depth estimations, using the equation:

   <p align="center">
     <img src="https://latex.codecogs.com/svg.latex?u_i^k=f_i^k\otimes%20d_i^k" alt="u_i^k = f_i^k \otimes d_i^k"/>
   </p>

   - Here, \( u_i^k \) represents the 3D features, \( f_i^k \) is the feature map, and \( d_i^k \) is the depth map.
   - **Alignment:** These features are then aligned to the current view using the vehicle’s ego-motion and pooled into BEV features.

2. **Temporal Fusion:**
   - **Enhancement of Static Object Perception:** A temporal fusion technique is applied to enhance the perception of static objects by using a self-attention mechanism that boosts the importance of features from previous time steps.
   - **Equation:**

   <p align="center">
     <img src="https://latex.codecogs.com/svg.latex?\tilde{x}_t=b_t+\sum_{i=1}^{t-1}\alpha^i\times\tilde{x}_{t-i}" alt="\tilde{x}_t = b_t + \sum_{i=1}^{t-1} \alpha^i \times \tilde{x}_{t-i}"/>
   </p>

   - Here, \( \tilde{x}_t \) represents the accumulated feature, and \( b_t \) is the BEV feature map.
   - **3D Convolutions:** These fused features are then processed with 3D convolutions to improve the perception of dynamic objects, using the equation:

   <p align="center">
     <img src="https://latex.codecogs.com/svg.latex?x_{1\sim%20t}=\mathcal{C}(\tilde{x}_{1\sim%20t},m_{1\sim%20t})" alt="x_{1\sim t} = \mathcal{C}(\tilde{x}_{1\sim t}, m_{1\sim t})"/>
   </p>

   - Where \( m_{1\sim t} \) is the ego-motion matrix, and \( \mathcal{C} \) represents the 3D convolution network.

---

## Prediction: Dual Pathway Probabilistic Future Modeling

**Overview:**
In dynamic driving environments, predicting future trajectories is challenging due to the uncertainty and stochastic nature of the future. Traditional methods often predict future trajectories deterministically or using a finite set of probable outcomes. However, this approach is insufficient to capture the complexities of interactions among various agents, traffic elements, and road conditions.

**Objective:** 
The aim is to model the uncertainty in future predictions by considering the stochastic nature of the driving environment.

**Methodology:**

1. **Uncertainty Modeling:**
   - **Gaussian Distribution:** The future uncertainty is modeled as diagonal Gaussians with a mean (\(\mu\)) and variance (\(\sigma^2\)). Here, \(\mu\) and \(\sigma^2\) represent the latent channels in the model. 
   - **Sampling During Training:** During training, the system samples from a Gaussian distribution \(\eta_t \sim N(\mu_t, \sigma_t^2)\), but during inference (actual operation), it samples from \(\eta_t \sim N(\mu_t, 0)\), meaning only the mean is considered.

2. **Dual Pathway Architecture:**

   - **Pathway a:** Integrates BEV features up to the current timestamp with the uncertainty distribution. This pathway uses historical features as input to a GRU (Gated Recurrent Unit), where the first feature \(x_1\) is used as the initial hidden state.
   - **Pathway b:** Uses the sampled Gaussian distribution \(\eta_t\) as input to a GRU, with the current feature \(x_t\) as the initial hidden state.

3. **Prediction Combination:**
   - The predicted feature for the next time step \(\hat{x}_{t+1}\) is generated by combining the outputs from both pathways:
   
   <p align="center">
     <img src="https://latex.codecogs.com/svg.latex?\hat{x}_{t+1}=\mathcal{G}(x_t,\eta_t)\oplus\mathcal{G}(x_{0:t})" alt="\hat{x}_{t+1} = \mathcal{G}(x_t, \eta_t) \oplus \mathcal{G}(x_{0:t})"/>
   </p>

   - Here, \(\mathcal{G}\) represents the GRU process, and \(\oplus\) denotes the combination of these predictions.
   - This combined prediction serves as the base for future state predictions (up to \(H\) horizons).

4. **Decoding:**

   - The combined features from both pathways are fed into a decoder, which has multiple output heads. These heads generate different interpretable intermediate representations such as:
     - **Instance Segmentation:** Outputs instance centerness, offset, and future flow for identifying objects like vehicles and pedestrians.
     - **Semantic Segmentation:** Focuses on key actors like vehicles and pedestrians.
     - **HD Map Elements:** Generates interpretable map elements like drivable areas and lane boundaries, which are crucial for autonomous driving.
     - **Cost Volume:** A specific head is designed to represent the cost associated with each possible trajectory within the planning horizon.

---

## Planning Module

**Objective of the Planner:**

The main goal is to plan a safe and comfortable trajectory that will guide the SDV towards a target point while avoiding obstacles and considering traffic rules.

**Trajectory Sampling and Cost Function:**

The system generates a set of possible trajectories using a simplified vehicle model (the bicycle model) and evaluates each trajectory using a cost function. The trajectory with the lowest cost is selected as the best option.

### Equation (5):
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?f(\tau,o,m;w)=f_o(\tau,o,m;w_o)+f_v(\tau;w_v)+f_r(\tau;w_r)" alt="f(\tau, o, m; w) = f_o(\tau, o, m; w_o) + f_v(\tau; w_v) + f_r(\tau; w_r)"/>
</p>

This equation describes the total cost function \( f(\tau, o, m; w) \), which is a sum of three sub-costs:

- \( f_o \): Evaluates the trajectory based on occupancy predictions and map representations, considering safety and compliance with traffic rules.
- \( f_v \): Comes from the prediction module and is based on the learned features (e.g., the predicted future states of the environment).
- \( f_r \): Considers the overall performance of the trajectory, including comfort (e.g., minimizing sudden jerks or sharp turns) and progress towards the destination.

### Sub-Costs:

- **Safety Cost:** Ensures the SDV avoids collisions with other objects and maintains a safe distance from obstacles, particularly at high speeds.
- **Cost Volume:** A learned representation generated by the prediction module that reflects the complexity of the environment. It is clipped to ensure it doesn't dominate the evaluation of trajectories.
- **Comfort and Progress:** Penalizes trajectories that involve excessive lateral acceleration, jerk, or curvature, and rewards trajectories that efficiently move towards the destination.

### High-Level Commands and Target Information:

The cost function does not inherently include target information (e.g., the final destination), which is often available in traditional map-based routing. Instead, the planner uses high-level commands (like "Go Straight" or "Turn Left") to evaluate and select trajectories that align with the desired action.

### Selecting the Optimal Trajectory:

### Equation (6):
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\tau^*=\arg\min_{\tau_h}f(\tau_h,c)=\arg\min_{\tau_h}f(\tau_h,o,m;w)" alt="\tau^* = \arg \min_{\tau_h} f(\tau_h, c) = \arg \min_{\tau_h} f(\tau_h, o, m; w)"/>
</p>

This equation identifies the optimal trajectory \( \tau^* \) from the set of possible trajectories \( \tau_h \) by minimizing the cost function \( f(\tau_h, o, m; w) \).

- \( \tau_h \) represents the set of possible trajectories under the given high-level command, and \( c \) represents the overall cost map.

### GRU-Based Refinement:

After selecting the optimal trajectory, the system further refines it using a GRU network. This step integrates information from the front-view camera (such as the status of traffic lights) to ensure the trajectory is safe and appropriate given the current traffic conditions.

The GRU refines the trajectory by processing the trajectory points \( \tau^* \) and adjusting them based on real-time visual information from the cameras.

---

## Breakdown of the Loss Function

### Overall Loss Function (Equation 7):
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mathcal{L}=\mathcal{L}_{per}+\alpha\mathcal{L}_{pre}+\beta\mathcal{L}_{pla}" alt="\mathcal{L} = \mathcal{L}_{per} + \alpha \mathcal{L}_{pre} + \beta \mathcal{L}_{pla}"/>
</p>

### Components:

- **\( \mathcal{L}_{per} \): Perception loss.**
- **\( \alpha \mathcal{L}_{pre} \): Prediction loss, scaled by a learnable weight \( \alpha \).**
- **\( \beta \mathcal{L}_{pla} \): Planning loss, scaled by a learnable weight \( \beta \).**

### Learnable Weights:
\( \alpha \) and \( \beta \) are not fixed constants but are learnable parameters. This allows the model to dynamically balance the contribution of each loss component based on the gradients during training. This approach follows protocols from previous research, such as in [30, 11].

### Perception Loss \( \mathcal{L}_{per} \):

- **Components:**
  - **Segmentation Loss:** This includes the loss for segmenting both current and past frames. It also includes losses related to mapping (e.g., lane and drivable area prediction) and depth prediction.
  - **Top-k Cross-Entropy Loss:** Used for semantic segmentation, focusing on the most relevant classes (since the BEV image is largely dominated by background).
  - **L2 and L1 Losses:** Used for instance segmentation tasks like centerness supervision and offset/flow prediction.
  - **Depth Loss:** While some methods optimize depth prediction implicitly, ST-P3 uses a pre-generated depth value from another network for direct supervision.

### Prediction Loss \( \mathcal{L}_{pre} \):

- **Semantic and Instance Segmentation:** The prediction module also infers future semantic and instance segmentation, using a similar top-k cross-entropy loss as in the perception task.
- **Discounting Future Losses:** Future predictions are more uncertain, so losses for future timestamps are exponentially discounted to account for this uncertainty.

### Planning Loss \( \mathcal{L}_{pla} \):

- **Components:**
  - **Max-Margin Loss:** The model treats expert behavior \( \tau_h \) as a positive example and trajectories sampled from the set \( \tau \) as negative examples. The max-margin loss helps ensure that the expert behavior is preferred over sampled trajectories.
  - **L1 Distance Loss:** Measures the distance between the planned trajectory and the expert trajectory. This loss is used to refine the selected trajectory and bring it closer to what a human expert might choose.

### Equation (8):
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_{pla}=\max_{\tau}\left[f(\tau_h,c)-f(\tau,c)+d(\tau_h,\tau)\right]_++d(\tau_h,\tau_o^*)" alt="\mathcal{L}_{pla} = \max_{\tau} \left[ f(\tau_h, c) - f(\tau, c) + d(\tau_h, \tau) \right]_+ + d(\tau_h, \tau_o^*)"/>
</p>

- **ReLU Function \( [\cdot]_+ \):** Ensures that the loss is non-negative.
- **Distance \( d(\tau_h, \tau) \):** Measures how far the sampled trajectory \( \tau \) is from the expert trajectory \( \tau_h \). The goal is to minimize this distance for the selected trajectory.

---

## Experiments

### Overview
- ST-P3 is evaluated in both open-loop and closed-loop environments.

### Datasets Used
- **nuScenes Dataset**: Used for open-loop evaluation, focusing on past 1.0s context to predict 2.0s into the future (3 frames in the past, 4 frames in the future).
- **CARLA Simulator**: Used for closed-loop experiments to demonstrate the robustness and applicability of ST-P3.

### Open-loop Experimental Results on nuScenes
- **Perception**:
  - Evaluated on map representation (drivable area and lanes) and semantic segmentation (vehicles and pedestrians).
  - **Metric**: Intersection-over-Union (IoU) in BEV segmentation.
  - **Results**:
    - ST-P3 outperforms other models in most cases, achieving the highest mean IoU value (42.69%).
    - The Egocentric Aligned Accumulation algorithm contributed to surpassing the previous state-of-the-art by 2.51%.
- **Prediction**:
  - Focuses on predicting future segmentation in BEV.
  - **Metrics**: IoU, Panoptic Quality (PQ), Recognition Quality (RQ), Segmentation Quality (SQ).
  - **Results**:
    - ST-P3 achieves state-of-the-art results across all metrics.
    - The Gaussian version of ST-P3 performs slightly worse than the Bernoulli version, but is chosen for its smaller memory usage.
- **Planning**:
  - **Evaluation Metrics**: L2 error (between planned and human driving trajectories) and collision rate.
  - **Results**:
    - ST-P3 achieves the lowest collision rate, indicating superior safety in planned trajectories.
    - Although the Vanilla approach had the lowest L2 error, it resulted in the highest collision rates.

### Closed-loop Planning Results on CARLA Simulator
- **Closed-Loop Experiments**:
  - Conducted in the CARLA simulator to assess ST-P3's robustness in dynamic and cumulative error-prone environments.
- **Metrics**:
  - Route Completion (RC): The percentage of the route completed.
  - Driving Score (DS): RC weighted by penalties for collisions with pedestrians, vehicles, etc.
- **Results**:
  - ST-P3 outperforms vision-based baselines in all scenarios.
  - Achieves better route completion in long-range tests compared to a LiDAR-based method.
  - ST-P3 demonstrates impressive recovery from collisions, aided by front-view vision refinement.

### Ablation Study
- **Objective**:
  - To evaluate the effectiveness of various components in ST-P3, such as depth supervision, Egocentric Aligned Accumulation (EAA), Dual Modelling, and the sampler and refinement units.
- **Perception Module**:
  - **Experiments 1-3**: Examine the impact of depth supervision and EAA on perception tasks.
  - **Results**:
    - EAA improves vehicle IoU by 0.79%.
    - Explicit depth supervision adds a further 1.31% improvement.
- **Prediction Module**:
  - **Experiments 4-6**: Assess the impact of Dual Modelling and Loss for All timestamps (LFA) on prediction tasks.
  - **Results**:
    - Dual Modelling, which considers uncertainty and historical continuity, enhances vehicle IoU by 1.54% and vehicle PQ by 3.09%.
- **Planning Module**:
  - **Experiments 7-9**: Focus on the sampler and GRU refinement units in planning.
  - **Results**:
    - A sampler without front-view vision refinement (Exp.7) or implicit models without prior sampling knowledge (Exp.8) result in higher L2 errors and collision rates.
    - ST-P3’s design significantly improves the safety and accuracy of planned trajectories.
