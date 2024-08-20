## Overview and Motivation

### Traditional Autonomous Driving Systems
- Typically use a modular pipeline where tasks like perception, prediction, and planning are handled in separate stages.
- The output of one module serves as the input to the next, often leading to suboptimal feature representation and compounded errors across stages.

### End-to-End Vision-Based Approach
- Proposes a unified model (ST-P3) that directly takes raw sensor data (e.g., camera images) as input and outputs control signals or planned routes.
- The goal is to optimize feature representations simultaneously across perception, prediction, and planning tasks within a single network.

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

## Experimental Evaluation

### Datasets Used
- **nuScenes Dataset**: For open-loop evaluation.
- **CARLA Simulator**: For closed-loop evaluation to test real-time driving scenarios.

### Results
- **Perception**: Achieves high Intersection-over-Union (IoU) scores in BEV segmentation tasks, outperforming previous methods.
- **Prediction**: Excels in predicting future instances and maintaining semantic consistency over time, showing superior performance in metrics like Panoptic Quality (PQ) and recognition quality.
- **Planning**: Demonstrates lower collision rates and more accurate trajectory following compared to other methods, ensuring safer driving outcomes.

## Related Work

### Interpretable End-to-End Framework
- **Focus on Explicit Design for Interpretability**:
  - Emphasis on the importance of having a clear, interpretable design to ensure system safety, particularly in LiDAR-based approaches.
- **Examples of LiDAR-Based Approaches**:
  - **NMP (Neural Motion Planner)**: Utilizes LiDAR and HD maps to predict future bounding boxes of actors and learns a cost volume for selecting the best trajectory.
  - **P3**: Ensures consistency between planning and perception by using a differentiable occupancy representation that explicitly informs planning tasks.
  - **MP3**: Builds an online map from segmentations and the current and future states of other agents; feeds these results into a sampler-based planner to obtain a safe trajectory without relying on HD maps.
  - **LookOut**: Predicts multiple possible futures for a scene and selects the best trajectory by optimizing over a set of contingency plans.
  - **DSDNet**: Considers interactions between actors and provides socially consistent multimodal future predictions, using predicted future distributions to plan safe maneuvers.

### Performance in Urban Scenarios
- These LiDAR-based methods perform well in challenging urban environments, but comparisons are difficult due to the lack of publicly available datasets and baselines.

### Bird's Eye View (BEV) Representation
- **Importance in Planning and Control**:
  - BEV representation is ideal for planning and control tasks because it avoids issues like occlusion and scale distortion, while preserving the 3D scene layout.
- **Challenges with Vision Inputs**:
  - Projecting vision inputs (camera views) to BEV space is non-trivial, especially since ground truth BEV data is often unavailable for direct supervision.
- **Methods for BEV Projection**:
  - **Implicit Projection Methods**:
    - Some methods implicitly project image inputs into BEV, but these often lack quality due to the absence of ground truth for supervision.
  - **Explicit Projection Techniques**:
    - Uses homography to project images into BEV space.
    - Spatial cross-attention methods use predefined BEV queries to obtain BEV features.
    - LSS and FIERY perform projection using estimated depth and image intrinsics, showing impressive performance.
    - ST-P3 enhances this by accumulating all past 3D features to the current ego view, providing better feature representation for subsequent tasks.

### Future Prediction
- **Traditional Methods**:
  - Typically rely on ground truth perception information and HD maps for input, but are prone to cumulative errors when perception inputs are derived from real-world sensors.
- **End-to-End Approaches**:
  - Focus on future trajectory prediction as a key step, often depending on LiDAR and HD maps.
- **Vision-Based Future Prediction**:
  - Recent methods use only camera inputs to predict future scenes in BEV semantic segmentation, achieving good performance.
- **Challenge**:
  - These methods often do not fully capture or exploit the evolution process of past events.
- **ST-P3’s Approach**:
  - Combines probabilistic uncertainty with past dynamics to predict diverse and plausible future scenes, inspired by video future prediction methods.

### Motion Planning
- **Implicit vs. Explicit Methods**:
  - **Implicit Methods**:
    - Directly generate planned trajectories or control commands but suffer from robustness and interpretability issues.
  - **Explicit Methods**:
    - Construct a cost map with a trajectory sampler to choose the optimal trajectory by minimizing the cost.
    - Cost maps can be hand-crafted or learned from intermediate representations like segmentations and HD maps.
- **DSDNet**:
  - Combines hand-crafted and learned costs to create an integrated cost volume for trajectory selection.
- **ST-P3’s Approach**:
  - Adopts a combination of a sampler and GRU refinement unit, with navigation signals to further adjust and optimize the chosen trajectory.

### Prediction: Dual Pathway Probabilistic Future Modelling
- **Traditional Motion Prediction**:
  - Traditionally, motion prediction algorithms predict future trajectories as deterministic or multi-modal results.
  - These methods struggle to account for the complexity of future interactions among agents, traffic elements, and road environments.
- **Challenge of Finite Probability Modeling**:
  - A finite set of predicted outcomes cannot adequately capture the complexity and stochasticity of future scenarios in dynamic driving environments.
- **Incorporating Conditional Uncertainty**:
  - To address the stochastic nature of future events, the approach models future uncertainty using diagonal Gaussian distributions.
- **Gaussian Distribution**:
  - Mean and variance represent the latent channels.
- **Sampling Process**:
  - **During training**: Samples are drawn from a normal distribution.
  - **During inference**: Samples are drawn from a deterministic distribution, effectively ignoring variance for deterministic predictions.
- **Dual Pathway Modeling**:
  - **Architecture**: Integrates Bird's Eye View (BEV) features up to the current timestamp with the future uncertainty distribution into two pathways:
    - **First Pathway**:
      - Uses historical features as inputs to a Gated Recurrent Unit (GRU) network.
      - The initial hidden state for prediction is set by the first feature.
    - **Second Pathway**:
      - Uses the sampled uncertainty features as inputs to another GRU.
      - The initial hidden state for prediction is set by the current feature.
  - **Combination of Pathways**:
    - The predicted feature at time `t+1` is a mixture of the outputs from both pathways, forming a mixed Gaussian.
    - This combined output serves as the basis for further prediction steps.
  - **Recursive Future Prediction**:
    - The dual pathway approach is used to recursively predict future states, where `H` is the prediction horizon.
    - All features, including historical and predicted future states, are fed into a decoder module.
  - **Decoder Module**:
    - **Multiple Output Heads**:
      - The decoder generates different interpretable intermediate representations through various output heads.
    - **Instance Segmentation**: Outputs include instance centerness, offset, and future flow.
    - **Semantic Segmentation**: Focuses on identifying key actors like vehicles and pedestrians.
    - **Drivable Area and Lane Representation**:
      - The system generates interpretable map representations for drivable areas and lanes, crucial in autonomous driving.
    - **Cost Volume Representation**:
      - A cost volume head is designed to represent the expense associated with each possible location that the self-driving vehicle (SDV) might occupy within the planning horizon.
    - **Past Features Accuracy**:
      - The model also decodes features from past frames to improve the accuracy of historical features.
      - Accurate historical features are essential for enhancing the overall prediction performance, contributing to better future predictions.

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
