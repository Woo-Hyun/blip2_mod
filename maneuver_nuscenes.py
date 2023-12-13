import torch
import numpy as np
from tqdm import tqdm

def process_maneuver(datas:Dict,
                        ) -> str :
    # classifier for maneuver 
    kMaxDisplacementForStationary = 3.0          # (m)
    kMaxLateralDisplacementForStraight = 3.0     # (m)
    kMinLongitudinalDisplacementForUTurn = -5.0  # (m)
    kMaxAbsHeadingDiffForStraight = 0.35   # (rad)
    heading_delta_threshold = 2.85
    agent_positions = datas['agent_xy']
    origin_heading_vector = agent_positions[19] - agent_positions[16]
    origin_heading_delta =  torch.atan2(origin_heading_vector[1], origin_heading_vector[0])
    origin_rotate_mat = torch.tensor([[torch.cos(origin_heading_delta), -torch.sin(origin_heading_delta)],
                            [torch.sin(origin_heading_delta), torch.cos(origin_heading_delta)]])
    agent_positions = torch.matmul(agent_positions,origin_rotate_mat)
    start_heading_delta = torch.atan2(origin_heading_vector[1],origin_heading_vector[0])
    final_heading_delta = torch.atan2(agent_positions[-1][1] - agent_positions[-3][1], agent_positions[-1][0] - agent_positions[-3][0]) 
    if np.abs(final_heading_delta) > heading_delta_threshold :
        final_heading_delta = 0
    heading_delta = final_heading_delta - start_heading_delta
    xy_delta = agent_positions[-1] - agent_positions[19]
    final_displacement = np.linalg.norm(xy_delta)
    if final_displacement < kMaxDisplacementForStationary:
        return "stationary"
    if (np.abs(xy_delta[1]) < kMaxLateralDisplacementForStraight) or (np.abs(heading_delta) < kMaxAbsHeadingDiffForStraight):
        return "straight"
    if heading_delta < -kMaxAbsHeadingDiffForStraight and xy_delta[1]:
        return "right_u_turn" if xy_delta[0] < kMinLongitudinalDisplacementForUTurn \
            else "right_turn"
    if xy_delta[0] < kMinLongitudinalDisplacementForUTurn:
        return "left_u_turn"
    return "left_turn"