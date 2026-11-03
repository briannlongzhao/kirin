"""Transfer SMAL skinning weights to a generated mesh."""

import numpy as np
import torch


def build_kinematic_tree(parents):
    children = [[] for _ in range(len(parents))]
    for i, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(i)
    return {"parents": parents, "children": children, "num_joints": len(parents)}


@torch.no_grad()
def compute_smal_skinning_weights(mesh_vertices, smal_model, smal_params, k=10):
    """
    Skinning weights for each mesh vertex as the inverse-distance weighted mean
    of the SMAL weights of its k nearest vertices on the fitted T-pose SMAL mesh.
    """
    device = smal_model.J_regressor.device
    smal_weights = smal_model.lbs_weights.cpu().numpy()
    smal_vertices = smal_model(
        betas=smal_params["betas"].to(device),
        pose=smal_params["pose"].to(device),
        global_orient=smal_params["global_orient"].to(device),
        transl=smal_params["transl"].to(device),
        pose2rot=False,
    ).vertices.squeeze()
    assert smal_vertices.shape[0] == smal_weights.shape[0], "SMAL vertices and weights count mismatch"

    diff = smal_vertices.unsqueeze(0) - torch.from_numpy(mesh_vertices).unsqueeze(1).to(device)
    distances_sq, indices = torch.topk(torch.sum(diff**2, dim=-1), k=k, dim=1, largest=False)
    inv_distances = 1.0 / (torch.sqrt(distances_sq) + 1e-8)
    weights_normalized = (inv_distances / inv_distances.sum(dim=1, keepdim=True)).cpu().numpy()
    indices = indices.cpu().numpy()

    weights = np.zeros((mesh_vertices.shape[0], smal_weights.shape[1]))
    for i in range(mesh_vertices.shape[0]):
        weights[i] = np.sum(smal_weights[indices[i]] * weights_normalized[i][:, np.newaxis], axis=0)
    return weights


def apply_bone_hierarchy_smoothing(weights, kinematic_tree, smoothing_factor=0.1):
    """Reduce a joint's weight where its parent or children have high influence, then renormalize."""
    smoothed_weights = weights.copy()
    for joint_idx in range(weights.shape[1]):
        parent_idx = kinematic_tree["parents"][joint_idx]
        if parent_idx >= 0:
            smoothed_weights[weights[:, parent_idx] > 0.1, joint_idx] *= (1 - smoothing_factor)
        for child_idx in kinematic_tree["children"][joint_idx]:
            smoothed_weights[weights[:, child_idx] > 0.1, joint_idx] *= (1 - smoothing_factor * 0.5)
    return smoothed_weights / (np.sum(smoothed_weights, axis=1, keepdims=True) + 1e-8)


def skinning(mesh, smal_model, smal_params, parents, apply_smoothing=True, disabled_joints=None):
    """
    Skinning weights (V, 35) for the mesh vertices. Columns of disabled_joints
    are zeroed and each row is renormalized; vertices left without weight are
    bound to the root joint.
    """
    weights = compute_smal_skinning_weights(np.array(mesh.vertices), smal_model, smal_params)
    if apply_smoothing:
        weights = apply_bone_hierarchy_smoothing(weights, build_kinematic_tree(parents))

    if disabled_joints:
        for joint_idx in disabled_joints:
            if 0 <= joint_idx < weights.shape[1]:
                weights[:, joint_idx] = 0.0
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        valid_vertices = weight_sums[:, 0] > 1e-8
        weights[valid_vertices] = weights[valid_vertices] / weight_sums[valid_vertices]
        weights[~valid_vertices, 0] = 1.0
    return weights
