"""Fit SMAL shape and pose to a generated T-pose mesh."""

import torch
from pytorch3d.loss import chamfer_distance, mesh_edge_loss
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from tqdm import tqdm


def find_bottom_quadrant_anchor_vertices(mesh_vertices, num_extrema=100):
    """
    Find one anchor vertex per XZ quadrant of the bottom 10% of the mesh: the
    vertex closest to the mean of the quadrant's lowest num_extrema vertices.
    """
    min_y = torch.min(mesh_vertices[:, 1])
    y_range = torch.max(mesh_vertices[:, 1]) - min_y
    bottom_mask = mesh_vertices[:, 1] <= min_y + 0.1 * y_range
    bottom_vertices = mesh_vertices[bottom_mask]
    bottom_indices = torch.where(bottom_mask)[0]

    bottom_xz = bottom_vertices[:, [0, 2]]
    centroid_xz = torch.mean(bottom_xz, dim=0)
    quadrant_masks = [
        (bottom_xz[:, 0] >= centroid_xz[0]) & (bottom_xz[:, 1] >= centroid_xz[1]),
        (bottom_xz[:, 0] < centroid_xz[0]) & (bottom_xz[:, 1] >= centroid_xz[1]),
        (bottom_xz[:, 0] < centroid_xz[0]) & (bottom_xz[:, 1] < centroid_xz[1]),
        (bottom_xz[:, 0] >= centroid_xz[0]) & (bottom_xz[:, 1] < centroid_xz[1]),
    ]

    anchor_vertex_indices = []
    for mask in quadrant_masks:
        quadrant_verts = bottom_vertices[mask]
        quadrant_global_indices = bottom_indices[mask]
        k = min(num_extrema, quadrant_verts.shape[0])
        _, bottom_k_indices = torch.topk(quadrant_verts[:, 1], k=k, largest=False)
        mean_point = torch.mean(quadrant_verts[bottom_k_indices], dim=0)
        closest_local_idx = torch.argmin(torch.norm(quadrant_verts - mean_point, dim=1))
        anchor_vertex_indices.append(quadrant_global_indices[closest_local_idx].item())
    return anchor_vertex_indices


def chamfer_distance_loss(smal_vertices, target_vertices, smal_faces, target_faces, num_samples=1000):
    """Chamfer distance between points sampled uniformly on the two mesh surfaces."""
    smal_mesh = Meshes(verts=smal_vertices.unsqueeze(0), faces=smal_faces.unsqueeze(0))
    target_mesh = Meshes(verts=target_vertices.unsqueeze(0), faces=target_faces.unsqueeze(0))
    smal_samples = sample_points_from_meshes(smal_mesh, num_samples)
    target_samples = sample_points_from_meshes(target_mesh, num_samples)
    chamfer_dist, _ = chamfer_distance(smal_samples, target_samples)
    return chamfer_dist


def mesh_edge_smoothness_loss(smal_vertices, smal_faces):
    """PyTorch3D mesh edge loss on the SMAL mesh."""
    return mesh_edge_loss(Meshes(verts=smal_vertices.unsqueeze(0), faces=smal_faces.unsqueeze(0)))


def fit_smal_to_mesh(
        mesh, smal_model, smal_params_path=None,
        chamfer_loss_weight=1.0, chamfer_samples=1000, mesh_edge_loss_weight=0.01, foot_anchor_weight=0.1,
        iter=3000, lr=1e-3, lr_shape=None, lr_pose=None, lr_global_orient=None, lr_transl=None,
):
    """
    Fit SMAL betas, pose, global orientation and translation to a T-pose mesh
    (Y up, facing +Z) with Chamfer, edge and foot-anchor losses.

    Returns a dict with betas (1, 41), pose (1, 34, 3, 3), global_orient
    (1, 1, 3, 3), transl (1, 3) and joints (1, 35, 3).
    """
    device = smal_model.J_regressor.device
    mesh_vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    target_faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)
    if foot_anchor_weight > 0:
        mesh_anchor_points = mesh_vertices[find_bottom_quadrant_anchor_vertices(mesh_vertices)]

    batch_size = 1
    betas = torch.nn.Parameter(torch.zeros(batch_size, 41))
    pose = torch.nn.Parameter(torch.tensor([1., 0., 0., 0., 1., 0.])[None].repeat(34, 1))
    # Rotate the SMAL rest pose into the mesh frame.
    rot_x_neg90 = torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
    rot_y_neg90 = torch.tensor([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
    global_orient = torch.nn.Parameter(matrix_to_rotation_6d(rot_y_neg90 @ rot_x_neg90))

    # Start with the SMAL mesh centered at the origin.
    with torch.no_grad():
        initial_smal_output = smal_model(
            betas=torch.zeros(batch_size, 41).to(device),
            pose=rotation_6d_to_matrix(pose)[None].to(device),
            global_orient=rotation_6d_to_matrix(global_orient)[None, None].to(device),
            transl=torch.zeros(batch_size, 3).to(device),
            pose2rot=False,
        )
        initial_smal_vertices = initial_smal_output.vertices.squeeze(0)
        initial_transl = -torch.mean(initial_smal_vertices, dim=0)
    transl = torch.nn.Parameter(initial_transl.unsqueeze(0).repeat(batch_size, 1))

    optimizer = torch.optim.Adam([
        {"params": [betas], "lr": lr_shape if lr_shape is not None else lr},
        {"params": [pose], "lr": lr_pose if lr_pose is not None else lr},
        {"params": [global_orient], "lr": lr_global_orient if lr_global_orient is not None else lr},
        {"params": [transl], "lr": lr_transl if lr_transl is not None else lr},
    ])
    smal_faces = smal_model.faces.clone().detach().to(dtype=torch.long, device=device)
    smal_anchor_indices = find_bottom_quadrant_anchor_vertices(initial_smal_vertices)

    for _ in tqdm(range(iter), desc="Fitting SMAL to T-pose mesh"):
        optimizer.zero_grad()
        output = smal_model(
            betas=betas.to(device),
            pose=rotation_6d_to_matrix(pose)[None].to(device),
            global_orient=rotation_6d_to_matrix(global_orient)[None, None].to(device),
            transl=transl.to(device),
            pose2rot=False,
        )
        smal_vertices = output.vertices.squeeze(0)
        total_loss = torch.tensor(0.0, device=device)
        if chamfer_loss_weight > 0:
            total_loss = total_loss + chamfer_loss_weight * chamfer_distance_loss(
                smal_vertices, mesh_vertices, smal_faces, target_faces, num_samples=chamfer_samples)
        if mesh_edge_loss_weight > 0:
            total_loss = total_loss + mesh_edge_loss_weight * mesh_edge_smoothness_loss(smal_vertices, smal_faces)
        if foot_anchor_weight > 0:
            foot_anchor_loss = torch.mean(torch.norm(mesh_anchor_points - smal_vertices[smal_anchor_indices], dim=1))
            total_loss = total_loss + foot_anchor_weight * foot_anchor_loss
        total_loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_output = smal_model(
            betas=betas.to(device),
            pose=rotation_6d_to_matrix(pose)[None].to(device),
            global_orient=rotation_6d_to_matrix(global_orient)[None, None].to(device),
            transl=transl.to(device),
            pose2rot=False,
        )
    smal_params = {
        "betas": betas.detach().cpu(),
        "pose": rotation_6d_to_matrix(pose)[None].detach().cpu(),
        "global_orient": rotation_6d_to_matrix(global_orient)[None, None].detach().cpu(),
        "transl": transl.detach().cpu(),
        "joints": final_output.joints.detach().cpu(),
    }
    if smal_params_path is not None:
        torch.save(smal_params, smal_params_path)
    return smal_params
