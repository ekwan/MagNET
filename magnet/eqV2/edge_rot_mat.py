import torch


def init_edge_rot_mat(edge_distance_vec):
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

    # Two atoms at (nearly) the same position cannot define a local frame: the normalization below
    # would divide by ~zero and silently produce NaN or a garbage frame that propagates through the
    # whole forward pass. Refuse the input instead. This is reachable from raw geometries, e.g.
    # overlapping atoms in an unrelaxed explicit-solvent snapshot.
    min_distance = torch.min(edge_vec_0_distance)
    if min_distance < 0.0001:
        raise ValueError(
            f"overlapping atoms: smallest interatomic distance {float(min_distance):.2e} is too "
            "small to define a local frame; check the input geometry for coincident atoms"
        )

    norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

    edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
    edge_vec_2 = edge_vec_2 / (
        torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1)
    )
    # Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
    # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
    edge_vec_2b = edge_vec_2.clone()
    edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
    edge_vec_2b[:, 1] = edge_vec_2[:, 0]
    edge_vec_2c = edge_vec_2.clone()
    edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
    edge_vec_2c[:, 2] = edge_vec_2[:, 1]
    vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(
        -1, 1
    )
    vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(
        -1, 1
    )

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(
        torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2
    )
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(
        torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2
    )

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
    # Check the vectors aren't aligned
    assert torch.max(vec_dot) < 0.99

    norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
    norm_z = norm_z / (
        torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True))
    )
    norm_z = norm_z / (
        torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1)
    )
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / (
        torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True))
    )

    # Construct the 3D rotation matrix
    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)

    edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

    return edge_rot_mat.detach()