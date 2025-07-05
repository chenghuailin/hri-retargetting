import torch
# def calc_point2seg_dist(P0, P1, P2):
#     s = P2 - P1
#     lmbda = torch.norm((P0 - P1),s) / (torch.norm(s) * torch.norm(s))
#     if 0 <= lmbda and lmbda <= 1:
#         dist = torch.norm(P0-(P1+lmbda * s))
#     else:
#         dist = torch.min(torch.norm(P0-P1),torch.norm(P0-P2))

def calc_point2seg_dist(P0, P1, P2):
    # 向量 s 表示线段方向
    s = P2 - P1
    s_norm_sq = torch.sum(s * s, dim=-1, keepdim=True)  # 避免重复 norm 计算
    
    # 计算 lambda（投影比例因子），clamp 到 [0, 1] 区间
    proj = torch.sum((P0 - P1) * s, dim=-1, keepdim=True) / (s_norm_sq + 1e-12)  # 避免除0
    lmbda = proj.clamp(0.0, 1.0)
    
    # 投影点 = P1 + lambda * s
    proj_point = P1 + lmbda * s
    
    # 最终距离 = 点到投影点的距离
    dist = torch.norm(P0 - proj_point, dim=-1)
    return dist

def calc_seg2seg_dist(P1, P2, Q1, Q2):
    """
    Efficient, branchless computation of distance between segment P1P2 and Q1Q2
    Supports broadcasting across batch dimensions.
    """
    s1 = P2 - P1  # (N, D)
    s2 = Q2 - Q1
    r = P1 - Q1

    M2 = torch.sum(s1 * s1, dim=-1, keepdim=True)  # ||s1||^2
    N2 = torch.sum(s2 * s2, dim=-1, keepdim=True)  # ||s2||^2
    P = torch.sum(s1 * s2, dim=-1, keepdim=True)   # dot(s1, s2)
    r1 = torch.sum(r * s1, dim=-1, keepdim=True)   # dot(r, s1)
    r2 = torch.sum(r * s2, dim=-1, keepdim=True)   # dot(r, s2)

    C = M2 * N2 - P ** 2 + 1e-12  # 防止除以 0
    lambda_1 = (P * r2 - N2 * r1) / C
    lambda_2 = (M2 * r2 - P * r1) / C

    # Clamp 投影点在 [0, 1] 区间
    lambda_1_clamped = lambda_1.clamp(0.0, 1.0)
    lambda_2_clamped = lambda_2.clamp(0.0, 1.0)

    # 最近点坐标
    closest_P = P1 + lambda_1_clamped * s1
    closest_Q = Q1 + lambda_2_clamped * s2

    # 计算最近距离
    dist_direct = torch.norm(closest_P - closest_Q, dim=-1)

    # 针对不在区间 [0, 1] 内的情况，回退为点到线段距离
    # 构造 4 种情况的距离
    dists = torch.stack([
        calc_point2seg_dist(P1, Q1, Q2),
        calc_point2seg_dist(P2, Q1, Q2),
        calc_point2seg_dist(Q1, P1, P2),
        calc_point2seg_dist(Q2, P1, P2),
    ], dim=-1)

    dist_fallback = torch.min(dists, dim=-1).values

    # 判断 λ 是否都在 [0, 1] 内
    in_range = ((lambda_1 >= 0) & (lambda_1 <= 1) &
                (lambda_2 >= 0) & (lambda_2 <= 1)).squeeze(-1)

    # 用 where 选择最终距离（无 if）
    final_dist = torch.where(in_range, dist_direct, dist_fallback)

    return final_dist

# 2025.05.15
# Add the distance calculation between segment and cuboid
# def calc_seg2cuboid_dist(P1,P2, xlim, ylim, zlim):
#     """
#     calculate the diatance from a segment to a cuboid.
#     P1: the relative pos of the segment point from the cuboid coordinate. (batch*3)
#     P2: another endpoint from the cuboid coordinate. (batch * 3)
#     xlim: the range of x coordinate of the cuboid from the cuboid local coordinate. (batch *(xmin,xmax))
#     ylim: the range of y coordinate of the cuboid from the cuboid local coordinate. batch* (ymin,ymax)
#     zlim: the range of z coordinate of the cuboid from the cuboid local coordinate. batch * (zmin,zmax)
    
#     return: min distance from the segment to the cuboid. (batch,)
#     """
#     batch = P1.shape[0]
#     # print("batch: ", batch)
#     # print("xlim: ", xlim)
#     # print("ylim: ", ylim)
#     # print("zlim: ", zlim)
#     # print("P1 shape: ",P1.shape)
#     # 端点到长方体距离
#     def point2cuboid_dist(P):
#         dx = torch.maximum(xlim[...,0] - P[..., 0], P[..., 0] - xlim[...,1])
#         dy = torch.maximum(ylim[...,0] - P[..., 1], P[..., 1] - ylim[...,1])
#         dz = torch.maximum(zlim[...,0] - P[..., 2], P[..., 2] - zlim[...,1])
#         d = torch.stack([dx, dy, dz], dim=-1).clamp(min=0.0)
#         return torch.norm(d, dim=-1)
#     d1 = point2cuboid_dist(P1)
#     d2 = point2cuboid_dist(P2)
#     # 面交点检测及距离
#     s = P2 - P1
#     face_dists = []
#     for i, (low, high) in enumerate((xlim, ylim, zlim)):
#         for side in (low, high):
#             denom = s[:, i]
#             t = (side - P1[:, i]) / (denom + 1e-12)
#             mask = (t >= 0) & (t <= 1)
#             pts = P1 + t.unsqueeze(-1) * s
#             # 检查另两维是否在范围内
#             j, k = [idx for idx in (0, 1, 2) if idx != i]
#             in_plane = mask & (pts[:, j] >= (ylim if i==0 else xlim)[0]) & (pts[:, j] <= (ylim if i==0 else xlim)[1])
#             in_plane = in_plane & (pts[:, k] >= (zlim if {0,1} - {i} else ylim)[0]) & (pts[:, k] <= (zlim if {0,1} - {i} else ylim)[1])
#             d_plane = torch.abs(pts[:, i] - side)
#             # 不满足条件的设为大值
#             d_plane = torch.where(in_plane, d_plane, torch.full_like(d_plane, float('inf')))
#             face_dists.append(d_plane)
#     # 边缘距离
#     corners = torch.tensor([
#         [xlim[0], ylim[0], zlim[0]], [xlim[0], ylim[0], zlim[1]],
#         [xlim[0], ylim[1], zlim[0]], [xlim[0], ylim[1], zlim[1]],
#         [xlim[1], ylim[0], zlim[0]], [xlim[1], ylim[0], zlim[1]],
#         [xlim[1], ylim[1], zlim[0]], [xlim[1], ylim[1], zlim[1]],
#     ], device=P1.device, dtype=P1.dtype)
#     edges_idx = torch.tensor([
#         [0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],
#         [3,7],[4,5],[4,6],[5,7],[6,7]
#     ], device=P1.device)
#     Q1 = corners[edges_idx[:, 0]].unsqueeze(0).expand(batch, -1, 3)
#     Q2 = corners[edges_idx[:, 1]].unsqueeze(0).expand(batch, -1, 3)
#     P1_e = P1.unsqueeze(1).expand(-1, edges_idx.shape[0], 3)
#     P2_e = P2.unsqueeze(1).expand(-1, edges_idx.shape[0], 3)
#     d_edges = calc_seg2seg_dist(P1_e, P2_e, Q1, Q2)
#     d_edges_min = torch.min(d_edges, dim=-1).values
#     # 汇总最小距离
#     all_dists = torch.stack([d1, d2, d_edges_min] + face_dists, dim=-1)
#     return torch.min(all_dists, dim=-1).values

# P1, P2: (B, 3)
# xlim, ylim, zlim: (B, 2)
def calc_seg2cuboid_dist(P1, P2, xlim, ylim, zlim):
    B = P1.shape[0]
    # 端点到长方体距离
    def point2cuboid(P):
        dx = torch.max(xlim[:,0] - P[:,0], P[:,0] - xlim[:,1])
        dy = torch.max(ylim[:,0] - P[:,1], P[:,1] - ylim[:,1])
        dz = torch.max(zlim[:,0] - P[:,2], P[:,2] - zlim[:,1])
        d = torch.stack([dx, dy, dz], dim=-1).clamp(min=0.0)
        return torch.norm(d, dim=-1)

    d1 = point2cuboid(P1)
    d2 = point2cuboid(P2)

    # 面交点检测及距离
    s = P2 - P1
    face_dists = []  # 存 (B,) 张量
    lims = [xlim, ylim, zlim]
    for i in range(3):
        lim = lims[i]
        low, high = lim[:,0], lim[:,1]
        for side in (low, high):
            denom = s[:, i]
            t = (side - P1[:, i]) / (denom + 1e-12)
            mask = (t >= 0) & (t <= 1)
            pts = P1 + t.unsqueeze(-1) * s
            # 检查另两维是否在对应范围内
            j, k = [d for d in range(3) if d != i]
            lim_j = lims[j]
            lim_k = lims[k]
            in_plane = mask & (pts[:, j] >= lim_j[:,0]) & (pts[:, j] <= lim_j[:,1])
            in_plane = in_plane & (pts[:, k] >= lim_k[:,0]) & (pts[:, k] <= lim_k[:,1])
            d_plane = torch.abs(pts[:, i] - side)
            d_plane = torch.where(in_plane, d_plane, torch.full_like(d_plane, float('inf')))
            face_dists.append(d_plane)

    # 边缘距离
    corners = torch.stack([torch.stack([xlim[:,a], ylim[:,b], zlim[:,c]], dim=1)
                            for a in (0,1) for b in (0,1) for c in (0,1)], dim=1)  # (B,8,3)
    edges = torch.tensor([[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]],
                         device=P1.device)
    Q1 = corners[:, edges[:,0], :]  # (B,12,3)
    Q2 = corners[:, edges[:,1], :]
    P1_e = P1.unsqueeze(1).expand(-1, edges.size(0), -1)
    P2_e = P2.unsqueeze(1).expand(-1, edges.size(0), -1)
    d_edges = calc_seg2seg_dist(P1_e, P2_e, Q1, Q2)  # (B,12)
    d_edge_min = torch.min(d_edges, dim=1).values

    all_d = torch.stack([d1, d2, d_edge_min] + face_dists, dim=1)  # (B, 3+6)
    return torch.min(all_d, dim=1).values

# P1 = torch.tensor((0.02,-0.3,0.2)).unsqueeze(0)
# P2 = torch.tensor((0.02,-0.2,0.3)).unsqueeze(0)
# dist = calc_seg2cuboid_dist(P1,P2,(-0.06,0.06),(-0.105,0.105),(0,0.33))
# print(dist)

