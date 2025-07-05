# 2025.04.21 HIT-xiaowangzi
# Test the segment dist calculation in two ways:
# (1) Speed
# (2) Accuracy

import time
import torch
from segment_dist_lib import calc_point2seg_dist,calc_seg2seg_dist,calc_seg2cuboid_dist

def test_seg2seg():
    # 构造大规模随机线段数据
    N = 100_000
    P1 = torch.rand(N, 3, device='cuda')
    P2 = torch.rand(N, 3, device='cuda')
    Q1 = torch.rand(N, 3, device='cuda')
    Q2 = torch.rand(N, 3, device='cuda')

    # 预热一次避免 cold start
    _ = calc_seg2seg_dist(P1, P2, Q1, Q2)

    # CUDA 时间测量
    torch.cuda.synchronize()
    start = time.time()

    _ = calc_seg2seg_dist(P1, P2, Q1, Q2)

    torch.cuda.synchronize()
    end = time.time()

    print(f"Time for {N} segment-to-segment distances on GPU: {end - start:.4f} seconds")
    
def test_seg2cuboid():
        # 构造大规模随机线段数据
    N = 100_000
    P1 = torch.rand(N, 3, device='cuda')
    P2 = torch.rand(N, 3, device='cuda')
    xlim = (-0.06,0.06)
    ylim = (-0.105,0.105)
    zlim = (0,0.33)

    # 预热一次避免 cold start
    _ = calc_seg2cuboid_dist(P1,P2,xlim,ylim,zlim)

    # CUDA 时间测量
    torch.cuda.synchronize()
    start = time.time()

    _ = calc_seg2cuboid_dist(P1,P2,xlim,ylim,zlim)

    torch.cuda.synchronize()
    end = time.time()

    print(f"Time for {N} segment-to-cuboid distances on GPU: {end - start:.4f} seconds")
    
if __name__ == "__main__":
    test_seg2cuboid()