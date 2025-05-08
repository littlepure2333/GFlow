import time
import torch
# import pointops

def gen_line_set(xyz1, xyz2, rgb, device):
    N = xyz1.shape[0]

    diff = xyz2 - xyz1
    diff_norm = torch.norm(diff, dim=1, keepdim=True)

    line_set_xyz = []
    line_set_rgb = []
    line_set_scale = []
    point_set_xyz = []
    point_set_rgb = []
    point_set_scale = []

    for i in range(N):
        L = max(2, int(diff_norm[i] * 100))
        for j in range(L):
            t = j / (L-1)  
            increment = xyz1[i] + t * diff[i]
            if j < L-1:
                line_set_xyz.append(increment)
                line_set_rgb.append(rgb[i])
                line_set_scale.append(torch.ones(3, device=device))
            if j == (L-1):
                point_set_xyz.append(increment)
                point_set_rgb.append(rgb[i])
                point_set_scale.append(torch.ones(3, device=device))

    # last N are points
    line_set_xyz = torch.stack(line_set_xyz+point_set_xyz).to(device)
    line_set_rgb = torch.stack(line_set_rgb+point_set_rgb).to(device)
    line_set_scale = torch.stack(line_set_scale+point_set_scale).to(device)
    point_set_xyz = torch.stack(point_set_xyz).to(device)
    point_set_rgb = torch.stack(point_set_rgb).to(device)
    point_set_scale = torch.stack(point_set_scale).to(device)

    return line_set_xyz, line_set_rgb

def construct_list_of_attributes():
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(3):
        l.append('f_dc_{}'.format(i))
    l.append('opacity')
    for i in range(3):
        l.append('scale_{}'.format(i))
    for i in range(4):
        l.append('rot_{}'.format(i))
    return l

# def farthest_point_sampling(xyz, fps_number, device):
#     num_pts = xyz.shape[0]
#     print(num_pts)
#     start = time.time()
#     batch_size = 1
#     offset = torch.cumsum(torch.Tensor([num_pts] * batch_size).to(device), dim=0).to(torch.long)
#     new_offset = torch.cumsum(torch.Tensor([fps_number] * batch_size).to(device), dim=0).to(torch.long)

#     result = pointops.farthest_point_sampling(xyz, offset, new_offset)
#     print(result.shape)
#     end = time.time()
#     print("FPS time:", end-start)

#     return result