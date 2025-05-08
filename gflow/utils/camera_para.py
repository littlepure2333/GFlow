def extract_camera_parameters(intrinsic_matrix, extrinsic_matrix, W, H, img_name="00001"):
    # Extract focal lengths and principal point from the intrinsic matrix
    [fx, fy, cx, cy] = intrinsic_matrix.detach().cpu().numpy().tolist()

    R = extrinsic_matrix[:3, :3]  # 旋转矩阵部分
    t = extrinsic_matrix[:3, 3]   # 平移向量部分

    # 计算相机在世界坐标系中的位置
    # 通过求逆矩阵的方法，位置为 -R.T @ t
    camera_position = -R.T @ t

    # 计算相机在世界坐标系中的旋转
    # 旋转矩阵在世界坐标系中为 R.T
    camera_rotation = R.T

    # Return all extracted parameters
    return [{
        "id": 0,
        "img_name": img_name,
        "width": W,
        "height": H,
        "position": camera_position.tolist(),
        "rotation": camera_rotation.tolist(),
        "fx": fx,
        "fy": fy
    }]