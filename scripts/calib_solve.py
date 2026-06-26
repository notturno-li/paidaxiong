import cv2  # 负责核心视觉算法 (角点提取、PnP求解、Tsai-Lenz标定)
import numpy as np  # 负责高精度矩阵运算，机器人的世界本质上就是一堆多维数组
import os, glob, math  # 负责文件路径读取和基础数学(弧度转换)

# ==============================================================================
# 模块一：全局物理参数锁死区 (零误差的前提)
# ==============================================================================
# 【物理原理】：OpenCV 标定算法寻找的是黑白方块交汇的"十字内角点"，而不是方块本身。
# 你打印的是 8行11列 的方块，因此内角点必须是 (11, 8)。(OpenCV 习惯宽X在前，高Y在后)
CHECKERBOARD = (8, 11)

# 【绝对尺度】：单个方格的物理真实边长，单位：毫米(mm)。
# 视觉本身是没有"大小"概念的，全靠这个数值给系统注入真实的物理世界尺度。
SQUARE_SIZE = 10.0

IMAGE_DIR = "runs/calib_data/images/"
POSE_DIR = "runs/calib_data/poses/"

# 【关键】：内参分辨率必须与采集图像分辨率完全一致！
# 采集脚本(calib_collect.py / calib_collect_auto.py)用的是 1280x720，
# 这里若用 640x480 读内参，fx/fy/主点全错一倍，PnP 会系统性失真 → 标定误差爆炸。
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# 【相机内参矩阵】：自动从 D435 硬件读取，保证与采集时一致
# 如果相机未连接，则使用下方兜底值(1280x720 下的典型值)
try:
    import pyrealsense2 as rs
    _pipeline = rs.pipeline()
    _cfg = rs.config()
    _cfg.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, 30)
    _profile = _pipeline.start(_cfg)
    _intr = _profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    camera_matrix = np.array([[_intr.fx, 0.0, _intr.ppx],
                              [0.0, _intr.fy, _intr.ppy],
                              [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_coeffs = np.array([[c] for c in _intr.coeffs], dtype=np.float64)
    _pipeline.stop()
    print(f"[内参] 已从 D435 硬件读取 ({CAMERA_WIDTH}x{CAMERA_HEIGHT}): fx={_intr.fx:.2f} fy={_intr.fy:.2f} cx={_intr.ppx:.2f} cy={_intr.ppy:.2f}")
except Exception as _e:
    print(f"[内参] D435 未连接，使用 1280x720 兜底值: {_e}")
    camera_matrix = np.array([[1212.0, 0.0, 636.0],
                              [0.0, 1212.0, 500.0],
                              [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)


# ==============================================================================
# 模块二：空间几何转换引擎 (示教器语言 -> 数学矩阵)
# ==============================================================================
def txt_to_matrix(txt_path):
    """
    【算法名称】：Z-Y-X 欧拉角转旋转矩阵 (Euler to Rotation Matrix)

    【为什么要有这个函数？】
    机械臂示教器上显示的是人类看得懂的欧拉角 (比如倾斜了 15 度：Rx=15)。
    但是，计算机底层的 Tsai-Lenz 算法不认识度数，它只认识 3x3 的旋转矩阵。
    此函数就是连接机械臂物理世界和 OpenCV 数学世界的"翻译官"。
    """
    with open(txt_path, 'r') as f:
        data = f.read().replace(',', ' ').split()

    # 1. 提取平移向量 Translation (X, Y, Z)，单位：毫米
    t_vec = np.array([[float(data[0])], [float(data[1])], [float(data[2])]])

    # 2. 将欧拉角从"度(Degree)"转换为"弧度(Radian)" (计算机三角函数只认弧度)
    rx, ry, rz = map(math.radians, [float(data[3]), float(data[4]), float(data[5])])

    # 3. 构建三个基础旋转矩阵
    # 绕 X 轴旋转矩阵 (Roll)
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(rx), -math.sin(rx)],
                    [0, math.sin(rx), math.cos(rx)]])

    # 绕 Y 轴旋转矩阵 (Pitch)
    R_y = np.array([[math.cos(ry), 0, math.sin(ry)],
                    [0, 1, 0],
                    [-math.sin(ry), 0, math.cos(ry)]])

    # 绕 Z 轴旋转矩阵 (Yaw)
    R_z = np.array([[math.cos(rz), -math.sin(rz), 0],
                    [math.sin(rz), math.cos(rz), 0],
                    [0, 0, 1]])

    # 4. 矩阵相乘 (核心避坑点)
    # 【工业标准】：矩阵乘法不满足交换律！主流机械臂(如Dobot/KUKA)默认采用 Z-Y-X 的内在旋转顺序。
    # 数学表达为：R = R_z * R_y * R_x (代码中用 @ 符号表示矩阵点乘)
    R_matrix = R_z @ R_y @ R_x

    return R_matrix, t_vec


# ==============================================================================
# 模块三：数据装载与 A、B 矩阵构建 (AX = XB 的前置准备)
# ==============================================================================
# 准备空列表，用来装方程 AX=XB 中的 A 和 B
R_gripper2base, t_gripper2base = [], []  # 装载矩阵 A (机械臂末端在基座下的姿态)
R_target2cam, t_target2cam = [], []  # 装载矩阵 B (标定板在相机下的姿态)

img_files = sorted(glob.glob(IMAGE_DIR + "*.jpg"))
print(f"🔄 [系统流] 正在读取 {len(img_files)} 组标定数据，准备构建空间方程...")

for img_path in img_files:
    idx = os.path.splitext(os.path.basename(img_path))[0]
    txt_path = os.path.join(POSE_DIR, f"{idx}.txt")

    # ---------------------------------------------------------
    # 步骤 1：利用视觉计算矩阵 B (标定板相对于相机的空间位姿)
    # ---------------------------------------------------------
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 寻找标定板特征点
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if not ret: continue

    # 【亚像素优化】：将角点精度从 1 个像素，强行逼近到 0.001 个像素，直接决定了抓取的毫米级精度！
    corners_subpix = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )

    # 构建标定板在真实世界的三维坐标系 (假设标定板左上角第一个点为 X:0, Y:0, Z:0)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

    # 【PnP 求解算法 (Perspective-n-Point)】
    # 原理：已知 3D 物理点(objp)和对应的 2D 像素点(corners_subpix)，结合相机内参，
    # 反推出相机拍摄这张照片时，相机相对于标定板在空间中的 旋转向量(rvec) 和 平移向量(tvec)。
    _, rvec, tvec = cv2.solvePnP(objp, corners_subpix, camera_matrix, dist_coeffs)

    # 将旋转向量(3x1)转换为计算所需的旋转矩阵(3x3)
    R_cam, _ = cv2.Rodrigues(rvec)

    R_target2cam.append(R_cam)
    t_target2cam.append(tvec)

    # ---------------------------------------------------------
    # 步骤 2：利用示教器数据计算矩阵 A (法兰盘相对于基座的空间位姿)
    # ---------------------------------------------------------
    R_grip, t_grip = txt_to_matrix(txt_path)
    R_gripper2base.append(R_grip)
    t_gripper2base.append(t_grip)

# ==============================================================================
# 模块四：终极矩阵解算 (寻找那个被螺丝拧死的神秘 X 矩阵)
# ==============================================================================
print(f"⚙️ [数学引擎] 数据就绪。正在启动 Tsai-Lenz 算法破解 AX=XB 方程组...")

R_cam2grip, t_cam2grip = cv2.calibrateHandEye(
    R_gripper2base, t_gripper2base,
    R_target2cam, t_target2cam,
    method=cv2.CALIB_HAND_EYE_TSAI  # 指定使用 Tsai-Lenz 算法
)

# ==============================================================================
# 模块五：组装与下发 4x4 齐次变换矩阵
# ==============================================================================
T_matrix = np.eye(4)  # 先生成一个 4x4 的对角线为1的单位矩阵
T_matrix[:3, :3] = R_cam2grip  # 把求出来的 3x3 旋转矩阵塞进左上角
T_matrix[:3, 3] = t_cam2grip.flatten()  # 把求出来的平移塞进右上角

print("\n Hand-eye calibration done!")
print("👇 终极 4x4 手眼变换矩阵 (T_cam_to_gripper) 如下：")
print(np.round(T_matrix, 4))

# ==============================================================================
# 模块五点五：标定误差评估 (闭环一致性验证)
# ==============================================================================
# 【原理】：标定板被螺丝拧死在桌面上，它在机械臂【基座坐标系】下的位姿是恒定不变的。
# 对每一组采样，理论上：T_target2base = T_gripper2base @ X @ T_target2cam 都应当完全相等。
# 因此把每组算出的 T_target2base 收集起来，统计它们之间的离散程度：
#   - 平移离散越小 => 手眼平移标得越准 (单位 mm)
#   - 旋转离散越小 => 手眼旋转标得越准 (单位 度)
def _to_homo(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = np.asarray(t, dtype=np.float64).flatten()
    return M

_X = _to_homo(R_cam2grip, t_cam2grip)
_t2b = [_to_homo(Rg, tg) @ _X @ _to_homo(Rc, tc)
        for Rg, tg, Rc, tc in zip(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)]

# 平移误差：各组标定板原点相对于"均值位置"的偏差
_trans = np.array([T[:3, 3] for T in _t2b])
_trans_err = np.linalg.norm(_trans - _trans.mean(axis=0), axis=1)

# 旋转误差：先用 SVD 求出平均旋转矩阵，再算各组相对均值的角度偏差
_R_sum = sum(T[:3, :3] for T in _t2b)
_U, _, _Vt = np.linalg.svd(_R_sum)
_R_mean = _U @ _Vt
if np.linalg.det(_R_mean) < 0:
    _U[:, -1] *= -1
    _R_mean = _U @ _Vt
_rot_err = []
for T in _t2b:
    _ang = (np.trace(_R_mean.T @ T[:3, :3]) - 1) / 2
    _rot_err.append(math.degrees(math.acos(max(-1.0, min(1.0, _ang)))))
_rot_err = np.array(_rot_err)

print("\n📏 [误差评估] 标定板在基座系下应恒定，离散度即标定误差：")
print(f"   有效样本           : {len(_t2b)} 组")
print(f"   平移误差 RMS       : {np.sqrt(np.mean(_trans_err ** 2)):.3f} mm")
print(f"   平移误差 最大      : {_trans_err.max():.3f} mm")
print(f"   旋转误差 RMS       : {np.sqrt(np.mean(_rot_err ** 2)):.4f} °")
print(f"   旋转误差 最大      : {_rot_err.max():.4f} °")
if np.sqrt(np.mean(_trans_err ** 2)) > 5.0:
    print("   ⚠️ 平移误差偏大(>5mm)，建议检查采样姿态多样性或重新采集数据")

# ==============================================================================
# 模块六：导出标定结果
# ==============================================================================
# 导出供 GUI 控制总线直接调用
output_file = 'hand_eye_result.yaml'
fs = cv2.FileStorage(output_file, cv2.FILE_STORAGE_WRITE)
fs.write("Transformation_Matrix", T_matrix)
fs.release()
print(f"Hand-eye matrix saved to: {output_file}")

# Auto-update competition.yaml
from pathlib import Path as _P
import yaml as _y
_cp = _P(__file__).resolve().parent.parent / 'configs' / 'competition.yaml'
if _cp.exists():
    try:
        _cfg = _y.safe_load(_cp.read_text(encoding='utf-8')) or {}
        _cfg.setdefault('calibration', {})

        # 只让矩阵的每一行用流式(保留方括号、紧凑成一行)，其余配置仍是块式(无花括号)
        class _FlowList(list):
            pass
        _y.add_representer(_FlowList,
            lambda dumper, data: dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True))

        _cfg['calibration']['transform_camera_to_gripper'] = [_FlowList(row) for row in T_matrix.tolist()]
        _cp.write_text(_y.dump(_cfg, allow_unicode=True, sort_keys=False, default_flow_style=False), encoding='utf-8')
        print(f"Hand-eye matrix written to {_cp}")
    except Exception as e:
        print(f"Failed to update competition.yaml: {e}")
else:
    print(f"Not found: {_cp}")
