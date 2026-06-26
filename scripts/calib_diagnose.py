# -*- coding: utf-8 -*-
"""
手眼标定数据诊断脚本（在本机运行，使用干净文件）
================================================
用途：当 calib_solve.py 解出的标定误差异常大时，定位根因。
回答三个问题：
  1. 图像/位姿文件是否完整（排除文件损坏）
  2. 哪种欧拉角约定最匹配你的机械臂（排除约定配错）
  3. 旋转一致性是否达标 / 坏帧集中在哪几组（排除个别坏数据）

运行： python calib_diagnose.py
"""
import cv2, numpy as np, glob, os, math, itertools

CHECKERBOARD = (8, 11)
SQUARE_SIZE  = 10.0
IMAGE_DIR    = "runs/calib_data/images/"
POSE_DIR     = "runs/calib_data/poses/"
# 内参兜底值（与 calib_solve.py 保持一致；若有 D435 会更准，这里只为诊断够用）
cam  = np.array([[606.21, 0, 317.80], [0, 606.35, 249.82], [0, 0, 1]], dtype=np.float64)
dist = np.zeros((5, 1))


def check_files():
    print("=" * 60)
    print("【第1步】文件完整性检查")
    print("=" * 60)
    imgs = sorted(glob.glob(IMAGE_DIR + "*.jpg"),
                  key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    bad_img, bad_pose = [], []
    for ip in imgs:
        idx = os.path.splitext(os.path.basename(ip))[0]
        with open(ip, "rb") as f:
            data = f.read()
        # 合法 JPEG 必须以 FFD9 结尾
        if len(data) < 4 or data[-2:] != b"\xff\xd9":
            bad_img.append(idx)
        pp = os.path.join(POSE_DIR, idx + ".txt")
        if os.path.exists(pp):
            with open(pp, "rb") as f:
                praw = f.read()
            if b"\x00" in praw:
                bad_pose.append(idx)
    print(f"图像总数: {len(imgs)}")
    print(f"  截断/损坏的图像(结尾非FFD9): {len(bad_img)} 张" +
          (f"  -> {bad_img}" if bad_img else "  (全部完整 ✅)"))
    print(f"  含空字节的位姿文件: {len(bad_pose)} 个" +
          (f"  -> {bad_pose}" if bad_pose else "  (全部干净 ✅)"))
    return bad_img, bad_pose


def ang(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))

def Rx(a): a = math.radians(a); return np.array([[1,0,0],[0,math.cos(a),-math.sin(a)],[0,math.sin(a),math.cos(a)]])
def Ry(a): a = math.radians(a); return np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])
def Rz(a): a = math.radians(a); return np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])
F = {"X": Rx, "Y": Ry, "Z": Rz}


def load_data():
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE
    poses, Rb, ids, reproj = [], [], [], []
    imgs = sorted(glob.glob(IMAGE_DIR + "*.jpg"),
                  key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    for ip in imgs:
        idx = os.path.splitext(os.path.basename(ip))[0]
        img = cv2.imread(ip)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, c = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if not ret:
            print(f"  [漏检] 第{idx}组未找到棋盘格，跳过")
            continue
        c = cv2.cornerSubPix(gray, c, (11, 11), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        ok, rv, tv = cv2.solvePnP(objp, c, cam, dist)
        proj, _ = cv2.projectPoints(objp, rv, tv, cam, dist)
        err = float(np.linalg.norm(proj.reshape(-1, 2) - c.reshape(-1, 2), axis=1).mean())
        R, _ = cv2.Rodrigues(rv)
        # 防御性读取：去掉可能的空字节
        raw = open(os.path.join(POSE_DIR, idx + ".txt"), "rb").read().replace(b"\x00", b"")
        d = raw.decode("ascii", "ignore").replace(",", " ").split()
        poses.append([float(x) for x in d[:6]])
        Rb.append(R); ids.append(idx); reproj.append(err)
    return poses, Rb, ids, reproj


def main():
    bad_img, bad_pose = check_files()

    poses, Rb, ids, reproj = load_data()
    N = len(Rb)
    print()
    print("=" * 60)
    print("【第2步】PnP 重投影误差（图像/角点质量）")
    print("=" * 60)
    reproj = np.array(reproj)
    print(f"有效样本: {N} 组")
    print(f"重投影误差: 平均 {reproj.mean():.3f}px  最大 {reproj.max():.3f}px  (正常应<0.5px)")
    worst = np.argsort(reproj)[::-1][:5]
    print("最差5帧: " + ", ".join(f"#{ids[i]}({reproj[i]:.2f}px)" for i in worst))

    print()
    print("=" * 60)
    print("【第3步】欧拉角约定搜索（臂相对旋转角 应等于 相机相对旋转角）")
    print("=" * 60)
    camang = [ang(Rb[i + 1] @ Rb[i].T) for i in range(N - 1)]
    orders = ["".join(p) for p in itertools.permutations("XYZ")]
    signs  = list(itertools.product([1, -1], repeat=3))
    results = []
    for order in orders:
        for s in signs:
            Rg = [F[order[0]](s[0] * p[3 + "XYZ".index(order[0])]) @
                  F[order[1]](s[1] * p[3 + "XYZ".index(order[1])]) @
                  F[order[2]](s[2] * p[3 + "XYZ".index(order[2])]) for p in poses]
            armang = [ang(Rg[i + 1] @ Rg[i].T) for i in range(N - 1)]
            diff = np.abs(np.array(armang) - np.array(camang))
            results.append((diff.mean(), order, s))
    results.sort()
    print(f"{'欧拉序':<8}{'符号':<16}{'平均角差(°)':>12}")
    for d, order, s in results[:6]:
        print(f"{order:<8}{str(s):<16}{d:>12.2f}")
    best_d = results[0][0]
    print()
    if best_d < 1.5:
        print(f"✅ 最优约定角差 {best_d:.2f}° < 1.5°：数据是好的，只需把 calib_solve.py")
        print(f"   的欧拉约定改成 {results[0][1]} / 符号{results[0][2]} 即可。")
    else:
        print(f"⚠️  即使最优约定，角差仍有 {best_d:.2f}°（应<1.5°）。")
        print(f"   说明不是欧拉约定问题，而是【位姿与图像不对应】或【数据采集本身有误】。")
        print(f"   重点排查：拍照时机械臂是否真的停稳；GetPose 读到的是否就是当前帧的位姿；")
        print(f"   标定板是否全程固定不动。")

    print()
    print("=" * 60)
    print("【第4步】逐帧一致性（用最优约定，找离群坏帧）")
    print("=" * 60)
    d0, order0, s0 = results[0]
    Rg = [F[order0[0]](s0[0] * p[3 + "XYZ".index(order0[0])]) @
          F[order0[1]](s0[1] * p[3 + "XYZ".index(order0[1])]) @
          F[order0[2]](s0[2] * p[3 + "XYZ".index(order0[2])]) for p in poses]
    armang = [ang(Rg[i + 1] @ Rg[i].T) for i in range(N - 1)]
    bad = 0
    for i in range(N - 1):
        diff = abs(armang[i] - camang[i])
        if diff > 3:
            bad += 1
            if bad <= 15:
                print(f"  #{ids[i]}->#{ids[i+1]}: 臂{armang[i]:5.1f}°  相机{camang[i]:5.1f}°  差{diff:5.1f}°")
    print(f"\n角差>3°的相邻对: {bad}/{N-1}")


if __name__ == "__main__":
    main()
