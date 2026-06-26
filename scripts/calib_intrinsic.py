"""
Camera Intrinsic Calibration (Zhang's checkerboard method)
Usage:
  1. Run this script; camera preview will appear
  2. Move the checkerboard in front of the camera at various angles/distances/tilts
  3. When green grid appears (board detected), press S to save
  4. Collect 20+ images covering different regions of the frame
  5. Press C to compute, or Q to quit without computing
  6. Results are merged into configs/competition.yaml and runs/intrinsic/intrinsic.yaml
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import cv2, numpy as np, yaml
import pyrealsense2 as rs

CHECKERBOARD  = (8, 11)   # inner corners (cols-1, rows-1); competition board 10mm grid
SQUARE_SIZE   = 10.0      # mm per square
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CAMERA_FPS    = 30
SAVE_DIR      = Path("runs/intrinsic/images")
OUTPUT_DIR    = Path("runs/intrinsic")
OUTPUT_YAML   = OUTPUT_DIR / "intrinsic.yaml"
CONFIG_YAML   = Path("configs/competition.yaml")
SOLVE_SCRIPT  = Path("scripts/calib_solve.py")


def calibrate(img_paths):
    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE
    objpoints, imgpoints, img_size = [], [], None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    print(f"\nComputing with {len(img_paths)} images...")
    for path in img_paths:
        img  = cv2.imread(str(path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if not ret:
            print(f"  [skip] {path.name}")
            continue
        objpoints.append(objp)
        imgpoints.append(cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), crit))
    if len(objpoints) < 6:
        print(f"Only {len(objpoints)} valid images (need >= 6).")
        return None
    rms, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    print(f"Done! Valid:{len(objpoints)}  RMS={rms:.4f} px")
    print("  [OK]" if rms < 0.5 else ("  [!!]" if rms < 1.0 else "  [XX] too large"))
    return K, dist, rms


def save_results(K, dist, rms):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fx, fy = float(K[0,0]), float(K[1,1])
    cx, cy = float(K[0,2]), float(K[1,2])
    dl = dist.flatten().tolist()
    OUTPUT_YAML.write_text(yaml.dump({
        "rms_error": round(rms,6), "image_size": [CAMERA_WIDTH, CAMERA_HEIGHT],
        "camera_matrix": {"fx":fx,"fy":fy,"cx":cx,"cy":cy},
        "dist_coeffs": [round(v,8) for v in dl],
    }, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"\nSaved -> {OUTPUT_YAML}")
    print(f"  fx={fx:.3f}  fy={fy:.3f}  cx={cx:.3f}  cy={cy:.3f}")
    print(f"  dist={[round(v,5) for v in dl]}")
    if CONFIG_YAML.exists():
        try:
            cfg = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8")) or {}
            cfg.setdefault("calibration", {})
            cfg["calibration"]["camera_matrix"] = {"fx":fx,"fy":fy,"cx":cx,"cy":cy}
            cfg["calibration"]["dist_coeffs"] = [round(v,8) for v in dl]
            CONFIG_YAML.write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False,
                                             default_flow_style=False), encoding="utf-8")
            print(f"Merged -> {CONFIG_YAML}")
        except Exception as e:
            print(f"Failed to update competition.yaml: {e}")
    if SOLVE_SCRIPT.exists():
        src = SOLVE_SCRIPT.read_text(encoding="utf-8")
        new_K = (f"camera_matrix = np.array([[{fx:.4f}, 0.0, {cx:.4f}],\n"
                 f"                          [0.0, {fy:.4f}, {cy:.4f}],\n"
                 f"                          [0.0, 0.0, 1.0]], dtype=np.float64)")
        new_d = (f"dist_coeffs = np.array([[{dl[0]:.8f}],\n"
                 f"                        [{dl[1]:.8f}],\n"
                 f"                        [{dl[2]:.8f}],\n"
                 f"                        [{dl[3]:.8f}],\n"
                 f"                        [{dl[4]:.8f}]], dtype=np.float64)")
        src = re.sub(r"camera_matrix\s*=\s*np\.array\(\[.*?\],\s*dtype=np\.float64\)", new_K, src, flags=re.DOTALL)
        src = re.sub(r"dist_coeffs\s*=\s*np\.(zeros|array)\([^)]*\)(\s*#[^\n]*)?", new_d, src)
        SOLVE_SCRIPT.write_text(src, encoding="utf-8")
        print(f"Patched -> {SOLVE_SCRIPT}")


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)
    pipeline.start(cfg)
    img_count = 1
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    print("Move checkerboard at varied angles.  S=save  C=compute  Q=quit\n")
    try:
        while True:
            color_frame = pipeline.wait_for_frames().get_color_frame()
            if not color_frame:
                continue
            img  = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            disp = img.copy()
            saved = img_count - 1
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                cs = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), crit)
                cv2.drawChessboardCorners(disp, CHECKERBOARD, cs, ret)
                cv2.putText(disp, f"Captured:{saved}  Press S to save #{img_count}",
                            (30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            else:
                cv2.putText(disp, f"Captured:{saved}  Searching board...",
                            (30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,165,255), 2)
            cv2.putText(disp, "S=capture  C=calculate  Q=quit",
                        (30, CAMERA_HEIGHT-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
            cv2.imshow("Intrinsic Calibration - S:capture C:calc Q:quit", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                paths = sorted(SAVE_DIR.glob("*.jpg"))
                if not paths:
                    print("No images yet.")
                    continue
                res = calibrate(paths)
                if res:
                    K, dist, rms = res
                    print("\n" + "="*55)
                    print(f"  RMS = {rms:.4f} px")
                    print(f"  fx={K[0,0]:.3f}  fy={K[1,1]:.3f}  cx={K[0,2]:.3f}  cy={K[1,2]:.3f}")
                    print(f"  dist={[round(v,5) for v in dist.flatten().tolist()]}")
                    print("="*55)
                    if input("Write to competition.yaml and calib_solve.py? [y/N] ").strip().lower() == 'y':
                        save_results(K, dist, rms)
                    else:
                        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                        OUTPUT_YAML.write_text(yaml.dump({
                            "rms_error": round(rms,6), "image_size": [CAMERA_WIDTH, CAMERA_HEIGHT],
                            "camera_matrix": {"fx":float(K[0,0]),"fy":float(K[1,1]),"cx":float(K[0,2]),"cy":float(K[1,2])},
                            "dist_coeffs": [round(v,8) for v in dist.flatten().tolist()],
                        }, allow_unicode=True, sort_keys=False), encoding="utf-8")
                        print(f"Saved to {OUTPUT_YAML}. Main config NOT updated.")
                break
            if key == ord('s') and ret:
                p = SAVE_DIR / f"{img_count}.jpg"
                cv2.imwrite(str(p), img)
                print(f"#{img_count} saved -> {p}")
                img_count += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    sys.exit(main())
