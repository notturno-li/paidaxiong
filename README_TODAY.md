# 2026-08-18 今日国赛程序

无网络现场请双击 `open_today_offline_runbook.bat`，按 `TODAY_OFFLINE_RUNBOOK.md` 的顺序操作。

题面整理见 `../gs/2026-08-18_今日国赛题整理.md`，明日预测见 `TOMORROW_TASK_PREDICTION.md`。

## 启动顺序

1. 双击 `run_today_dataset_studio.bat`，按固定顺序使用 7 个中文标签采集、标注并训练。训练产物必须是 `models/field_models/today_shape_*/best.pt`；程序自动选择最新模型，并严格核对模型类别顺序。
2. 在 `configs/today.yaml` 中示教并填写 `robot.photo_pose`、7 个 `robot.bins`、手眼矩阵和抓取 Z 偏置。当前位姿全部是占位值。`robot.bins.<类别>` 是释放时的 TCP 位姿，不是相机坐标，也不是手眼矩阵。
3. 用模拟模式运行 `run_today_national.bat`，验证“启动相机”“单次运行”“自动运行”。模拟自动流程会按高度顺序完成 5 件。
4. 空载、低速验证拍照点和 7 个组装点；确认基坐标角度方向、`tool_yaw_offset_deg` 和吸取不打滑后，才能把 `mode` 改为 `hardware`。
5. 真机运行前再执行完整测试和 GUI 校验。

```bat
python -m unittest discover -s tests -v
python tools\verify_today_gui.py --windows
```

## 今日位姿示教

现场示教器中记录的 6 轴位姿顺序为 `X,Y,Z,Rx,Ry,Rz`，单位为 mm/度。

- `robot.photo_pose`：相机拍照位，必须高于 `robot.min_grasp_z_mm`。
- `robot.bins.<类别>`：该类别装配槽的最终释放 TCP 位姿，`Rz` 是零件在槽中的目标角度。
- `robot.safe_z_mm`：所有跨区域移动的安全高度。
- `assembly.slots.<类别>.approach_clearance_mm`：该槽位最终下降前的上方接近距离。

示教脚本支持今日配置的行内数组和拍照位：

```bat
python scripts\teach_competition_poses.py --config configs\today.yaml --skip-home --skip-fixed
```

脚本会在每个步骤显示当前值，把机械臂手动点动到位后按回车读取 `GetPose()`，最后写回配置。首次使用建议加 `--dry-run` 检查输出。

## 手眼采集内参

手眼采集、求解和正式运行统一使用 `640x480 @ 30 FPS`。运行 `run_hand_eye_calibration.bat` 后，先选择 `5` 归档旧数据，再选择 `1` 重新采集；采集脚本会把 RealSense 当前彩色流内参和相机序列号写入 `runs/calib_data/camera_intrinsics.yaml`。

诊断和求解只使用这份与图像同批次保存的内参，不读取 `configs/competition.yaml` 或 `runs/intrinsic/intrinsic.yaml`。发现旧的 `1280x720` 图像、错误分辨率内参或不同相机序列号时会直接停止。菜单选项 `4` 只是外部内参核验，使用原厂 RealSense 时通常不用执行。

## 角度估计

今日配置的角度默认由 RGB-D 深度点转换到机械臂基坐标系后计算，不再依赖图像坐标的固定符号。圆柱忽略角度；正方体按 90 度对称；六棱柱按 60 度对称；长方体和平行四边形按 180 度对称；梯形使用有方向轮廓。

真实模式下 `orientation.require_valid: true` 时，深度点不足、主方向不明显或轮廓不稳定会阻断运动，不会把失败结果当成 0 度。

## 模块 B TCP 输入

今日 GUI 默认作为 TCP 客户端连接 DVS。端点配置位于 `configs/today.yaml`：

```yaml
task_outputs:
  tcp:
    enabled: true
    host: 192.168.5.4
    port: 12345      # 端口确定后填写，修改后重启 GUI
```

连接在后台运行，断开后每秒自动重连；GUI 每 100 ms 取出新报文并刷新监控区，不会阻塞相机或急停。默认以换行符分隔报文，推荐 DVS 使用以下任一种 UTF-8 格式：

```text
gear=36,1.5,57.0,51.2
recognition=OK,DOBOT123,QR001
defect=NG,2,18.6,3.1
measurement=25.0,18.0,32.2,16.4,45.0
```

也可每行发送一个 JSON 对象：

```json
{"gear":"36,1.5,57.0,51.2","defect":"NG,2,18.6,3.1"}
```

字段前缀支持英文键和 `field_aliases` 中配置的中文别名。若 DVS 只能发送不带字段名的纯文本，需把 `default_key` 设置为 `gear`、`recognition`、`defect` 或 `measurement`；多类结果共用一个连接时必须携带字段前缀或使用 JSON，防止内容进入错误监控项。

## 模块 B 文件兼容输入

原有文本文件输入仍保留。需要启用时把 `task_outputs.file_fallback_enabled` 改为 `true`，并可把 `task_outputs.directory` 改为现场的绝对路径。TCP 模式下默认关闭文件回读，避免目录里的旧结果覆盖实时 TCP 数据。文件名及建议内容：

```text
齿轮识别.txt                    z,m,da,df 或 NG,...
字符识别与二维码识别.txt        OK/NG,字符信息,二维码信息
缺陷检测.txt                    NG,缺陷数量,最大面积,最小面积
尺寸测量.txt                    保留 DVS 的完整原始测量行
```

尺寸测量在题面页 9 与页 15 存在 C1/C2/C3 字段冲突。程序不删字段，只显示完整原始行；验证前以裁判现场口径为准。

## 现场阻断条件

- 最新模型不存在或模型自带类别顺序与 7 个中文标签不完全一致时，真机启动会失败，避免静默错分拣。
- 拍照点 Z 低于安全下限时自动流程会停止。
- 基坐标角度到机械臂 TCP yaw 的零偏、槽位 Rz 和吸盘抗滑移没有用实物校验前，不得执行自动组装。
- 当前默认 `simulation`。模型、位姿、标定和角度映射四项未完成前不要切换到 `hardware`。
