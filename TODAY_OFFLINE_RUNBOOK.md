# 今日任务离线执行手册

适用工程：`robotgame`

用途：现场没有网络、没有 AI 时，按本文档完成安装、标定、示教、验证和正式运行。

## 0. 先确认两个前提

1. 当前手眼程序按“相机安装在机械臂末端”设计，使用 `相机 -> 末端` 手眼矩阵。
2. 如果相机固定在桌面或外部支架上，立即停止，不要运行当前手眼求解；固定相机需要另一套 `相机 -> 基座` 标定程序。

相机、相机支架、吸盘、机械臂底座、工作台或装配盘只要移动过，都要按本文后面的“什么变化后重做什么”处理。

## 1. 首次完整准备：严格按顺序执行

### 第 1 步：安装离线依赖（只在新电脑执行）

双击：

```text
install_offline.bat
```

如果提示缺少 `wheelhouse`，说明离线依赖包没有提前准备，不能靠现场网络临时下载。

### 第 2 步：用机械臂示教器完成基础设置

这一步没有对应的 Python 程序，需要在 Dobot 示教器或厂家工具中完成：

- 确认机械臂 IP 为 `192.168.5.1`，Dashboard 端口为 `29999`。
- 确认运行和标定使用相同的 `User`、`Tool` 编号。
- 标定吸盘 TCP，TCP 原点应位于实际吸取中心。
- 确认吸盘 DO 编号、吸取电平、释放电平。
- 低速测试吸取、释放和急停。
- 确认 `GetPose()` 返回的是手眼程序所需的末端/工具位姿。

没有完成 TCP 和工具坐标确认，不要开始手眼标定。

### 第 3 步：手眼标定

准备标定板：当前程序使用 `8 x 11` 个内角点，方格边长配置为 `10.0 mm`。实际标定板不一致时，先修改脚本参数，不能带错尺寸求解。

双击：

```text
run_hand_eye_calibration.bat
```

菜单操作顺序：

```text
5  归档旧采集数据
1  自动采集图像和机器人位姿
2  诊断采集数据和欧拉角约定
3  求解手眼矩阵并写入比赛配置
Q  退出
```

采集要求：

- 标定板固定不动，相机随机械臂移动。
- 采集 15～20 组，覆盖不同 X/Y/Z 和不同倾角。
- 机械臂停稳、绿色角点完整显示后按 `S`。
- 每张图必须有同编号的机器人位姿文件。
- 按 `Q` 结束采集后再诊断和求解。

相机内参规则：

- 手眼采集、诊断、求解和正式运行统一使用 `640 x 480 @ 30 FPS`。
- 采集时会自动读取 RealSense 当前彩色流内参。
- 本批次内参、畸变参数和相机序列号保存在 `runs/calib_data/camera_intrinsics.yaml`。
- 求解不读取 `configs/competition.yaml` 或 `runs/intrinsic/intrinsic.yaml` 中的内参。
- 发现旧 `1280 x 720` 图像、错误分辨率内参或不同相机序列号时，程序会直接停止。
- 菜单 `4` 是外部内参核验，原厂 RealSense 通常不需要执行。

求解完成后，检查输出：

- 有效样本建议不少于 15 组。
- 平移误差 RMS 建议不超过 3～5 mm。
- 旋转误差 RMS 建议不超过 1～2 度。
- 误差过大时不要继续，重新检查标定板尺寸、姿态对应关系、欧拉角约定和相机固定情况。

### 第 4 步：采集、标注和训练七类模型

已有合格现场模型时可跳过。没有模型时双击：

```text
run_today_dataset_studio.bat
```

依次完成采集、标注、检查和训练。类别顺序必须保持：

```text
六棱柱
正方体
圆柱
平行四边形
长方体
大圆柱
梯形
```

训练结束后确认存在：

```text
models/field_models/today_shape_*/best.pt
```

没有正确模型或类别顺序不一致，禁止进入真机模式。

模型配置：D:\Projects\raicom\26\ss\robotgame\models\field_models\today_shape_20260819_093529\model_profile.yaml 权重：D:\Projects\raicom\26\ss\robotgame\models\field_models\today_shape_20260819_093529\best.pt

### 第 5 步：示教拍照位和七个装配位

先确认机械臂周围无人、无障碍物。第一次可以执行只预览不写配置：

```bat
python scripts\teach_competition_poses.py --config configs\today.yaml --skip-home --skip-fixed --dry-run
```

确认流程后正式写入：

```bat
python scripts\teach_competition_poses.py --config configs\today.yaml --skip-home --skip-fixed
```

按提示依次示教：

```text
photo_pose
六棱柱装配释放位
正方体装配释放位
圆柱装配释放位
平行四边形装配释放位
长方体装配释放位
大圆柱装配释放位
梯形装配释放位
```

每个 `robot.bins.<类别>` 都是最终释放时的 TCP 六轴位姿，不是相机坐标。位姿格式为 `[X,Y,Z,Rx,Ry,Rz]`，单位为 mm/度。

当前 `today.yaml` 的 `home_pose` 和 `fixed_test_pose` 可能继承自 `competition.yaml`，上述命令不会重新写这两个值。必须用示教器单独确认，并把现场值明确填写到 `configs/today.yaml`：

```text
robot.home_pose
robot.fixed_test_pose
```

### 第 6 步：检查今日配置

打开 `configs/today.yaml`，逐项确认：

```text
camera.width = 640
camera.height = 480
camera.fps = 30
robot.home_pose
robot.photo_pose
robot.fixed_test_pose
robot.bins 下的七个类别位姿
robot.safe_z_mm
robot.min_grasp_z_mm
robot.grasp_x_offset_mm
robot.grasp_y_offset_mm
robot.grasp_z_offset_mm
calibration.transform_camera_to_gripper
orientation.tool_yaw_offset_deg
assembly.slots 下的 approach_clearance_mm
task_outputs.tcp.host
task_outputs.tcp.port
```

第一次验证时必须保持：

```yaml
mode: simulation
```

### 第 7 步：运行离线自检

执行标定相关测试：

```bat
python -m unittest tests.test_calib_metadata tests.test_object_pose -v
```

执行今日配置自检：

```bat
python tools\preflight.py --config configs\today.yaml
```

注意：不要用默认参数直接运行 `run_preflight.bat` 检查今日任务，因为它默认指向通用 `field.yaml`。

出现任何 `[阻止]` 时，不得启动机械臂动作。

### 第 8 步：运行模拟流程

双击：

```text
run_today_national.bat
```

GUI 中依次点击：

```text
启动相机
单次运行
自动运行
```

确认检测类别、抓取点、物体高度、角度、执行顺序和七个目标槽位逻辑正确。

### 第 9 步：运行只连接、不运动的硬件自检

连接 RealSense、机械臂和 DVS 后执行：

```bat
python tools\preflight.py --config configs\today.yaml --hardware
```

该命令只探测连接，不应发送业务运动命令。确认没有 `[阻止]`。

### 第 10 步：空载低速验证固定位置

先把 `configs/today.yaml` 改为：

```yaml
mode: hardware
```

再次双击：

```text
run_today_national.bat
```

先不要放工件。低速逐个验证：

```text
home_pose
photo_pose
fixed_test_pose
七个装配槽上方接近位
七个装配最终释放位
吸盘打开和关闭
```

检查所有路径不会碰撞，所有 Z 均高于安全下限。

### 第 11 步：逐类单件验证

每次只放一个物体，在 GUI 中执行“单次运行”。顺序建议：

```text
长方体
平行四边形
梯形
正方体
六棱柱
圆柱
大圆柱
```

前三类优先验证角度。每类确认：类别正确、抓取点安全、深度合理、角度正确、吸取不滑动、目标槽正确、释放可靠。

如果所有有方向工件都固定偏转同一个角度，只调整 `orientation.tool_yaw_offset_deg`。梯形整体反向时检查 `directed_sign`。不得通过关闭 `orientation.require_valid` 强行运行。

### 第 12 步：多件自动运行

只有空载和七类单件全部通过后，才点击：

```text
自动运行
```

第一次多件运行继续使用低速度，并让操作员始终处于急停可达位置。

## 2. 以后每天开机的最短顺序

硬件和安装都没有变化时，不需要每天重做手眼、TCP 或模型训练。

```text
1. 检查相机、吸盘、机械臂底座、工作台和装配盘没有移动
2. 检查电源、网线、USB、气路和急停
3. python tools\preflight.py --config configs\today.yaml --hardware
4. 双击 run_today_national.bat
5. 点击“启动相机”
6. 空载确认 home_pose 和 photo_pose
7. 放一个有方向工件执行“单次运行”
8. 确认角度和装配位后再执行“自动运行”
```

## 3. 什么变化后重做什么

| 现场变化 | 必须重做 |
|---|---|
| 更换电脑，工程和离线依赖完整 | 安装依赖、自检；通常不用重做标定 |
| 更换 RealSense 相机 | 归档旧数据、重新手眼标定和抓取验证 |
| 更换镜头或改变彩色流分辨率 | 内参核验、重新手眼标定和抓取验证 |
| 相机或相机支架移动 | 重新手眼标定 |
| 吸盘、法兰连接或 TCP 改变 | 重新 TCP 标定、手眼标定和抓取偏置验证 |
| 机械臂底座移动 | 重新手眼标定，并重新示教全部固定位姿 |
| 工作台或物料区移动 | 重新验证桌面平面、抓取偏置和拍照位 |
| 装配盘移动 | 重新示教七个装配位 |
| 光照或背景明显变化 | 重新验证模型；必要时补数据训练 |
| 类别或题面改变 | 重新配置类别、训练模型和示教对应槽位 |
| 工件在吸盘上滑动 | 检查吸盘、气压、TCP、抓取点和角度偏置 |

## 4. 常见阻止信息

| 提示 | 处理方法 |
|---|---|
| 缺少 `camera_intrinsics.yaml` | 菜单选 `5` 归档旧数据，再选 `1` 重新采集 |
| 图像是 `1280 x 720` | 归档旧数据，重新采集 `640 x 480` |
| 相机序列号不一致 | 确认是否接错相机；更换相机后必须重新手眼 |
| 内参不一致 | 确认分辨率和设备；不要复制其他内参文件 |
| 图像缺少同编号位姿 | 删除/归档不完整批次后重新采集 |
| 手眼 RMS 误差过大 | 检查方格尺寸、角点数、姿态覆盖、欧拉角和板是否移动 |
| 模型不存在 | 运行 `run_today_dataset_studio.bat` 完成训练 |
| 类别顺序不一致 | 按固定七类顺序重新配置或训练 |
| 拍照位或目标位 Z 过低 | 重新示教，不得降低安全线绕过 |
| DVS 连接失败 | 核对双方客户端/服务端角色、IP、端口、防火墙和网段 |
| 角度无效 | 检查深度、遮挡、反光和工件点数，不得强制当作 0 度 |

## 5. 必须备份的文件

完成现场标定后，复制到两个不同 U 盘：

```text
configs/today.yaml
configs/competition.yaml
runs/calib_data/camera_intrinsics.yaml
runs/calib_data/images/
runs/calib_data/poses/
hand_eye_result.yaml
models/field_models/today_shape_*/
```

同时记录：相机序列号、TCP 工具号、User 坐标号、吸盘 DO、标定日期、操作者和最终误差。

## 6. 一页式顺序

```text
示教器完成 TCP/吸盘设置
        ↓
run_hand_eye_calibration.bat：5 → 1 → 2 → 3
        ↓
run_today_dataset_studio.bat（已有正确模型则跳过）
        ↓
teach_competition_poses.py 示教 photo_pose 和七个装配位
        ↓
确认 home_pose、fixed_test_pose 和 today.yaml
        ↓
标定测试 + today.yaml 离线自检
        ↓
run_today_national.bat 模拟运行
        ↓
硬件自检
        ↓
空载低速验证
        ↓
七类单件验证
        ↓
多件自动运行
```

