# 全国赛模块化程序

入口为 `python -m app.modular_main`，Windows 默认运行 `run_competition_gui.bat`（`run_national_gui.bat` 也可用）。原省赛入口 `python -m app.main` 和 `run_provincial_gui.bat` 保留，已有标定矩阵、抓取补偿和料盒坐标继续来自 `configs/competition.yaml`。

国赛 GUI 实际加载 `configs/field.yaml`。它继承 `national.yaml`，保存现场参数不会覆盖省赛标定或国赛模板。

现场禁止使用 AI 时，正常流程不需要编辑代码或 YAML：

- “通信/高级”填写 DVS 的现场端点和报文格式。
- “任务向导”从 10 类高概率题型生成 Profile、判定、路线、循环参数、永久配方和推荐模块队列。
- “动作编辑器”用表格拼装固定取料、送检、回收、吸盘和等待步骤。
- “模型管理”扫描现场训练模型、导入外部 `.pt`，验证成功后热加载并同步类别配置。
- “赛前自检”核对依赖、模型、标定、固定机器人端点、DVS、字段、动作引用、配方和安全策略。

需要现场拍摄、多人标注和训练 YOLO 时，双击 `run_dataset_studio.bat`。队友可通过页面显示的局域网地址并行标注，训练完成的 `best.pt + model_profile.yaml` 会自动出现在比赛 GUI 的“模型管理”列表。完整流程见 [DATASET_STUDIO_GUIDE.md](DATASET_STUDIO_GUIDE.md)。普通 USB 相机可使用 `run_dataset_studio_usb.bat`。

完整的离线操作顺序见 [OFFLINE_FIELD_MANUAL.md](OFFLINE_FIELD_MANUAL.md)，打印版可双击 `open_field_manual.bat` 打开。

## 结构

- `app/core/`：相机、检测、测高、坐标转换、机器人控制、JSON TCP 传输和 DobotVisionStudio 通信。
- `app/core/vision_task.py`：DVS 字段标准化、公差/质量判定、识别校验、条件路由和动作位姿解析。
- `app/modules/core.py`：功能模块注册表、依赖解析和顺序执行器。
- `configs/national.yaml`：国赛界面标题、现场安全选项和可组合配方。
- `app/modular_main.py`：模块库、执行队列、配方选择、单模块执行和组合执行。
- `dataset_studio/`：独立的采集、局域网多人标注、数据集构建和 Ultralytics 训练工作台。

模块会自动补齐依赖。例如现场只运行“视觉单物料分拣”，执行器会依次补齐相机启动、机器人连接、单帧识别和坐标解算。已经就绪的依赖不会重复启动，显式加入队列的模块仍可重复执行。

在模块编排台排列好队列后点击“保存配方”，当前顺序会写入 `configs/field.yaml` 并立即出现在配方下拉框中，重启后仍然保留。

“完整结果”会显示 DVS 原始返回、标准化字段、尺寸越界项、判定路线、抓放位姿、连续任务统计和槽位计数，适合尺寸测量或综合识别题核对全部字段。

连续任务会记录每件工件的字段、识别值、判定、路线和违规原因，并汇总 OK/NG 与各路线数量。“导出结果”将完整数据保存为 JSON，并把逐件记录另存为 Excel 可直接打开的 UTF-8 CSV。

## 现场建议配方

- `赛台初始化`：空台面时连接设备、标定桌面并回安全点。
- `视觉与坐标联调`：不让机器人抓取，只检查类别、高度和基坐标。
- `JSON 通信握手`：生成抓取动作 JSON，发送到机器人侧任务服务器并检查 ACK。
- `固定点抓放`、`单物料分拣`、`连续分拣`：按需独立运行。
- `国赛自动分拣`：默认的一键连续分拣配方。
- `DVS 通信端点扫描`：列出当前计算机可用串口，不发送报文。
- `DVS 触发与结果读取`：按配置向 DobotVisionStudio 触发流程并读取结果。
- `DVS 协同自动分拣`：先获取 DVS 数据，再执行自动分拣。
- `DVS 尺寸测量与公差判定`：按现场 min/max 自动输出 OK/NG 和越界项。
- `DVS 字符/条码/二维码识别`：检查要求的识别结果是否完整。
- `DVS 缺陷判定与分拣`、`DVS 类别/颜色分拣`、`DVS 定位引导装配`：把 DVS 字段直接路由为机械臂抓放动作。
- `预览并校验 DVS 动作坐标`：只计算抓取/放置位姿和安全下限，不连接机器人；现场首次联调应先单独运行它。
- `DVS 多工件连续任务`：按现场配置循环触发、解析、判定、路由和抓放，支持空料停止与急停取消。

连续任务还可指定“每件检测前动作”和“每件完成后动作”。典型组合是 `inspection_feed_template → DVS 触发/判定 → DVS 坐标分流`，可直接覆盖教学案例中的“取料区 → 检测区 → 合格/废料区”流程。

## 三条独立通信链路

机械臂 Dashboard 仍使用省赛参数 `192.168.5.1:29999`，国赛配置直接继承，不需要修改。

`command_transport` 是机器人侧 JSON 任务服务；`vision_studio` 是 DobotVisionStudio 的通信设备。两者不能共用配置，也都不会覆盖机械臂 Dashboard 端口。

## DobotVisionStudio 通信

DobotVisionStudio 手册明确支持串口、TCP 客户端和 TCP 服务端。当前 IP、端口、COM 口及比赛报文尚未确认，所以 [national.yaml](configs/national.yaml) 使用 `vision_studio.mode: disabled`，空参数不会触发连接。

串口模式示例：

```yaml
vision_studio:
  mode: serial
  send_terminator: "\n"
  receive_terminator: "\n"
  serial:
    port: COM3
    baudrate: 115200
  protocol:
    exchange_command: RUN
    delimiter: ","
    response_fields: [label, u, v, height_mm]
```

如果 DVS 与本程序在同一台电脑，最简单的 TCP 方案是在 DVS 中新建 TCP 服务端，自行选一个未占用端口，然后本程序使用 `tcp_client` 连接 `127.0.0.1`。端口不是 DVS 固定内置值，必须在两端配置一致：

```yaml
vision_studio:
  mode: tcp_client
  tcp:
    host: 127.0.0.1
    port: 19001
```

DVS 也可作为 TCP 客户端，此时将本程序设为 `tcp_server`，`host` 填本机监听地址（通常 `0.0.0.0`），并在 DVS 中把目标地址指向本机及同一端口。

手册没有规定比赛专用触发字符串和返回字段。拿到现场协议后，只需填写 `exchange_command`、结束符、`delimiter` 和 `response_fields`；返回 JSON 时无需配置字段映射。

教学案例中已经出现三种典型返回格式，可直接按下面方式适配：

```yaml
# NG   312.5   -18.2（任意数量空格或制表符）
protocol:
  delimiter: whitespace
  response_fields: [status, x, y]

# 绿色,312.5,-18.2
protocol:
  delimiter: ","
  response_fields: [label, x, y]

# {"status":"OK","x":312.5,"y":-18.2}
protocol:
  response_fields: []  # JSON 会自动解析
```

如果由 DVS 在流程结束后主动上报结果，把 `protocol.send_before_receive` 设为 `false`，本程序将只等待并解析数据，不要求配置 `exchange_command`。默认值 `true` 表示本程序先发送触发/查询字符串，再等待结果。

DVS 返回值同时保存在 `workflow.vision_studio_data` 和 `workflow.shared_data["vision_studio"]`。后续比赛若要求用 DVS 返回的类别、坐标、流程状态或测量值参与决策，可新增业务模块读取共享数据，不需要修改串口/TCP 驱动。

DVS 手册定义的 `SetGlobalValue:name=value` 已作为“设置 DVS 全局变量”模块开放。可在“通信/高级”的通信页填写 `task=1; threshold=0.8`，再把该模块放到“DVS 触发/取结果”之前。

不启动 GUI 时，可用独立探针先验证通信。它不会连接机械臂：

```bash
python tools/probe_dvs.py --discover
python tools/probe_dvs.py --connect
python tools/probe_dvs.py --exchange --count 3
python tools/probe_dvs.py --exchange --message get_photo
python tools/probe_dvs.py --set-global task=1
```

Windows 双击 `run_dvs_probe.bat` 可先列出串口和当前 TCP 配置。

没有 DVS 设备时可演练真实 TCP 收发。先启动模拟服务端，再在另一个终端运行探针：

```bash
python tools/mock_dvs_server.py --count 3
python tools/probe_dvs.py --config configs/mock_dvs.yaml --exchange --count 3
```

`19001` 只用于本机测试，不代表国赛端口。Windows 也可先双击 `run_mock_dvs.bat`。

`vision_tasks` 是现场题型适配层。正常现场使用“任务向导”生成；下面的字段仅用于赛前开发或高级排错：

- `aliases`：把 DVS 的实际字段名映射为统一字段。
- `measurement_limits`：尺寸测量允许的最小值和最大值。
- `identifier_fields`：本题必须返回的 OCR、条码或二维码字段。
- `status` 和 `route`：把视觉结果转成 OK/NG 或类别动作路线。
- `pick`、`place`：从返回字段生成抓取/装配位姿，或按路线选固定放置点。

`place.route_slots` 可让同一路线按顺序使用多个放置槽位；`place.route_stacks` 可按 `base_pose + 层号 × layer_height_mm` 自动堆叠。槽位或层数用完后流程会报错停止，不会覆盖已放置工件。“赛台初始化”和连续任务默认会清零计数，也可单独运行“清零槽位/堆叠计数”。

“任务向导”提供仅通信、缺陷、类别、尺寸、尺寸后分拣、识别、识别核验后分拣、动态装配、固定送检和综合自定义十类入口，并可在分拣题中选择固定点、顺序槽位或自动堆叠。识别核验支持精确值、允许列表和正则表达式；综合题可叠加状态、尺寸、识别、路由和动作。

“动作编辑器”可从 `inspection_feed_template` 和 `inspection_retrieve_template` 开始，用于固定取料送检、从检测区取回等步骤。先在位姿库填写 `input_above`、`input_pick`、`inspection_above` 等现场示教位姿，再单独运行“预览并校验固定动作序列”，确认通过后才能执行。

资料依据和题型概率见 [NATIONAL_TASK_PREDICTION.md](NATIONAL_TASK_PREDICTION.md)。

比赛当天按 [OFFLINE_FIELD_MANUAL.md](OFFLINE_FIELD_MANUAL.md) 和 [FIELD_DAY_GUIDE.md](FIELD_DAY_GUIDE.md) 的顺序联调，先通信、再字段、再无运动预览，最后才启用机器人动作。

## 离线备份

联网时先运行 `prepare_offline_dependencies.bat` 下载 Windows 离线 wheel。备用电脑可运行 `install_offline.bat` 恢复依赖。`prepare_field_package.bat` 会先执行自检，再把程序、配置、模型、文档以及已有 wheelhouse 打成 ZIP。

串口通信所需的纯 Python `pyserial 3.5` 已随工程放在 `serial/`，即使现场无法运行 pip，DVS 串口扫描和通信仍可使用；许可证见 `third_party_licenses/pyserial-LICENSE.txt`。

## Windows BAT 入口

| 文件 | 用途 |
|---|---|
| `run_competition_gui.bat` / `run_national_gui.bat` | 启动国赛模块化 GUI，两者功能相同 |
| `run_provincial_gui.bat` | 启动保留的省赛 GUI |
| `run_dataset_studio.bat` | 用 RealSense 启动数据集采集、局域网标注和训练工作台 |
| `run_dataset_studio_usb.bat` | 用普通 USB 相机启动数据集工作台 |
| `run_hand_eye_calibration.bat` | 手眼标定中文菜单：采集、诊断、求解、内参和旧数据归档 |
| `run_preflight.bat` | 运行赛前自检 |
| `run_dvs_probe.bat` | 扫描并测试 DobotVisionStudio 通信端点 |
| `run_mock_dvs.bat` | 启动本机 DVS 模拟服务器，用于无硬件演练 |
| `install_offline.bat` | 从 `wheelhouse` 离线安装 Python 依赖 |
| `prepare_offline_dependencies.bat` | 联网时下载现场离线依赖 |
| `prepare_field_package.bat` | 自检后生成国赛现场 ZIP |
| `open_field_manual.bat` | 打开现场快速操作卡 |

`run_hand_eye_calibration.bat` 的自动采集只会向 `192.168.5.1:29999` 发送 `GetPose()` 读取当前位姿，不会下发机械臂运动指令。求解成功后会更新 `configs/competition.yaml` 的 `transform_camera_to_gripper`，国赛 `field.yaml` 会通过继承自动使用该矩阵。

## 机器人任务 TCP 模式

`configs/national.yaml` 默认使用 `command_transport.mode: preview`，不会连接未知的机器人任务服务端口。该设置与 DVS 通信无关。机器人侧任务服务器的 IP 和端口确认后改为：

```yaml
command_transport:
  mode: tcp
  host: 192.168.5.1
  port: 10000
```

服务器需接收一行 UTF-8 JSON，并返回 `{"ok": true}`、`{"status": "accepted"}` 或文本 `OK`。未收到有效 ACK 时模块会失败，界面不会显示伪成功。

## 国赛规则状态

项目内只有黑龙江省赛任务书，公开检索也未找到可核验的 2026 全国赛正式任务书。因此 `national.yaml` 是“全国赛可调整预设”，并不冒充正式评分流程。正式任务书到手后，通常只需修改 `recipes`；如果出现新硬件能力，再向模块注册表添加一个独立模块。
