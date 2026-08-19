from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Iterable, Optional

from app.core.workflow import CompetitionWorkflow


ModuleAction = Callable[[CompetitionWorkflow], Any]


@dataclass(frozen=True)
class FeatureModule:
    module_id: str
    name: str
    group: str
    action: ModuleAction
    dependencies: tuple[str, ...] = ()
    description: str = ""
    moves_robot: bool = False
    cacheable: bool = False


@dataclass(frozen=True)
class ModuleExecution:
    module_id: str
    name: str
    elapsed_s: float
    value: Any = None


class ModuleRegistry:
    def __init__(self):
        self._modules: dict[str, FeatureModule] = {}

    def register(self, module: FeatureModule) -> None:
        if module.module_id in self._modules:
            raise ValueError(f"模块重复注册: {module.module_id}")
        self._modules[module.module_id] = module

    def get(self, module_id: str) -> FeatureModule:
        try:
            return self._modules[module_id]
        except KeyError as exc:
            raise KeyError(f"未知功能模块: {module_id}") from exc

    def all(self) -> list[FeatureModule]:
        return list(self._modules.values())

    def resolve(self, module_ids: Iterable[str], include_dependencies: bool = True) -> list[FeatureModule]:
        ordered: list[FeatureModule] = []
        visited: set[str] = set()
        visiting: set[str] = set()

        def visit(module_id: str) -> None:
            if module_id in visited:
                return
            if module_id in visiting:
                raise RuntimeError(f"模块依赖存在循环: {module_id}")
            module = self.get(module_id)
            visiting.add(module_id)
            if include_dependencies:
                for dependency in module.dependencies:
                    visit(dependency)
            visiting.remove(module_id)
            visited.add(module_id)
            ordered.append(module)

        for requested in module_ids:
            visit(requested)
        return ordered


class ModuleExecutor:
    def __init__(self, workflow: CompetitionWorkflow, registry: ModuleRegistry, logger=print):
        self.workflow = workflow
        self.registry = registry
        self.log = logger
        self.completed: set[str] = set()

    def execute(
        self,
        module_ids: Iterable[str],
        auto_dependencies: bool = True,
        progress: Optional[Callable[[str, str], None]] = None,
    ) -> list[ModuleExecution]:
        requested = list(module_ids)
        if not requested:
            raise RuntimeError("模块队列为空")
        plan = self.registry.resolve(requested, include_dependencies=auto_dependencies)
        explicit = set(requested)
        results: list[ModuleExecution] = []
        for module in plan:
            if module.cacheable and module.module_id in self.completed and module.module_id not in explicit:
                self.log(f"跳过已就绪依赖：{module.name}")
                if progress is not None:
                    progress(module.module_id, "skipped")
                continue
            if self.workflow.robot.cancel_event.is_set():
                raise RuntimeError("任务已取消")
            self.log(f"模块开始：{module.name}")
            if progress is not None:
                progress(module.module_id, "running")
            started = perf_counter()
            try:
                value = module.action(self.workflow)
            except Exception:
                if progress is not None:
                    progress(module.module_id, "failed")
                raise
            elapsed = perf_counter() - started
            self.completed.add(module.module_id)
            results.append(ModuleExecution(module.module_id, module.name, elapsed, value))
            if progress is not None:
                progress(module.module_id, "success")
            self.log(f"模块完成：{module.name} ({elapsed:.2f}s)")
        return results

    def invalidate(self, *module_ids: str) -> None:
        for module_id in module_ids:
            self.completed.discard(module_id)


def build_standard_registry() -> ModuleRegistry:
    registry = ModuleRegistry()
    modules = [
        FeatureModule("camera.start", "启动 RGB/Depth 相机", "视觉", lambda w: w.start_camera(), cacheable=True),
        FeatureModule("camera.capture", "保存当前图像", "视觉", lambda w: w.save_current_frame(), ("camera.start",)),
        FeatureModule("calibration.table", "标定桌面平面", "标定", lambda w: w.calibrate_table(), ("camera.start",)),
        FeatureModule("vision.detect", "单帧识别与测高", "视觉", lambda w: w.detect_once(), ("camera.start",)),
        FeatureModule("vision_studio.discover", "扫描 DVS 端点", "DVS", lambda w: w.discover_vision_studio()),
        FeatureModule("vision_studio.connect", "连接 DVS", "DVS", lambda w: w.connect_vision_studio(), cacheable=True),
        FeatureModule("vision_studio.set_globals", "设置 DVS 全局变量", "DVS", lambda w: w.set_vision_studio_globals(), ("vision_studio.connect",)),
        FeatureModule("vision_studio.exchange", "DVS 触发/取结果", "DVS", lambda w: w.exchange_vision_studio(), ("vision_studio.connect",)),
        FeatureModule("vision_result.normalize", "标准化 DVS 返回字段", "结果处理", lambda w: w.normalize_vision_result(), ("vision_studio.exchange",)),
        FeatureModule("vision_result.measure", "尺寸与公差判定", "结果处理", lambda w: w.evaluate_measurements(), ("vision_result.normalize",)),
        FeatureModule("vision_result.quality", "缺陷 OK/NG 判定", "结果处理", lambda w: w.evaluate_quality_status(), ("vision_result.normalize",)),
        FeatureModule("vision_result.identify", "OCR/条码/二维码校验", "结果处理", lambda w: w.validate_identifiers(), ("vision_result.normalize",)),
        FeatureModule("vision_result.route", "按结果选择动作路线", "结果处理", lambda w: w.route_vision_result(), ("vision_result.normalize",)),
        FeatureModule("vision_result.plan_motion", "预览并校验 DVS 动作坐标", "坐标", lambda w: w.plan_result_pick_place(), ("vision_result.normalize",)),
        FeatureModule("vision_result.reset_places", "清零槽位/堆叠计数", "结果处理", lambda w: w.reset_placement_counters()),
        FeatureModule("robot.connect", "连接并使能机器人", "机器人", lambda w: w.connect_robot(), cacheable=True),
        FeatureModule("robot.sequence_plan", "预览并校验固定动作序列", "坐标", lambda w: w.plan_action_sequence()),
        FeatureModule("robot.sequence_execute", "执行固定取放/送检序列", "机器人", lambda w: w.execute_action_sequence(), ("robot.sequence_plan", "robot.connect"), moves_robot=True),
        FeatureModule("robot.fixed_pick", "固定点抓放", "机器人", lambda w: w.fixed_point_test(), ("robot.connect",), moves_robot=True),
        FeatureModule("target.calculate", "解算抓取坐标", "坐标", lambda w: w.calculate_grasp(), ("vision.detect",)),
        FeatureModule("network.send", "下发 JSON 并握手", "通信", lambda w: w.send_grasp_command(), ("robot.connect", "target.calculate")),
        FeatureModule("robot.pick_one", "视觉单物料分拣", "分拣", lambda w: w.execute_grasp(), ("robot.connect", "target.calculate"), moves_robot=True),
        FeatureModule("robot.result_pick_place", "按 DVS 坐标抓放/装配", "机器人", lambda w: w.execute_result_pick_place(), ("vision_result.plan_motion", "robot.connect"), moves_robot=True),
        FeatureModule("sort.auto", "多目标连续分拣", "分拣", lambda w: w.auto_run(), ("camera.start", "robot.connect"), moves_robot=True),
        FeatureModule("task.dvs_cycle", "DVS 多工件连续任务", "流程", lambda w: w.run_dvs_cycle(), moves_robot=True),
        FeatureModule("robot.home", "机器人返回安全点", "机器人", lambda w: w.return_home(), ("robot.connect",), moves_robot=True),
    ]
    for module in modules:
        registry.register(module)
    return registry
