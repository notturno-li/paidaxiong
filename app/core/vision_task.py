from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping


@dataclass(frozen=True)
class VisionTaskResult:
    fields: dict[str, Any]
    decision: str | None = None
    route: str | None = None
    violations: tuple[str, ...] = ()
    identifiers: dict[str, Any] = field(default_factory=dict)


class VisionTaskProcessor:
    """Convert task-specific DVS replies into one configurable data contract."""

    def __init__(self, settings: Mapping[str, Any]):
        self.settings = dict(settings)
        self.validate_settings()

    def validate_settings(self) -> None:
        aliases = self.settings.get("aliases", {})
        if not isinstance(aliases, Mapping):
            raise RuntimeError("任务 Profile 的 aliases 必须是映射")
        for name, candidates in aliases.items():
            if not isinstance(name, str) or not isinstance(candidates, (str, list, tuple)):
                raise RuntimeError("aliases 必须使用 canonical: [alias1, alias2] 格式")
        identifiers = self.settings.get("identifier_fields", [])
        if not isinstance(identifiers, (list, tuple)):
            raise RuntimeError("identifier_fields 必须是字段名列表")
        identifier_rules = self.settings.get("identifier_rules", {})
        if not isinstance(identifier_rules, Mapping):
            raise RuntimeError("identifier_rules 必须是映射")
        unknown_rule_fields = set(str(name) for name in identifier_rules) - set(
            str(name) for name in identifiers
        )
        if unknown_rule_fields:
            raise RuntimeError(
                "identifier_rules 包含未列入 identifier_fields 的字段: "
                + ", ".join(sorted(unknown_rule_fields))
            )
        for name, rule in identifier_rules.items():
            if not isinstance(rule, Mapping):
                raise RuntimeError(f"identifier_rules.{name} 必须是映射")
            configured = sum(
                rule.get(key) not in (None, "", []) for key in ("equals", "allowed", "regex")
            )
            if configured != 1:
                raise RuntimeError(
                    f"identifier_rules.{name} 必须且只能配置 equals、allowed 或 regex"
                )
            if rule.get("allowed") is not None and not isinstance(
                rule.get("allowed"), (list, tuple, set)
            ):
                raise RuntimeError(f"identifier_rules.{name}.allowed 必须是列表")
            if rule.get("regex") not in (None, ""):
                try:
                    re.compile(str(rule["regex"]))
                except re.error as exc:
                    raise RuntimeError(f"identifier_rules.{name}.regex 无效: {exc}") from exc
        status = self.settings.get("status", {})
        if not isinstance(status, Mapping):
            raise RuntimeError("status 必须是映射")
        limits = self.settings.get("measurement_limits", {})
        if not isinstance(limits, Mapping):
            raise RuntimeError("measurement_limits 必须是映射")
        for name, rule in limits.items():
            if not isinstance(rule, Mapping):
                raise RuntimeError(f"测量字段 {name} 必须配置 min/max")
            for bound in ("min", "max"):
                if rule.get(bound) is not None:
                    try:
                        float(rule[bound])
                    except (TypeError, ValueError) as exc:
                        raise RuntimeError(f"测量字段 {name}.{bound} 必须是数值") from exc
        route = self.settings.get("route", {})
        if not isinstance(route, Mapping) or not isinstance(route.get("map", {}), Mapping):
            raise RuntimeError("route 及 route.map 必须是映射")
        for section_name in ("pick", "place"):
            section = self.settings.get(section_name, {})
            if not isinstance(section, Mapping):
                raise RuntimeError(f"{section_name} 必须是映射")
            fields = section.get("fields", [])
            if not isinstance(fields, (list, tuple)) or len(fields) > 6:
                raise RuntimeError(f"{section_name}.fields 必须是最多 6 项的列表")
            if fields:
                defaults = section.get("defaults", [])
                if not isinstance(defaults, (list, tuple)) or len(defaults) != 6:
                    raise RuntimeError(f"{section_name}.defaults 必须包含 6 个位姿值")
            if section.get("pose") is not None:
                self._validate_pose(section["pose"], section_name)
            offsets = section.get("offsets")
            if offsets is not None and (not isinstance(offsets, (list, tuple)) or len(offsets) != 6):
                raise RuntimeError(f"{section_name}.offsets 必须包含 6 个值")
            route_poses = section.get("route_poses", {})
            if not isinstance(route_poses, Mapping):
                raise RuntimeError(f"{section_name}.route_poses 必须是映射")
            for route_name, pose in route_poses.items():
                self._validate_pose(pose, f"{section_name}.route_poses.{route_name}")
            route_slots = section.get("route_slots", {})
            if not isinstance(route_slots, Mapping):
                raise RuntimeError(f"{section_name}.route_slots 必须是映射")
            for route_name, slots in route_slots.items():
                if not isinstance(slots, (list, tuple)) or not slots:
                    raise RuntimeError(f"{section_name}.route_slots.{route_name} 必须是非空位姿列表")
                for index, pose in enumerate(slots, 1):
                    self._validate_pose(pose, f"{section_name}.route_slots.{route_name}[{index}]")
            route_stacks = section.get("route_stacks", {})
            if not isinstance(route_stacks, Mapping):
                raise RuntimeError(f"{section_name}.route_stacks 必须是映射")
            for route_name, stack in route_stacks.items():
                if not isinstance(stack, Mapping):
                    raise RuntimeError(f"{section_name}.route_stacks.{route_name} 必须是映射")
                self._validate_pose(
                    stack.get("base_pose"),
                    f"{section_name}.route_stacks.{route_name}.base_pose",
                )
                try:
                    layer_height = float(stack.get("layer_height_mm"))
                    max_layers = int(stack.get("max_layers"))
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"{section_name}.route_stacks.{route_name} 需要 layer_height_mm 和 max_layers"
                    ) from exc
                if layer_height <= 0 or max_layers <= 0:
                    raise RuntimeError(
                        f"{section_name}.route_stacks.{route_name} 的层高和最大层数必须大于 0"
                    )

    def normalize(self, data: Any) -> VisionTaskResult:
        if not isinstance(data, Mapping):
            raise RuntimeError(
                "DVS 结果不是字段对象，请在 vision_studio.protocol.response_fields 中配置返回字段"
            )
        fields = {str(key): value for key, value in data.items()}
        for canonical, candidates in self.settings.get("aliases", {}).items():
            if canonical in fields:
                continue
            names = [candidates] if isinstance(candidates, str) else list(candidates or [])
            for name in names:
                if str(name) in fields:
                    fields[str(canonical)] = fields[str(name)]
                    break
        return VisionTaskResult(fields=fields)

    def require_fields(self, result: VisionTaskResult, field_names: list[str]) -> None:
        missing = [name for name in field_names if self._is_missing(result.fields.get(name))]
        if missing:
            raise RuntimeError("DVS 结果缺少必需字段: " + ", ".join(missing))

    def evaluate_measurements(self, result: VisionTaskResult) -> VisionTaskResult:
        limits = self.settings.get("measurement_limits", {})
        if not isinstance(limits, Mapping) or not limits:
            raise RuntimeError("当前视觉任务未配置 measurement_limits，不能执行尺寸公差判定")
        self.require_fields(result, [str(name) for name in limits])
        violations: list[str] = []
        for name, rule in limits.items():
            if not isinstance(rule, Mapping):
                raise RuntimeError(f"测量字段 {name} 的限制必须是 min/max 对象")
            try:
                value = float(result.fields[str(name)])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"测量字段 {name} 不是数值: {result.fields.get(str(name))}") from exc
            minimum = rule.get("min")
            maximum = rule.get("max")
            if minimum is not None and value < float(minimum):
                violations.append(f"{name}={value:g} < {float(minimum):g}")
            if maximum is not None and value > float(maximum):
                violations.append(f"{name}={value:g} > {float(maximum):g}")
        decision = "NG" if violations or result.decision == "NG" else "OK"
        return VisionTaskResult(
            fields=result.fields,
            decision=decision,
            route=result.route,
            violations=tuple(violations),
            identifiers=result.identifiers,
        )

    def evaluate_status(self, result: VisionTaskResult) -> VisionTaskResult:
        status_cfg = self.settings.get("status", {})
        field_name = str(status_cfg.get("field", "status"))
        self.require_fields(result, [field_name])
        raw = result.fields[field_name]
        ok_values = status_cfg.get("ok_values", ["OK", "PASS", 1, True])
        ng_values = status_cfg.get("ng_values", ["NG", "FAIL", 0, False])
        if self._matches_any(raw, ok_values):
            decision = "OK"
        elif self._matches_any(raw, ng_values):
            decision = "NG"
        else:
            raise RuntimeError(f"无法把状态字段 {field_name}={raw!r} 映射为 OK/NG")
        return VisionTaskResult(
            fields=result.fields,
            decision=decision,
            route=result.route,
            violations=result.violations,
            identifiers=result.identifiers,
        )

    def validate_identifiers(self, result: VisionTaskResult) -> VisionTaskResult:
        names = [str(name) for name in self.settings.get("identifier_fields", [])]
        if not names:
            raise RuntimeError("当前视觉任务未配置 identifier_fields")
        self.require_fields(result, names)
        identifiers = {name: result.fields[name] for name in names}
        violations = list(result.violations)
        rules = self.settings.get("identifier_rules", {})
        for name, rule in rules.items():
            field_name = str(name)
            self.require_fields(result, [field_name])
            raw = result.fields[field_name]
            case_sensitive = bool(rule.get("case_sensitive", False))
            if rule.get("equals") is not None:
                matched = self._text_equal(raw, rule["equals"], case_sensitive)
                description = f"应等于 {rule['equals']!r}"
            elif rule.get("allowed") is not None:
                matched = any(
                    self._text_equal(raw, candidate, case_sensitive)
                    for candidate in rule["allowed"]
                )
                description = f"不在允许列表 {list(rule['allowed'])!r}"
            else:
                flags = 0 if case_sensitive else re.IGNORECASE
                matched = re.fullmatch(str(rule["regex"]), str(raw), flags=flags) is not None
                description = f"不匹配正则 {rule['regex']!r}"
            if not matched:
                violations.append(f"{field_name}={raw!r} {description}")
        decision = result.decision
        if rules:
            decision = "NG" if violations or result.decision == "NG" else "OK"
        return VisionTaskResult(
            fields=result.fields,
            decision=decision,
            route=result.route,
            violations=tuple(violations),
            identifiers=identifiers,
        )

    def resolve_route(self, result: VisionTaskResult) -> VisionTaskResult:
        route_cfg = self.settings.get("route", {})
        mapping = route_cfg.get("map", {})
        if not isinstance(mapping, Mapping) or not mapping:
            raise RuntimeError("当前视觉任务未配置 route.map")
        field_name = str(route_cfg.get("field", "decision"))
        raw = result.decision if field_name == "decision" else result.fields.get(field_name)
        if raw is None:
            raise RuntimeError(f"路由字段 {field_name} 为空")
        route = self._mapping_lookup(mapping, raw)
        if route is None:
            route = route_cfg.get("default")
        if route is None:
            raise RuntimeError(f"路由字段 {field_name}={raw!r} 没有对应动作")
        return VisionTaskResult(
            fields=result.fields,
            decision=result.decision,
            route=str(route),
            violations=result.violations,
            identifiers=result.identifiers,
        )

    def resolve_pick_pose(self, result: VisionTaskResult) -> list[float]:
        return self._resolve_pose(self.settings.get("pick", {}), result.fields, "抓取")

    def resolve_place_pose(
        self,
        result: VisionTaskResult,
        placement_index: int = 0,
    ) -> list[float]:
        place_cfg = self.settings.get("place", {})
        route_key = result.route or "__default__"
        route_slots = place_cfg.get("route_slots", {})
        if route_slots:
            slots = self._mapping_lookup(route_slots, route_key)
            if slots is None:
                raise RuntimeError(f"未配置路线 {route_key} 的顺序放置槽位")
            if placement_index >= len(slots):
                raise RuntimeError(
                    f"路线 {route_key} 的放置槽位已用完: index={placement_index} capacity={len(slots)}"
                )
            return self._validate_pose(slots[placement_index], f"路线 {route_key} 第 {placement_index + 1} 个槽位")
        route_stacks = place_cfg.get("route_stacks", {})
        if route_stacks:
            stack = self._mapping_lookup(route_stacks, route_key)
            if stack is None:
                raise RuntimeError(f"未配置路线 {route_key} 的堆叠参数")
            max_layers = int(stack["max_layers"])
            if placement_index >= max_layers:
                raise RuntimeError(
                    f"路线 {route_key} 的堆叠层数已满: index={placement_index} capacity={max_layers}"
                )
            pose = self._validate_pose(stack["base_pose"], f"路线 {route_key} 堆叠基准")
            pose[2] += placement_index * float(stack["layer_height_mm"])
            return pose
        route_poses = place_cfg.get("route_poses", {})
        if route_poses:
            if not result.route:
                raise RuntimeError("放置点按 route_poses 配置，但尚未执行结果路由")
            pose = self._mapping_lookup(route_poses, result.route)
            if pose is None:
                raise RuntimeError(f"未配置路由 {result.route} 的放置坐标")
            return self._validate_pose(pose, f"路由 {result.route} 放置")
        return self._resolve_pose(place_cfg, result.fields, "放置")

    def _resolve_pose(self, config: Any, fields: Mapping[str, Any], name: str) -> list[float]:
        if not isinstance(config, Mapping):
            raise RuntimeError(f"{name}点配置必须是对象")
        if config.get("pose") is not None:
            return self._validate_pose(config["pose"], name)
        field_names = list(config.get("fields", []))
        defaults = list(config.get("defaults", []))
        if not field_names:
            raise RuntimeError(f"当前视觉任务未配置 {name}点 fields 或 pose")
        if len(field_names) > 6 or len(defaults) != 6:
            raise RuntimeError(f"{name}点要求 defaults 为 6 轴位姿，fields 最多 6 项")
        pose = [float(value) for value in defaults]
        for index, field_name in enumerate(field_names):
            key = str(field_name)
            if self._is_missing(fields.get(key)):
                raise RuntimeError(f"DVS 结果缺少{name}坐标字段: {key}")
            pose[index] = float(fields[key])
        offsets = list(config.get("offsets", [0, 0, 0, 0, 0, 0]))
        if len(offsets) != 6:
            raise RuntimeError(f"{name}点 offsets 必须包含 6 个值")
        return [value + float(offsets[index]) for index, value in enumerate(pose)]

    @staticmethod
    def _validate_pose(value: Any, name: str) -> list[float]:
        if not isinstance(value, (list, tuple)) or len(value) != 6:
            raise RuntimeError(f"{name}点必须包含 [x, y, z, rx, ry, rz] 6 个值")
        return [float(item) for item in value]

    @staticmethod
    def _mapping_lookup(mapping: Mapping[Any, Any], value: Any) -> Any:
        try:
            if value in mapping:
                return mapping[value]
        except TypeError:
            pass
        target = str(value).strip().casefold()
        for key, mapped in mapping.items():
            if str(key).strip().casefold() == target:
                return mapped
        return None

    @staticmethod
    def _matches_any(value: Any, candidates: Any) -> bool:
        values = candidates if isinstance(candidates, (list, tuple, set)) else [candidates]
        target = str(value).strip().casefold()
        return any(str(candidate).strip().casefold() == target for candidate in values)

    @staticmethod
    def _text_equal(left: Any, right: Any, case_sensitive: bool) -> bool:
        left_text = str(left).strip()
        right_text = str(right).strip()
        if case_sensitive:
            return left_text == right_text
        return left_text.casefold() == right_text.casefold()

    @staticmethod
    def _is_missing(value: Any) -> bool:
        return value is None or (isinstance(value, str) and not value.strip())
