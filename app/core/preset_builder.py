from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from app.core.vision_task import VisionTaskProcessor


TASK_TYPES: dict[str, dict[str, Any]] = {
    "communication": {
        "name": "仅通信/读取结果",
        "processor": "none",
        "modules": ["vision_studio.exchange", "vision_result.normalize"],
    },
    "defect_sort": {
        "name": "缺陷或状态 OK/NG 分拣",
        "processor": "status",
        "route": True,
        "modules": ["vision_result.quality", "vision_result.route"],
    },
    "category_sort": {
        "name": "颜色/类别/型号分拣",
        "processor": "none",
        "route": True,
        "modules": ["vision_result.route"],
    },
    "measurement": {
        "name": "尺寸测量与公差判定",
        "processor": "measurement",
        "modules": ["vision_result.measure"],
    },
    "measurement_sort": {
        "name": "尺寸公差判定后分拣",
        "processor": "measurement",
        "route": True,
        "modules": ["vision_result.measure", "vision_result.route"],
    },
    "recognition": {
        "name": "OCR/条码/二维码识别",
        "processor": "none",
        "identifiers": True,
        "modules": ["vision_result.identify"],
    },
    "recognition_sort": {
        "name": "OCR/条码/二维码核验后分拣",
        "processor": "none",
        "identifiers": True,
        "route": True,
        "modules": ["vision_result.identify", "vision_result.route"],
    },
    "guided_assembly": {
        "name": "视觉定位引导抓取/装配",
        "processor": "none",
        "motion": True,
        "modules": ["vision_result.plan_motion"],
    },
    "fixed_inspection": {
        "name": "固定取料送检后按结果分流",
        "processor": "status",
        "route": True,
        "motion": True,
        "before_sequence": "inspection_feed_template",
        "modules": ["task.dvs_cycle"],
    },
    "combined": {
        "name": "综合自定义任务（多能力组合）",
        "processor": "status_measurement",
        "identifiers": True,
        "route": True,
        "modules": ["vision_result.normalize"],
    },
}


@dataclass(frozen=True)
class BuiltTaskPreset:
    profile_name: str
    recipe_id: str
    recipe_name: str
    updates: dict[str, Any]
    module_ids: tuple[str, ...]


def safe_identifier(value: str, prefix: str = "field_task") -> str:
    text = re.sub(r"[^0-9A-Za-z_]+", "_", value.strip()).strip("_").lower()
    return text or prefix


def build_task_preset(spec: Mapping[str, Any], robot_config: Mapping[str, Any]) -> BuiltTaskPreset:
    task_type = str(spec.get("task_type") or "")
    if task_type not in TASK_TYPES:
        raise ValueError(f"不支持的任务类型: {task_type or '<empty>'}")
    template = TASK_TYPES[task_type]
    profile_name = str(spec.get("profile_name") or "").strip()
    recipe_name = str(spec.get("recipe_name") or "").strip()
    if not profile_name:
        raise ValueError("任务 Profile 名称不能为空")
    if not recipe_name:
        raise ValueError("现场配方名称不能为空")

    aliases = _clean_aliases(spec.get("aliases", {}))
    profile: dict[str, Any] = {"aliases": aliases}
    processor = str(spec.get("processor", template.get("processor", "none")))
    if processor not in {"none", "status", "measurement", "status_measurement"}:
        raise ValueError(f"不支持的结果处理方式: {processor}")
    if processor in {"status", "status_measurement"}:
        status = dict(spec.get("status", {}))
        field = str(status.get("field") or "status").strip()
        ok_values = _clean_values(status.get("ok_values", []))
        ng_values = _clean_values(status.get("ng_values", []))
        if not ok_values or not ng_values:
            raise ValueError("OK 值和 NG 值都不能为空")
        profile["status"] = {
            "field": field,
            "ok_values": ok_values,
            "ng_values": ng_values,
        }
    if processor in {"measurement", "status_measurement"}:
        limits = _clean_measurements(spec.get("measurement_limits", {}))
        if not limits:
            raise ValueError("尺寸题至少要填写一个测量字段及 min/max")
        profile["measurement_limits"] = limits

    validate_identifiers = bool(spec.get("validate_identifiers", template.get("identifiers")))
    identifiers = [str(item).strip() for item in spec.get("identifier_fields", []) if str(item).strip()]
    if validate_identifiers:
        if not identifiers:
            raise ValueError("识别题至少选择一个 OCR/条码/二维码字段")
        profile["identifier_fields"] = identifiers
        identifier_rules = _clean_identifier_rules(spec.get("identifier_rules", {}))
        if identifier_rules:
            profile["identifier_rules"] = identifier_rules
        if task_type == "recognition_sort" and not identifier_rules:
            raise ValueError("识别后分拣至少要填写一条期望值、允许列表或正则规则")

    route_enabled = bool(spec.get("route_enabled", template.get("route")))
    route_map = _clean_mapping(spec.get("route_map", {}))
    if route_enabled:
        if not route_map:
            raise ValueError("分拣题至少要填写一条“结果值 -> 路线”映射")
        profile["route"] = {
            "field": str(spec.get("route_field") or "decision").strip(),
            "map": route_map,
        }

    motion_enabled = bool(spec.get("execute_motion", template.get("motion", False)))
    pick = _clean_pose_section(spec.get("pick", {}), "抓取")
    place = _clean_pose_section(spec.get("place", {}), "放置")
    placement_mode = str(spec.get("placement_mode") or "fixed")
    placement_rows = list(spec.get("placements", []))
    if motion_enabled:
        if not pick:
            raise ValueError("启用机器人动作时必须配置抓取坐标字段或固定抓取位姿")
        profile["pick"] = pick
        if route_enabled:
            _apply_route_placements(
                place,
                route_map,
                placement_mode,
                placement_rows,
                spec,
                robot_config,
            )
        elif not place:
            raise ValueError("启用机器人动作时必须配置放置坐标字段或固定放置位姿")
        profile["place"] = place

    VisionTaskProcessor(profile).validate_settings()
    _validate_static_pose_safety(profile, robot_config)

    use_cycle = bool(spec.get("use_cycle", True))
    if use_cycle:
        module_ids = ["task.dvs_cycle"]
    else:
        module_ids = []
        if processor in {"status", "status_measurement"}:
            module_ids.append("vision_result.quality")
        if processor in {"measurement", "status_measurement"}:
            module_ids.append("vision_result.measure")
        if validate_identifiers:
            module_ids.append("vision_result.identify")
        if route_enabled:
            module_ids.append("vision_result.route")
        if motion_enabled:
            module_ids.append("robot.result_pick_place")
        if not module_ids:
            module_ids = ["vision_studio.exchange", "vision_result.normalize"]

    before_sequence = str(
        spec.get("before_each_sequence") or template.get("before_sequence") or ""
    ).strip() or None
    after_sequence = str(spec.get("after_each_sequence") or "").strip() or None
    cycle = {
        "max_items": int(spec.get("max_items", 1)),
        "delay_s": float(spec.get("delay_s", 0.2)),
        "processor": processor,
        "validate_identifiers": validate_identifiers,
        "route": route_enabled,
        "execute_motion": motion_enabled,
        "before_each_sequence": before_sequence,
        "after_each_sequence": after_sequence,
        "reset_placement_counters_on_start": True,
        "empty": {
            "field": str(spec.get("empty_field") or "").strip() or None,
            "values": _clean_values(spec.get("empty_values", ["EMPTY", "NONE", "NO_TARGET"])),
        },
        "on_error": "stop" if motion_enabled or before_sequence or after_sequence else str(spec.get("on_error", "stop")),
    }
    if cycle["max_items"] < 1 or cycle["max_items"] > 100:
        raise ValueError("最多处理件数必须在 1~100 之间")
    if cycle["delay_s"] < 0 or cycle["delay_s"] > 30:
        raise ValueError("每件间隔必须在 0~30 秒之间")

    response_fields = [
        str(item).strip() for item in spec.get("response_fields", []) if str(item).strip()
    ]
    if len(response_fields) != len(set(response_fields)):
        raise ValueError("DVS 返回字段顺序不能包含重复字段")
    _validate_response_contract(profile, response_fields, processor, route_enabled, motion_enabled)
    recipe_id = safe_identifier(str(spec.get("recipe_id") or profile_name))
    updates: dict[str, Any] = {
        "vision_tasks": {
            "active_profile": profile_name,
            "profiles": {profile_name: profile},
        },
        "dvs_cycle": cycle,
        "recipes": {
            recipe_id: {"name": recipe_name, "modules": module_ids},
        },
        "modules": {"default_recipe": recipe_id},
    }
    if response_fields:
        updates["vision_studio"] = {"protocol": {"response_fields": response_fields}}
    return BuiltTaskPreset(profile_name, recipe_id, recipe_name, updates, tuple(module_ids))


def _clean_aliases(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, Mapping):
        raise ValueError("字段别名必须是映射")
    output: dict[str, list[str]] = {}
    for canonical, candidates in value.items():
        name = str(canonical).strip()
        if not name:
            continue
        cleaned = _clean_values(candidates)
        output[name] = [str(item) for item in cleaned if str(item) != name]
    return output


def _clean_values(value: Any) -> list[Any]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return [item for item in values if not (isinstance(item, str) and not item.strip())]


def _clean_mapping(value: Any) -> dict[Any, str]:
    if not isinstance(value, Mapping):
        raise ValueError("路线映射必须是映射")
    return {
        key: str(route).strip()
        for key, route in value.items()
        if str(key).strip() and str(route).strip()
    }


def _clean_identifier_rules(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError("识别核验规则必须是映射")
    output: dict[str, dict[str, Any]] = {}
    for field, rule in value.items():
        name = str(field).strip()
        if not name or not isinstance(rule, Mapping):
            continue
        mode = str(rule.get("mode") or "").strip().lower()
        expected = rule.get("expected")
        if expected in (None, ""):
            continue
        if mode == "equals":
            parsed = {"equals": expected}
        elif mode == "allowed":
            values = _clean_values(expected)
            if not values:
                raise ValueError(f"识别字段 {name} 的允许列表不能为空")
            parsed = {"allowed": values}
        elif mode == "regex":
            parsed = {"regex": str(expected)}
        else:
            raise ValueError(f"识别字段 {name} 的核验方式必须是 equals、allowed 或 regex")
        parsed["case_sensitive"] = bool(rule.get("case_sensitive", False))
        output[name] = parsed
    return output


def _clean_measurements(value: Any) -> dict[str, dict[str, float | None]]:
    if not isinstance(value, Mapping):
        raise ValueError("测量限制必须是映射")
    output: dict[str, dict[str, float | None]] = {}
    for field, rule in value.items():
        name = str(field).strip()
        if not name or not isinstance(rule, Mapping):
            continue
        minimum = rule.get("min")
        maximum = rule.get("max")
        if minimum in (None, "") and maximum in (None, ""):
            continue
        parsed_min = None if minimum in (None, "") else float(minimum)
        parsed_max = None if maximum in (None, "") else float(maximum)
        if parsed_min is not None and parsed_max is not None and parsed_min > parsed_max:
            raise ValueError(f"测量字段 {name} 的 min 不能大于 max")
        output[name] = {"min": parsed_min, "max": parsed_max}
    return output


def _clean_pose_section(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name}配置必须是映射")
    fields = [str(item).strip() for item in value.get("fields", []) if str(item).strip()]
    pose = value.get("pose")
    output: dict[str, Any] = {}
    if fields:
        if len(fields) > 6:
            raise ValueError(f"{name}坐标字段最多 6 个")
        defaults = _pose(value.get("defaults", [0, 0, 0, 180, 0, 0]), f"{name}默认位姿")
        output.update({"fields": fields, "defaults": defaults})
    elif pose not in (None, [], ""):
        output["pose"] = _pose(pose, f"固定{name}位姿")
    offsets = value.get("offsets")
    if output and offsets not in (None, [], ""):
        output["offsets"] = _pose(offsets, f"{name}偏移")
    return output


def _apply_route_placements(
    place: dict[str, Any],
    route_map: Mapping[Any, str],
    mode: str,
    rows: list[Any],
    spec: Mapping[str, Any],
    robot_config: Mapping[str, Any],
) -> None:
    required_routes = set(route_map.values())
    parsed: dict[str, list[list[float]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        route = str(row.get("route") or "").strip()
        if route:
            parsed.setdefault(route, []).append(_pose(row.get("pose"), f"路线 {route} 放置位姿"))
    missing = sorted(required_routes - set(parsed))
    if missing:
        raise ValueError("启用机器人分拣时缺少路线放置位姿: " + ", ".join(missing))
    if mode == "fixed":
        place.clear()
        place["route_poses"] = {route: poses[0] for route, poses in parsed.items()}
    elif mode == "slots":
        place.clear()
        place["route_slots"] = parsed
    elif mode == "stack":
        layer_height = float(spec.get("layer_height_mm", 0))
        max_layers = int(spec.get("max_layers", 0))
        if layer_height <= 0 or max_layers <= 0:
            raise ValueError("堆叠模式的层高和最大层数必须大于 0")
        place.clear()
        place["route_stacks"] = {
            route: {
                "base_pose": poses[0],
                "layer_height_mm": layer_height,
                "max_layers": max_layers,
            }
            for route, poses in parsed.items()
        }
    else:
        raise ValueError(f"不支持的放置方式: {mode}")
    minimum_z = float(robot_config.get("min_grasp_z_mm", 60.0))
    for route, poses in parsed.items():
        if any(pose[2] < minimum_z for pose in poses):
            raise ValueError(f"路线 {route} 放置 Z 低于安全下限 {minimum_z:g}mm")


def _pose(value: Any, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise ValueError(f"{name}必须包含 X,Y,Z,Rx,Ry,Rz 六个数值")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}必须全部为数值") from exc


def _validate_static_pose_safety(profile: Mapping[str, Any], robot_config: Mapping[str, Any]) -> None:
    minimum_z = float(robot_config.get("min_grasp_z_mm", 60.0))
    for section_name in ("pick", "place"):
        section = profile.get(section_name, {})
        if not isinstance(section, Mapping):
            continue
        pose = section.get("pose")
        if pose is not None and float(pose[2]) < minimum_z:
            raise ValueError(f"固定{section_name}位姿 Z 低于安全下限 {minimum_z:g}mm")
        fields = section.get("fields", [])
        defaults = section.get("defaults", [])
        if fields and len(fields) < 3 and len(defaults) == 6 and float(defaults[2]) < minimum_z:
            raise ValueError(
                f"{section_name}未从 DVS 读取 Z，默认 Z 低于安全下限 {minimum_z:g}mm"
            )


def _validate_response_contract(
    profile: Mapping[str, Any],
    response_fields: list[str],
    processor: str,
    route_enabled: bool,
    motion_enabled: bool,
) -> None:
    # Empty means JSON/object reply, whose keys are not known until runtime.
    if not response_fields:
        return
    aliases = profile.get("aliases", {})
    available = set(response_fields)
    for canonical, candidates in aliases.items():
        candidate_list = candidates if isinstance(candidates, (list, tuple)) else [candidates]
        if str(canonical) in available or any(str(item) in available for item in candidate_list):
            available.add(str(canonical))
    required: set[str] = set()
    if processor in {"status", "status_measurement"}:
        required.add(str(profile.get("status", {}).get("field", "status")))
    if processor in {"measurement", "status_measurement"}:
        required.update(str(name) for name in profile.get("measurement_limits", {}))
    required.update(str(name) for name in profile.get("identifier_fields", []))
    if route_enabled:
        route_field = str(profile.get("route", {}).get("field", "decision"))
        if route_field != "decision":
            required.add(route_field)
    if motion_enabled:
        required.update(str(name) for name in profile.get("pick", {}).get("fields", []))
        required.update(str(name) for name in profile.get("place", {}).get("fields", []))
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            "DVS 返回字段顺序无法提供任务必需字段: "
            + ", ".join(missing)
            + "；请补充返回字段或字段别名"
        )
