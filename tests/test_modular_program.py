from __future__ import annotations

import unittest
import socket
from copy import deepcopy
from pathlib import Path
from threading import Event, Thread

from app.config import load_config, update_yaml_config
from app.core.action_sequence import ActionSequencePlanner
from app.core.command_transport import JsonCommandTransport
from app.core.preset_builder import build_task_preset
from app.core.workflow import CompetitionWorkflow
from app.core.vision_studio_client import DobotVisionStudioClient
from app.core.vision_task import VisionTaskProcessor, VisionTaskResult
from app.field_config_dialog import escape_control_text, unescape_control_text
from tools.mock_dvs_server import decode_escapes
from app.modules import FeatureModule, ModuleExecutor, ModuleRegistry, build_standard_registry


ROOT = Path(__file__).resolve().parents[1]


class FakeRobot:
    def __init__(self):
        self.cancel_event = Event()


class FakeWorkflow:
    def __init__(self):
        self.robot = FakeRobot()
        self.calls: list[str] = []


class ModuleRegistryTests(unittest.TestCase):
    def test_standard_module_automatically_resolves_dependencies(self):
        registry = build_standard_registry()
        resolved = [
            module.module_id
            for module in registry.resolve(["robot.pick_one"], include_dependencies=True)
        ]
        self.assertEqual(
            resolved,
            [
                "robot.connect",
                "camera.start",
                "vision.detect",
                "target.calculate",
                "robot.pick_one",
            ],
        )

    def test_dvs_motion_is_planned_before_robot_connection(self):
        registry = build_standard_registry()
        resolved = [
            module.module_id
            for module in registry.resolve(["robot.result_pick_place"], include_dependencies=True)
        ]
        self.assertLess(
            resolved.index("vision_result.plan_motion"),
            resolved.index("robot.connect"),
        )

    def test_executor_skips_ready_dependency_but_repeats_explicit_module(self):
        workflow = FakeWorkflow()
        registry = ModuleRegistry()
        registry.register(
            FeatureModule("prepare", "准备", "test", lambda w: w.calls.append("prepare"), cacheable=True)
        )
        registry.register(
            FeatureModule(
                "work",
                "执行",
                "test",
                lambda w: w.calls.append("work"),
                ("prepare",),
            )
        )
        executor = ModuleExecutor(workflow, registry, logger=lambda _message: None)
        executor.execute(["work"])
        executor.execute(["work"])
        self.assertEqual(workflow.calls, ["prepare", "work", "work"])

    def test_executor_stops_when_cancelled(self):
        workflow = FakeWorkflow()
        workflow.robot.cancel_event.set()
        registry = ModuleRegistry()
        registry.register(
            FeatureModule("work", "执行", "test", lambda w: w.calls.append("work"))
        )
        executor = ModuleExecutor(workflow, registry, logger=lambda _message: None)
        with self.assertRaisesRegex(RuntimeError, "已取消"):
            executor.execute(["work"])
        self.assertEqual(workflow.calls, [])

    def test_executor_reports_success_and_failure_progress(self):
        workflow = FakeWorkflow()
        registry = ModuleRegistry()
        registry.register(
            FeatureModule("prepare", "准备", "test", lambda w: w.calls.append("prepare"))
        )

        def fail(_workflow):
            raise RuntimeError("测试失败")

        registry.register(FeatureModule("work", "执行", "test", fail, ("prepare",)))
        executor = ModuleExecutor(workflow, registry, logger=lambda _message: None)
        events = []

        with self.assertRaisesRegex(RuntimeError, "测试失败"):
            executor.execute(["work"], progress=lambda module_id, state: events.append((module_id, state)))

        self.assertEqual(
            events,
            [
                ("prepare", "running"),
                ("prepare", "success"),
                ("work", "running"),
                ("work", "failed"),
            ],
        )

    def test_standard_tcp_recipe_executes_end_to_end_in_simulation(self):
        config = load_config(ROOT / "configs" / "national.yaml")
        config["mode"] = "simulation"
        workflow = CompetitionWorkflow(config, logger=lambda _message: None)
        executor = ModuleExecutor(workflow, build_standard_registry(), logger=lambda _message: None)
        results = executor.execute(["network.send"])
        self.assertEqual(results[-1].module_id, "network.send")
        payload, reply = results[-1].value
        self.assertEqual(payload["cmd"], "tcp_grasp_sequence")
        self.assertTrue(reply.ok)
        self.assertEqual(reply.mode, "simulation")

    def test_defect_route_recipe_executes_from_dvs_fields_in_simulation(self):
        config = load_config(ROOT / "configs" / "national.yaml")
        config["mode"] = "simulation"
        config["vision_studio"]["simulation_result"] = {
            "quality": "NG",
            "robot_x": 300,
            "robot_y": 20,
            "robot_z": 100,
        }
        config["vision_studio"]["protocol"]["exchange_command"] = "RUN"
        config["vision_tasks"]["active_profile"] = "defect_test"
        config["vision_tasks"]["profiles"]["defect_test"] = {
            "aliases": {
                "status": ["quality"],
                "x": ["robot_x"],
                "y": ["robot_y"],
                "z": ["robot_z"],
            },
            "status": {"field": "status", "ok_values": ["OK"], "ng_values": ["NG"]},
            "route": {"field": "decision", "map": {"OK": "ok", "NG": "reject"}},
            "pick": {
                "fields": ["x", "y", "z"],
                "defaults": [0, 0, 0, 180, 0, 0],
            },
            "place": {"route_poses": {"reject": [450, 100, 100, 180, 0, 0]}},
        }
        workflow = CompetitionWorkflow(config, logger=lambda _message: None)
        executor = ModuleExecutor(workflow, build_standard_registry(), logger=lambda _message: None)
        results = executor.execute(
            ["vision_result.quality", "vision_result.route", "robot.result_pick_place"]
        )
        self.assertEqual(results[-1].module_id, "robot.result_pick_place")
        self.assertEqual(workflow.vision_task_result.decision, "NG")
        self.assertEqual(workflow.vision_task_result.route, "reject")
        for actual, expected in zip(workflow.robot.current_pose, config["robot"]["home_pose"]):
            self.assertAlmostEqual(actual, expected, places=2)

    def test_dvs_cycle_processes_configured_number_of_results(self):
        config = load_config(ROOT / "configs" / "national.yaml")
        config["mode"] = "simulation"
        config["vision_studio"]["protocol"]["exchange_command"] = "RUN"
        config["vision_studio"]["simulation_result"] = {"status": "OK"}
        config["dvs_cycle"].update(
            {
                "max_items": 3,
                "delay_s": 0,
                "processor": "status",
                "route": False,
                "execute_motion": False,
            }
        )
        workflow = CompetitionWorkflow(config, logger=lambda _message: None)
        stats = workflow.run_dvs_cycle()
        self.assertEqual(stats["completed"], 3)
        self.assertEqual(stats["attempts"], 3)
        self.assertEqual(stats["decision_counts"], {"OK": 3})
        self.assertEqual(len(stats["records"]), 3)
        self.assertEqual(workflow.vision_task_result.decision, "OK")

    def test_dvs_cycle_stops_on_configured_empty_signal(self):
        config = load_config(ROOT / "configs" / "national.yaml")
        config["mode"] = "simulation"
        config["vision_studio"]["protocol"]["exchange_command"] = "RUN"
        config["vision_studio"]["simulation_result"] = {"status": "NO_TARGET"}
        config["dvs_cycle"].update(
            {
                "max_items": 8,
                "delay_s": 0,
                "processor": "status",
                "empty": {"field": "status", "values": ["NO_TARGET"]},
            }
        )
        workflow = CompetitionWorkflow(config, logger=lambda _message: None)
        stats = workflow.run_dvs_cycle()
        self.assertEqual(stats["completed"], 0)
        self.assertEqual(stats["attempts"], 1)
        self.assertTrue(stats["stopped_empty"])

    def test_dvs_cycle_combines_status_measurement_and_identifier_rules(self):
        config = load_config(ROOT / "configs" / "national.yaml")
        config["mode"] = "simulation"
        config["vision_studio"]["protocol"]["exchange_command"] = "RUN"
        config["vision_studio"]["simulation_result"] = {
            "status": "NG",
            "width": 20.0,
            "ocr": "A01",
        }
        config["vision_tasks"]["profiles"]["combined_test"] = {
            "status": {"field": "status", "ok_values": ["OK"], "ng_values": ["NG"]},
            "measurement_limits": {"width": {"min": 19.8, "max": 20.2}},
            "identifier_fields": ["ocr"],
            "identifier_rules": {"ocr": {"equals": "A01"}},
        }
        config["vision_tasks"]["active_profile"] = "combined_test"
        config["dvs_cycle"].update(
            {
                "max_items": 1,
                "delay_s": 0,
                "processor": "status_measurement",
                "validate_identifiers": True,
                "route": False,
                "execute_motion": False,
            }
        )
        workflow = CompetitionWorkflow(config, logger=lambda _message: None)
        stats = workflow.run_dvs_cycle()
        self.assertEqual(workflow.vision_task_result.decision, "NG")
        self.assertEqual(stats["decision_counts"], {"NG": 1})

    def test_dvs_cycle_runs_fixed_sequence_before_each_detection(self):
        config = load_config(ROOT / "configs" / "national.yaml")
        config["mode"] = "simulation"
        config["vision_studio"]["protocol"]["exchange_command"] = "RUN"
        config["vision_studio"]["simulation_result"] = {"status": "OK"}
        config["action_sequences"] = {
            "active_profile": "before",
            "pose_bank": {"observe": [300, 0, 180, 180, 0, 0]},
            "profiles": {
                "before": {"steps": [{"action": "movj", "pose_ref": "observe"}, {"action": "home"}]}
            },
        }
        config["dvs_cycle"].update(
            {
                "max_items": 2,
                "delay_s": 0,
                "processor": "none",
                "before_each_sequence": "before",
                "after_each_sequence": None,
                "execute_motion": False,
                "on_error": "stop",
            }
        )
        workflow = CompetitionWorkflow(config, logger=lambda _message: None)
        stats = workflow.run_dvs_cycle()
        self.assertEqual(stats["completed"], 2)
        move_commands = [command for command in workflow.robot.sent_commands if command.startswith("MovJ")]
        self.assertEqual(len(move_commands), 4)

    def test_dvs_cycle_rejects_skip_policy_when_fixed_sequence_moves_robot(self):
        config = load_config(ROOT / "configs" / "national.yaml")
        config["mode"] = "simulation"
        config["dvs_cycle"].update(
            {"before_each_sequence": "inspection_feed_template", "on_error": "skip"}
        )
        workflow = CompetitionWorkflow(config, logger=lambda _message: None)
        with self.assertRaisesRegex(RuntimeError, "on_error 必须为 stop"):
            workflow.run_dvs_cycle()


class ConfigurationTests(unittest.TestCase):
    def test_national_profile_inherits_calibrated_provincial_values(self):
        provincial = load_config(ROOT / "configs" / "competition.yaml")
        national = load_config(ROOT / "configs" / "national.yaml")
        self.assertEqual(national["competition"]["level"], "national")
        self.assertEqual(national["robot"]["home_pose"], provincial["robot"]["home_pose"])
        self.assertEqual(national["robot"]["ip"], provincial["robot"]["ip"])
        self.assertEqual(national["robot"]["dashboard_port"], provincial["robot"]["dashboard_port"])
        self.assertEqual(
            national["calibration"]["transform_camera_to_gripper"],
            provincial["calibration"]["transform_camera_to_gripper"],
        )
        self.assertIn(national["modules"]["default_recipe"], national["recipes"])
        self.assertEqual(national["vision_studio"]["mode"], "disabled")
        self.assertIsNone(national["vision_studio"]["tcp"]["host"])
        self.assertIsNone(national["vision_studio"]["tcp"]["port"])

    def test_field_profile_inherits_national_and_preserves_robot_endpoint(self):
        national = load_config(ROOT / "configs" / "national.yaml")
        field = load_config(ROOT / "configs" / "field.yaml")
        self.assertEqual(field["robot"]["ip"], national["robot"]["ip"])
        self.assertEqual(field["robot"]["dashboard_port"], national["robot"]["dashboard_port"])
        self.assertEqual(field["competition"]["level"], "national")

    def test_field_override_update_merges_without_dropping_existing_sections(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            path = Path(directory) / "field.yaml"
            update_yaml_config(
                path,
                {"vision_studio": {"mode": "serial"}, "dvs_cycle": {"max_items": 3}},
                extends="national.yaml",
            )
            update_yaml_config(
                path,
                {"vision_studio": {"serial": {"port": "COM7"}}},
                extends="national.yaml",
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn("extends: national.yaml", text)
            self.assertIn("max_items: 3", text)
            self.assertIn("port: COM7", text)

    def test_control_character_editor_round_trip(self):
        value = "\\prefix\r\n\t"
        self.assertEqual(unescape_control_text(escape_control_text(value)), value)
        self.assertEqual(decode_escapes(r"\r\n"), "\r\n")

    def test_all_national_vision_profiles_are_structurally_valid(self):
        config = load_config(ROOT / "configs" / "national.yaml")
        for name, profile in config["vision_tasks"]["profiles"].items():
            with self.subTest(profile=name):
                VisionTaskProcessor(profile)

    def test_national_action_sequence_templates_are_structurally_valid(self):
        config = load_config(ROOT / "configs" / "national.yaml")
        ActionSequencePlanner(config["action_sequences"], config["robot"])


class PresetBuilderTests(unittest.TestCase):
    def setUp(self):
        self.robot = load_config(ROOT / "configs" / "national.yaml")["robot"]

    def test_defect_wizard_builds_profile_cycle_and_recipe(self):
        result = build_task_preset(
            {
                "task_type": "defect_sort",
                "profile_name": "field_defect",
                "recipe_name": "现场缺陷分拣",
                "response_fields": ["status", "x", "y", "z"],
                "aliases": {"status": ["quality"], "x": ["robot_x"]},
                "status": {
                    "field": "status",
                    "ok_values": ["OK"],
                    "ng_values": ["NG"],
                },
                "route_map": {"OK": "ok", "NG": "reject"},
                "execute_motion": False,
                "use_cycle": True,
                "max_items": 6,
            },
            self.robot,
        )
        profile = result.updates["vision_tasks"]["profiles"]["field_defect"]
        self.assertEqual(profile["status"]["field"], "status")
        self.assertEqual(profile["route"]["map"]["NG"], "reject")
        self.assertEqual(result.updates["dvs_cycle"]["processor"], "status")
        self.assertEqual(result.module_ids, ("task.dvs_cycle",))
        self.assertEqual(
            result.updates["vision_studio"]["protocol"]["response_fields"],
            ["status", "x", "y", "z"],
        )

    def test_slot_wizard_groups_repeated_route_rows(self):
        result = build_task_preset(
            {
                "task_type": "category_sort",
                "profile_name": "field_slots",
                "recipe_name": "现场多槽位",
                "aliases": {"label": ["color"]},
                "route_field": "label",
                "route_map": {"red": "red_bin", "green": "green_bin"},
                "execute_motion": True,
                "pick": {
                    "fields": ["x", "y", "z"],
                    "defaults": [0, 0, 0, 180, 0, 0],
                    "offsets": [0, 0, 0, 0, 0, 0],
                },
                "placement_mode": "slots",
                "placements": [
                    {"route": "red_bin", "pose": [300, 100, 100, 180, 0, 0]},
                    {"route": "red_bin", "pose": [330, 100, 100, 180, 0, 0]},
                    {"route": "green_bin", "pose": [300, 160, 100, 180, 0, 0]},
                ],
                "use_cycle": True,
            },
            self.robot,
        )
        profile = result.updates["vision_tasks"]["profiles"]["field_slots"]
        self.assertEqual(len(profile["place"]["route_slots"]["red_bin"]), 2)
        self.assertTrue(result.updates["dvs_cycle"]["execute_motion"])
        self.assertEqual(result.updates["dvs_cycle"]["on_error"], "stop")

    def test_motion_wizard_rejects_missing_or_unsafe_route_pose(self):
        base = {
            "task_type": "defect_sort",
            "profile_name": "field_motion",
            "recipe_name": "现场动作",
            "status": {"ok_values": ["OK"], "ng_values": ["NG"]},
            "route_map": {"OK": "ok", "NG": "reject"},
            "execute_motion": True,
            "pick": {
                "fields": ["x", "y", "z"],
                "defaults": [0, 0, 0, 180, 0, 0],
            },
            "placement_mode": "fixed",
            "placements": [{"route": "ok", "pose": [300, 100, 100, 180, 0, 0]}],
        }
        with self.assertRaisesRegex(ValueError, "缺少路线放置位姿"):
            build_task_preset(base, self.robot)
        base["placements"].append(
            {"route": "reject", "pose": [300, 160, 20, 180, 0, 0]}
        )
        with self.assertRaisesRegex(ValueError, "低于安全下限"):
            build_task_preset(base, self.robot)

    def test_recognition_sort_wizard_builds_expected_value_rules(self):
        result = build_task_preset(
            {
                "task_type": "recognition_sort",
                "profile_name": "field_code_check",
                "recipe_name": "现场编码核验",
                "response_fields": ["ocr"],
                "aliases": {"ocr": ["text"]},
                "identifier_fields": ["ocr"],
                "identifier_rules": {
                    "ocr": {"mode": "allowed", "expected": ["A01", "A02"]}
                },
                "route_map": {"OK": "ok", "NG": "reject"},
                "execute_motion": False,
                "use_cycle": True,
            },
            self.robot,
        )
        profile = result.updates["vision_tasks"]["profiles"]["field_code_check"]
        self.assertEqual(profile["identifier_rules"]["ocr"]["allowed"], ["A01", "A02"])
        self.assertTrue(result.updates["dvs_cycle"]["validate_identifiers"])
        self.assertTrue(result.updates["dvs_cycle"]["route"])

    def test_combined_wizard_enables_multiple_processors(self):
        result = build_task_preset(
            {
                "task_type": "combined",
                "profile_name": "field_combined",
                "recipe_name": "现场综合任务",
                "response_fields": ["status", "width", "ocr"],
                "aliases": {},
                "processor": "status_measurement",
                "status": {"field": "status", "ok_values": ["OK"], "ng_values": ["NG"]},
                "measurement_limits": {"width": {"min": 19.8, "max": 20.2}},
                "validate_identifiers": True,
                "identifier_fields": ["ocr"],
                "identifier_rules": {"ocr": {"mode": "regex", "expected": r"A\d{2}"}},
                "route_enabled": True,
                "route_map": {"OK": "ok", "NG": "reject"},
                "execute_motion": False,
                "use_cycle": True,
            },
            self.robot,
        )
        cycle = result.updates["dvs_cycle"]
        self.assertEqual(cycle["processor"], "status_measurement")
        self.assertTrue(cycle["validate_identifiers"])
        self.assertTrue(cycle["route"])


class TransportTests(unittest.TestCase):
    def test_preview_transport_returns_ack_without_network(self):
        config = load_config(ROOT / "configs" / "national.yaml")
        config["mode"] = "hardware"
        config["command_transport"]["mode"] = "preview"
        reply = JsonCommandTransport(config).send({"cmd": "test"})
        self.assertTrue(reply.ok)
        self.assertEqual(reply.mode, "preview")
        self.assertIn('"cmd":"test"', reply.raw)

    def test_vision_studio_disabled_mode_fails_without_guessing_endpoint(self):
        config = load_config(ROOT / "configs" / "national.yaml")
        config["mode"] = "hardware"
        client = DobotVisionStudioClient(config)
        with self.assertRaisesRegex(RuntimeError, "尚未配置"):
            client.connect()

    def test_vision_studio_simulation_parses_json_result(self):
        config = load_config(ROOT / "configs" / "national.yaml")
        config["vision_studio"]["protocol"]["exchange_command"] = "RUN"
        config["vision_studio"]["simulation_result"] = {
            "label": "banana",
            "height_mm": 35.0,
        }
        reply = DobotVisionStudioClient(config, simulation=True).exchange()
        self.assertEqual(reply.data["label"], "banana")
        self.assertEqual(reply.data["height_mm"], 35.0)

    def test_vision_studio_tcp_client_maps_delimited_result(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        received: list[bytes] = []

        def serve() -> None:
            connection, _address = server.accept()
            with connection:
                received.append(connection.recv(1024))
                connection.sendall(b"banana,320,240,35.5\n")
            server.close()

        thread = Thread(target=serve, daemon=True)
        thread.start()
        config = load_config(ROOT / "configs" / "national.yaml")
        config["mode"] = "hardware"
        config["vision_studio"].update(
            {
                "mode": "tcp_client",
                "send_terminator": "\n",
                "receive_terminator": "\n",
            }
        )
        config["vision_studio"]["tcp"] = {"host": "127.0.0.1", "port": port}
        config["vision_studio"]["protocol"] = {
            "exchange_command": "RUN",
            "delimiter": ",",
            "response_fields": ["label", "u", "v", "height_mm"],
        }
        client = DobotVisionStudioClient(config)
        try:
            reply = client.exchange()
        finally:
            client.close()
        thread.join(timeout=2.0)
        self.assertEqual(received, [b"RUN\n"])
        self.assertEqual(
            reply.data,
            {"label": "banana", "u": 320, "v": 240, "height_mm": 35.5},
        )

    def test_vision_studio_whitespace_delimiter_accepts_repeated_spaces(self):
        config = load_config(ROOT / "configs" / "national.yaml")
        config["vision_studio"]["protocol"] = {
            "delimiter": "whitespace",
            "response_fields": ["status", "x", "y"],
        }
        client = DobotVisionStudioClient(config, simulation=True)
        self.assertEqual(
            client._parse_reply("NG   312.5\t-18.2"),
            {"status": "NG", "x": 312.5, "y": -18.2},
        )

    def test_vision_studio_set_global_value_uses_documented_command(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        received: list[bytes] = []

        def serve() -> None:
            connection, _address = server.accept()
            with connection:
                received.append(connection.recv(1024))
            server.close()

        thread = Thread(target=serve, daemon=True)
        thread.start()
        config = load_config(ROOT / "configs" / "national.yaml")
        config["mode"] = "hardware"
        config["vision_studio"]["mode"] = "tcp_client"
        config["vision_studio"]["send_terminator"] = "\n"
        config["vision_studio"]["tcp"] = {"host": "127.0.0.1", "port": port}
        client = DobotVisionStudioClient(config)
        try:
            client.set_global_value("task", 2)
        finally:
            client.close()
        thread.join(timeout=2.0)
        self.assertEqual(received, [b"SetGlobalValue:task=2\n"])

    def test_workflow_sets_all_configured_dvs_globals(self):
        config = load_config(ROOT / "configs" / "national.yaml")
        config["mode"] = "simulation"
        config["vision_studio"]["protocol"]["global_values"] = {"task": 2, "threshold": 0.8}
        workflow = CompetitionWorkflow(config, logger=lambda _message: None)
        values = workflow.set_vision_studio_globals()
        self.assertEqual(values, {"task": 2, "threshold": 0.8})
        self.assertEqual(workflow.shared_data["vision_studio_globals"], values)

    def test_vision_studio_can_receive_unsolicited_result(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        def serve() -> None:
            connection, _address = server.accept()
            with connection:
                connection.sendall(b'{"flow":"done","result":1}\n')
            server.close()

        thread = Thread(target=serve, daemon=True)
        thread.start()
        config = load_config(ROOT / "configs" / "national.yaml")
        config["mode"] = "hardware"
        config["vision_studio"]["mode"] = "tcp_client"
        config["vision_studio"]["receive_terminator"] = "\n"
        config["vision_studio"]["tcp"] = {"host": "127.0.0.1", "port": port}
        config["vision_studio"]["protocol"]["send_before_receive"] = False
        client = DobotVisionStudioClient(config)
        try:
            reply = client.exchange()
        finally:
            client.close()
        thread.join(timeout=2.0)
        self.assertEqual(reply.data, {"flow": "done", "result": 1})


class VisionTaskProcessorTests(unittest.TestCase):
    def test_profile_validation_rejects_invalid_motion_defaults(self):
        with self.assertRaisesRegex(RuntimeError, "pick.defaults"):
            VisionTaskProcessor({"pick": {"fields": ["x", "y"], "defaults": [0, 0]}})

    def test_aliases_measurements_and_route_use_one_contract(self):
        processor = VisionTaskProcessor(
            {
                "aliases": {"width": ["w"], "status": ["result"]},
                "measurement_limits": {"width": {"min": 19.8, "max": 20.2}},
                "route": {"field": "decision", "map": {"OK": "pass", "NG": "reject"}},
            }
        )
        result = processor.normalize({"w": 20.3, "result": "OK"})
        result = processor.evaluate_measurements(result)
        result = processor.resolve_route(result)
        self.assertEqual(result.fields["width"], 20.3)
        self.assertEqual(result.decision, "NG")
        self.assertEqual(result.route, "reject")
        self.assertIn("width=20.3", result.violations[0])

    def test_identifier_validation_rejects_blank_values(self):
        processor = VisionTaskProcessor({"identifier_fields": ["ocr", "barcode", "qrcode"]})
        result = processor.normalize({"ocr": "A12", "barcode": "", "qrcode": "Q01"})
        with self.assertRaisesRegex(RuntimeError, "barcode"):
            processor.validate_identifiers(result)

    def test_identifier_rules_produce_ok_ng_for_exact_list_and_regex(self):
        processor = VisionTaskProcessor(
            {
                "identifier_fields": ["ocr", "barcode", "qrcode"],
                "identifier_rules": {
                    "ocr": {"equals": "A01", "case_sensitive": False},
                    "barcode": {"allowed": ["69001", "69002"]},
                    "qrcode": {"regex": r"Q\d{3}"},
                },
            }
        )
        ok = processor.validate_identifiers(
            processor.normalize({"ocr": "a01", "barcode": "69002", "qrcode": "Q123"})
        )
        self.assertEqual(ok.decision, "OK")
        ng = processor.validate_identifiers(
            processor.normalize({"ocr": "BAD", "barcode": "69002", "qrcode": "Q123"})
        )
        self.assertEqual(ng.decision, "NG")
        self.assertIn("ocr='BAD'", ng.violations[0])

    def test_dynamic_assembly_poses_are_resolved_from_fields(self):
        processor = VisionTaskProcessor(
            {
                "pick": {
                    "fields": ["part_x", "part_y", "part_z"],
                    "defaults": [0, 0, 0, 180, 0, 0],
                    "offsets": [1, -2, 3, 0, 0, 5],
                },
                "place": {
                    "fields": ["hole_x", "hole_y", "hole_z"],
                    "defaults": [0, 0, 0, 180, 0, 0],
                },
            }
        )
        result = processor.normalize(
            {"part_x": 300, "part_y": 20, "part_z": 100, "hole_x": 450, "hole_y": 30, "hole_z": 105}
        )
        self.assertEqual(processor.resolve_pick_pose(result), [301, 18, 103, 180, 0, 5])
        self.assertEqual(processor.resolve_place_pose(result), [450, 30, 105, 180, 0, 0])

    def test_route_slots_are_used_in_order_and_stop_when_full(self):
        processor = VisionTaskProcessor(
            {
                "place": {
                    "route_slots": {
                        "red": [
                            [400, 100, 100, 180, 0, 0],
                            [430, 100, 100, 180, 0, 0],
                        ]
                    }
                }
            }
        )
        result = VisionTaskResult(fields={}, route="red")
        self.assertEqual(processor.resolve_place_pose(result, 0)[0], 400)
        self.assertEqual(processor.resolve_place_pose(result, 1)[0], 430)
        with self.assertRaisesRegex(RuntimeError, "槽位已用完"):
            processor.resolve_place_pose(result, 2)

    def test_route_stack_increases_z_and_stops_at_max_layers(self):
        processor = VisionTaskProcessor(
            {
                "place": {
                    "route_stacks": {
                        "ok": {
                            "base_pose": [450, 100, 100, 180, 0, 0],
                            "layer_height_mm": 18,
                            "max_layers": 3,
                        }
                    }
                }
            }
        )
        result = VisionTaskResult(fields={}, route="ok")
        self.assertEqual(processor.resolve_place_pose(result, 2)[2], 136)
        with self.assertRaisesRegex(RuntimeError, "堆叠层数已满"):
            processor.resolve_place_pose(result, 3)


class WorkflowSafetyTests(unittest.TestCase):
    def test_fixed_action_sequence_is_planned_before_connect_and_executes_in_simulation(self):
        config = load_config(ROOT / "configs" / "national.yaml")
        config["mode"] = "simulation"
        config["action_sequences"] = {
            "active_profile": "demo",
            "pose_bank": {
                "above": [300, 0, 180, 180, 0, 0],
                "pick": [300, 0, 100, 180, 0, 0],
            },
            "profiles": {
                "demo": {
                    "steps": [
                        {"action": "movj", "pose_ref": "above"},
                        {"action": "movl_slow", "pose_ref": "pick"},
                        {"action": "suction", "value": True},
                        {"action": "suction", "value": False},
                        {"action": "home"},
                    ]
                }
            },
        }
        workflow = CompetitionWorkflow(config, logger=lambda _message: None)
        plan = workflow.plan_action_sequence()
        self.assertFalse(workflow.robot.connected)
        self.assertEqual(plan.profile, "demo")
        workflow.execute_action_sequence()
        self.assertTrue(workflow.robot.connected)
        self.assertEqual(workflow.robot.get_tool_do(config["robot"]["suction_tool_do"]), 0)
        for actual, expected in zip(workflow.robot.current_pose, config["robot"]["home_pose"]):
            self.assertAlmostEqual(actual, expected, places=2)

    def test_fixed_action_sequence_rejects_unsafe_z_before_connecting_robot(self):
        config = load_config(ROOT / "configs" / "national.yaml")
        config["mode"] = "simulation"
        config["action_sequences"] = {
            "active_profile": "unsafe",
            "pose_bank": {"unsafe": [300, 0, 10, 180, 0, 0]},
            "profiles": {"unsafe": {"steps": [{"action": "movl", "pose_ref": "unsafe"}]}},
        }
        workflow = CompetitionWorkflow(config, logger=lambda _message: None)
        with self.assertRaisesRegex(RuntimeError, "低于安全下限"):
            workflow.execute_action_sequence()
        self.assertFalse(workflow.robot.connected)

    def test_grasp_z_offset_is_applied_once_without_table_plane(self):
        base = load_config(ROOT / "configs" / "competition.yaml")
        base["mode"] = "simulation"
        base["robot"]["min_grasp_z_mm"] = -10000.0

        without_offset = deepcopy(base)
        without_offset["robot"]["grasp_z_offset_mm"] = 0.0
        workflow_a = CompetitionWorkflow(without_offset, logger=lambda _message: None)
        workflow_a.start_camera()
        workflow_a.detect_once()

        with_offset = deepcopy(base)
        with_offset["robot"]["grasp_z_offset_mm"] = 20.0
        workflow_b = CompetitionWorkflow(with_offset, logger=lambda _message: None)
        workflow_b.start_camera()
        workflow_b.detect_once()

        self.assertIsNotNone(workflow_a.current_target)
        self.assertIsNotNone(workflow_b.current_target)
        delta = workflow_b.current_target.base_pose[2] - workflow_a.current_target.base_pose[2]
        self.assertAlmostEqual(delta, 20.0, places=5)

    def test_simulated_motion_wait_reports_robot_ready(self):
        config = load_config(ROOT / "configs" / "competition.yaml")
        config["mode"] = "simulation"
        workflow = CompetitionWorkflow(config, logger=lambda _message: None)
        workflow.connect_robot()
        workflow.return_home()
        self.assertTrue(workflow.robot.enabled)

    def test_dvs_result_motion_rejects_pose_below_safety_floor(self):
        config = load_config(ROOT / "configs" / "national.yaml")
        config["mode"] = "simulation"
        config["vision_tasks"]["profiles"]["unsafe_test"] = {
            "pick": {"fields": ["x", "y", "z"], "defaults": [0, 0, 0, 180, 0, 0]},
            "place": {"pose": [450, 100, 100, 180, 0, 0]},
        }
        config["vision_tasks"]["active_profile"] = "unsafe_test"
        workflow = CompetitionWorkflow(config, logger=lambda _message: None)
        workflow.vision_studio_data = {"x": 300, "y": 20, "z": 10}
        workflow.normalize_vision_result()
        with self.assertRaisesRegex(RuntimeError, "低于安全下限"):
            workflow.execute_result_pick_place()
        self.assertFalse(workflow.robot.connected)


if __name__ == "__main__":
    unittest.main()
