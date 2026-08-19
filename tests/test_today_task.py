from __future__ import annotations

import socket
import threading
import time
import unittest
from pathlib import Path

from app.config import load_config
from app.core.task_output_receiver import TcpTaskOutputReceiver, parse_task_output_payload
from app.core.workflow import CompetitionWorkflow
from app.today_main import TodayMainWindow


ROOT = Path(__file__).resolve().parents[1]


class TodayTaskTests(unittest.TestCase):
    def make_workflow(self):
        config = load_config(ROOT / "configs" / "today.yaml")
        config["robot"]["suction_settle_s"] = 0
        config["robot"]["suction_release_settle_s"] = 0
        messages: list[str] = []
        return config, messages, CompetitionWorkflow(config, logger=messages.append)

    def test_today_config_matches_seven_class_five_object_task(self):
        config, _messages, _workflow = self.make_workflow()
        self.assertEqual(len(config["model"]["class_names"]), 7)
        self.assertEqual(config["workflow"]["auto_max_objects"], 5)
        self.assertGreaterEqual(config["model"]["conf_threshold"], 0.8)
        self.assertEqual(set(config["model"]["class_names"]), set(config["robot"]["bins"]))
        self.assertEqual(config["task_outputs"]["tcp"]["host"], "192.168.5.4")
        self.assertIsNone(config["task_outputs"]["tcp"]["port"])
        self.assertFalse(config["task_outputs"]["file_fallback_enabled"])

    def test_replace_marker_does_not_leak_into_loaded_config(self):
        config, _messages, _workflow = self.make_workflow()
        self.assertNotIn("__replace__", config["robot"]["bins"])

    def test_today_simulation_runs_five_objects_highest_first(self):
        _config, messages, workflow = self.make_workflow()
        workflow.start_camera()
        self.assertEqual(workflow.auto_run(), 5)
        recognition = [message for message in messages if message.startswith("识别成功")]
        self.assertEqual(len(recognition), 5)
        self.assertIn("六棱柱", recognition[0])
        self.assertIn("平行四边形", recognition[-1])

    def test_unsafe_photo_pose_is_rejected_before_detection(self):
        config, _messages, workflow = self.make_workflow()
        config["robot"]["photo_pose"][2] = config["robot"]["min_grasp_z_mm"] - 1
        workflow.start_camera()
        with self.assertRaisesRegex(RuntimeError, "拍照点"):
            workflow.auto_run()

    def test_single_run_lifts_unique_objects_and_applies_angle_to_yaw(self):
        _config, _messages, workflow = self.make_workflow()
        workflow.start_camera()
        first = workflow.execute_grasp_lift()
        second = workflow.execute_grasp_lift()
        self.assertNotEqual(first.detection.label, second.detection.label)
        self.assertAlmostEqual(first.base_pose[5], first.angle_deg, places=3)
        tool_index = int(_config["robot"].get("suction_tool_do", 1))
        self.assertEqual(workflow.robot.get_tool_do(tool_index), int(_config["robot"].get("suction_on_level", 1)))

    def test_recognition_output_uses_task_book_field_order(self):
        self.assertEqual(
            TodayMainWindow.parse_recognition_parts(["OK", "DOBOT123", "QR001"]),
            ("OK", "DOBOT123", "QR001"),
        )

    def test_tcp_task_output_parser_supports_prefixed_and_json_messages(self):
        settings = {
            "field_aliases": {
                "gear": ["齿轮识别"],
                "recognition": ["字符识别与二维码识别"],
            }
        }
        prefixed = parse_task_output_payload("齿轮识别=36,1.5,57.0,51.2", settings)
        self.assertEqual([(item.key, item.text) for item in prefixed], [("gear", "36,1.5,57.0,51.2")])
        file_alias = parse_task_output_payload("齿轮识别.txt=36,1.5,57.0,51.2", settings)
        self.assertEqual([(item.key, item.text) for item in file_alias], [("gear", "36,1.5,57.0,51.2")])
        values = parse_task_output_payload(
            '{"recognition":"OK,DOBOT123,QR001","defect":"NG,2,18.6,3.1"}',
            settings,
        )
        self.assertEqual(
            [(item.key, item.text) for item in values],
            [("recognition", "OK,DOBOT123,QR001"), ("defect", "NG,2,18.6,3.1")],
        )
        self.assertEqual(parse_task_output_payload("OK,DOBOT123,QR001", settings), [])
        raw = parse_task_output_payload("OK,DOBOT123,QR001", {"default_key": "recognition"})
        self.assertEqual([(item.key, item.text) for item in raw], [("recognition", "OK,DOBOT123,QR001")])

    def test_tcp_task_output_receiver_updates_from_live_socket(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        def serve_once():
            connection, _address = server.accept()
            with connection:
                connection.sendall(
                    b'gear=36,1.5,57.0,51.2\n'
                    b'{"recognition":"OK,DOBOT123,QR001"}\n'
                )

        thread = threading.Thread(target=serve_once, daemon=True)
        thread.start()
        receiver = TcpTaskOutputReceiver(
            {
                "task_outputs": {
                    "tcp": {
                        "enabled": True,
                        "host": "127.0.0.1",
                        "port": port,
                        "delimiter": "\\n",
                        "reconnect_interval_s": 0.05,
                    }
                }
            }
        )
        receiver.start()
        messages = []
        deadline = time.monotonic() + 2.0
        try:
            while time.monotonic() < deadline and len(messages) < 2:
                messages.extend(event[1] for event in receiver.drain() if event[0] == "data")
                time.sleep(0.01)
        finally:
            receiver.stop()
            server.close()
        self.assertEqual(
            [(item.key, item.text) for item in messages],
            [("gear", "36,1.5,57.0,51.2"), ("recognition", "OK,DOBOT123,QR001")],
        )


if __name__ == "__main__":
    unittest.main()
