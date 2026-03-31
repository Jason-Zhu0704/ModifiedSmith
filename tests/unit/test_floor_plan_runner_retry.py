import unittest

from scenesmith.floor_plan_agents.stateful_floor_plan_agent import StatefulFloorPlanAgent


class TestFloorPlanRunnerRetry(unittest.TestCase):
    def test_detects_provider_tool_call_name_error(self):
        err = Exception(
            "1 validation error for ResponseFunctionToolCall\n"
            "name\n"
            "  Input should be a valid string [type=string_type, input_value=None]"
        )
        self.assertTrue(StatefulFloorPlanAgent._is_provider_tool_call_name_error(err))

    def test_ignores_unrelated_errors(self):
        err = Exception("Timeout while calling model endpoint")
        self.assertFalse(StatefulFloorPlanAgent._is_provider_tool_call_name_error(err))
