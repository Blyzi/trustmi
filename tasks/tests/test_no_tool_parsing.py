from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from inspect_ai.model import ChatMessageUser, GenerateConfig
from inspect_ai.tool import ToolInfo

from benchmarks.model import TrustMIModelAPI
from utils.vllm_lens_wrapper import VLLMLensEngine


def _backend_response(
    content: str = "ordinary prose", *, hidden_params: dict | None = None
) -> SimpleNamespace:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=3,
            completion_tokens=2,
            total_tokens=5,
        ),
    )
    if hidden_params is not None:
        response._hidden_params = hidden_params
    return response


class InspectToolRequestTests(unittest.TestCase):
    def _api(self) -> TrustMIModelAPI:
        # ``@modelapi`` exports a registry factory; the provider class remains
        # in its closure. Bypass its heavyweight model-loading constructor for
        # this adapter-boundary unit test.
        api_type = next(
            cell.cell_contents
            for cell in TrustMIModelAPI.__closure__ or ()
            if isinstance(cell.cell_contents, type)
        )
        api = object.__new__(api_type)
        api.model_name = "allenai/Olmo-3-7B-Instruct"
        api.backend = SimpleNamespace(
            steered_completion=Mock(
                return_value=_backend_response(
                    hidden_params={
                        "steering_backend": "vllm_lens",
                        "steering_target": "latest_user",
                        "steering_range_count": 1,
                        "steering_ranges_verified": True,
                    }
                )
            )
        )
        api.vector = object()
        api.layer_rows = [0]
        api.strength = 1.0
        api.steering_target = "latest_user"
        api.think = False
        api.default_max_tokens = 64
        return api

    def test_empty_tools_select_olmo_no_tool_template_branch(self) -> None:
        """None, not [], selects OLMo's official no-functions wording."""
        api = self._api()

        asyncio.run(
            api.generate(
                [ChatMessageUser(content="Answer this question.")],
                [],
                "none",
                GenerateConfig(),
            )
        )

        call = api.backend.steered_completion.call_args
        self.assertIsNone(call.args[2])
        self.assertEqual(call.kwargs["steering_target"], "latest_user")

    def test_declared_tools_are_forwarded_unchanged(self) -> None:
        api = self._api()
        tool = ToolInfo(
            name="lookup",
            description="Look up a value.",
            parameters={
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        )

        asyncio.run(
            api.generate(
                [ChatMessageUser(content="Look this up.")],
                [tool],
                "auto",
                GenerateConfig(),
            )
        )

        call = api.backend.steered_completion.call_args
        self.assertEqual(
            call.args[2],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Look up a value.",
                        "parameters": {
                            "type": "object",
                            "properties": {"key": {"type": "string"}},
                            "required": ["key"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        )

    def test_lens_range_audit_is_serialized_in_output_metadata(self) -> None:
        api = self._api()
        api.backend.steered_completion.return_value = _backend_response(
            hidden_params={
                "steering_backend": "vllm_lens",
                "steering_target": "latest_user",
                "steering_range_count": 1,
                "steering_ranges_verified": True,
            }
        )

        output = asyncio.run(
            api.generate(
                [ChatMessageUser(content="Answer this question.")],
                [],
                "none",
                GenerateConfig(),
            )
        )

        self.assertEqual(
            output.metadata,
            {
                "steering_backend": "vllm_lens",
                "steering_target": "latest_user",
                "steering_range_count": 1,
                "steering_ranges_verified": True,
            },
        )
        self.assertEqual(json.loads(json.dumps(output.metadata)), output.metadata)

    def test_strength_zero_records_unsteered_lens_baseline(self) -> None:
        api = self._api()
        api.strength = 0.0
        api.backend.steered_completion.return_value = _backend_response(
            hidden_params={
                "steering_backend": "vllm_lens",
                "steering_target": "latest_user",
                "steering_range_count": 1,
                "steering_ranges_verified": False,
            }
        )

        output = asyncio.run(
            api.generate(
                [ChatMessageUser(content="Answer this question.")],
                [],
                "none",
                GenerateConfig(),
            )
        )

        self.assertEqual(
            output.metadata,
            {
                "steering_backend": "vllm_lens",
                "steering_target": "latest_user",
                "steering_range_count": 1,
                "steering_ranges_verified": False,
            },
        )

    def test_non_lens_backend_metadata_is_not_exposed(self) -> None:
        api = self._api()
        api.backend.steered_completion.return_value = _backend_response(
            hidden_params={"steering_backend": "unexpected"}
        )

        output = asyncio.run(
            api.generate(
                [ChatMessageUser(content="Answer this question.")],
                [],
                "none",
                GenerateConfig(),
            )
        )

        self.assertIsNone(output.metadata)


class ToolParserRequestTests(unittest.TestCase):
    @staticmethod
    def _engine(parser: Mock):
        engine = object.__new__(VLLMLensEngine)
        engine._reasoning_parser = None
        engine._tool_parser = parser
        return engine

    def test_no_tools_skips_model_tool_parser(self) -> None:
        parser = Mock()
        engine = self._engine(parser)
        parsed = engine.parse_output(
            "ordinary prose",
            SimpleNamespace(tools=None),
            thinking_disabled=True,
        )

        self.assertEqual(parsed, ("ordinary prose", None, [], False))
        parser.extract_tool_calls.assert_not_called()

    def test_declared_tools_still_use_model_tool_parser(self) -> None:
        tool_call = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="lookup", arguments='{"key":"x"}'),
        )
        parser = Mock()
        parser.extract_tool_calls.return_value = SimpleNamespace(
            tools_called=True,
            content="",
            tool_calls=[tool_call],
        )
        engine = self._engine(parser)
        request = SimpleNamespace(tools=[{"type": "function"}])

        content, reasoning, tool_calls, tools_called = engine.parse_output(
            '<tool_call>{"name":"lookup"}</tool_call>',
            request,
            thinking_disabled=True,
        )

        parser.extract_tool_calls.assert_called_once_with(
            '<tool_call>{"name":"lookup"}</tool_call>', request
        )
        self.assertEqual(content, "")
        self.assertIsNone(reasoning)
        self.assertTrue(tools_called)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].function.name, "lookup")


if __name__ == "__main__":
    unittest.main()
