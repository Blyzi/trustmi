from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, Mock, patch

import torch

from utils.steering import PromptLayout, load_steering_vector
from utils.vllm_common import (
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_REQUEST_TIMEOUT,
    infer_parsers,
)
from utils.vllm_lens_wrapper import VLLMLens


class OlmoIntegrationTests(unittest.TestCase):
    def test_olmo_model_names_select_native_tool_parser(self) -> None:
        for model_id in (
            "allenai/Olmo-3-7B-Instruct",
            "allenai/Olmo-3.1-32B-Instruct",
            "/models/olmo3_7b",
        ):
            with self.subTest(model_id=model_id):
                self.assertEqual(infer_parsers(model_id), (None, "olmo3"))

    def test_llama_model_names_select_native_tool_parser(self) -> None:
        for model_id in (
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "/models/llama31_8b",
        ):
            with self.subTest(model_id=model_id):
                self.assertEqual(infer_parsers(model_id), (None, "llama3_json"))
        backend = object.__new__(VLLMLens)
        backend.reasoning_parser_name = "qwen3"
        backend.tool_parser_name = "hermes"
        self.assertEqual(
            backend._resolve_parsers("/models/llama31_8b", {}),
            (None, "llama3_json"),
        )

    def test_lens_vector_loader_validates_configured_layer_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            vector_path = Path(directory) / "vector.pt"
            torch.save(torch.tensor([[1.0, 0.0], [0.0, 0.0]]), vector_path)

            vector, rows = load_steering_vector(vector_path, n_layers=2)

        self.assertEqual(tuple(vector.shape), (2, 2))
        self.assertEqual(rows, [0])

    def test_olmo_rule_suppresses_unrelated_global_reasoning_parser(self) -> None:
        backend = object.__new__(VLLMLens)
        backend.reasoning_parser_name = "qwen3"
        backend.tool_parser_name = "hermes"

        self.assertEqual(
            backend._resolve_parsers("allenai/Olmo-3-7B-Instruct", {}),
            (None, "olmo3"),
        )
        self.assertEqual(
            backend._resolve_parsers("unknown/model", {}),
            ("qwen3", "hermes"),
        )

    def test_vllm_lens_structured_completion_audits_the_span(self) -> None:
        backend = object.__new__(VLLMLens)
        backend.default_max_tokens = 64
        backend.max_model_len = DEFAULT_MAX_MODEL_LEN
        backend.request_timeout = DEFAULT_REQUEST_TIMEOUT
        final = SimpleNamespace(
            prompt_token_ids=[10, 11, 12, 13, 14, 15, 16],
            outputs=[
                SimpleNamespace(text="done", token_ids=[20, 21], finish_reason="stop")
            ],
            hook_results={
                "0": {
                    "seen": {0: 8},
                    "hits": [(0, 7, [(3, 4), (5, 6)])],
                }
            },
        )
        engine = SimpleNamespace(
            tokenizer=object(),
            reasoning_parser_name=None,
            tool_parser_name="olmo3",
            run_one=AsyncMock(return_value=final),
        )

        # Mirrors `Future.result`, which the caller now bounds with a timeout.
        def submit(coro):
            return SimpleNamespace(result=lambda timeout=None: asyncio.run(coro))

        engine.submit = submit
        backend.get_engine = Mock(return_value=engine)
        backend.prepare_hooks = Mock()
        response = SimpleNamespace(_hidden_params={})
        backend._build_response = Mock(return_value=response)

        with (
            patch(
                "utils.vllm_lens_wrapper.build_steered_prompt",
                return_value=PromptLayout(
                    (10, 11, 12, 13, 14, 15, 16),
                    ((1, 2), (3, 4)),
                    ((5, 6),),
                ),
            ),
            patch(
                "utils.vllm_lens_wrapper.make_span_hook", return_value="hook"
            ) as make_hook,
        ):
            got = backend.steered_completion(
                "allenai/Olmo-3-7B-Instruct",
                [{"role": "user", "content": "hello"}],
                None,
                torch.ones(1, 3),
                [0],
                2.0,
                steering_target="latest_user_and_tools",
            )

        self.assertIs(got, response)
        make_hook.assert_called_once_with(
            [0],
            {0: ANY},
            2.0,
            [(3, 4), (5, 6)],
            record=True,
        )
        self.assertEqual(
            response._hidden_params,
            {
                "steering_backend": "vllm_lens",
                "steering_target": "latest_user_and_tools",
                "steering_range_count": 2,
                "steering_ranges_verified": True,
            },
        )

    def test_vllm_lens_elo_generation_uses_the_shared_target_ranges(self) -> None:
        backend = object.__new__(VLLMLens)
        final = SimpleNamespace(
            outputs=[SimpleNamespace(text="done", token_ids=[20], finish_reason="stop")],
            hook_results={"0": {"seen": {0: 7}}},
        )
        engine = SimpleNamespace(run_one=AsyncMock(return_value=final))

        def submit(coro):
            return SimpleNamespace(result=lambda: asyncio.run(coro))

        engine.submit = submit
        backend.get_engine = Mock(return_value=engine)
        backend.prepare_hooks = Mock()
        layout = PromptLayout(
            (10, 11, 12, 13, 14, 15, 16),
            ((1, 2), (3, 4)),
            ((5, 6),),
        )

        with patch(
            "utils.vllm_lens_wrapper.make_span_hook", return_value="hook"
        ) as make_hook:
            texts = backend.generate_steered(
                "allenai/Olmo-3-7B-Instruct",
                [(layout, 2.0)],
                torch.ones(1, 3),
                [0],
                steering_target="latest_user_and_tools",
            )

        self.assertEqual(texts, ["done"])
        make_hook.assert_called_once_with(
            [0],
            {0: ANY},
            2.0,
            [(3, 4), (5, 6)],
        )

    def test_vllm_lens_zero_strength_attaches_no_hook(self) -> None:
        backend = object.__new__(VLLMLens)
        final = SimpleNamespace(
            outputs=[SimpleNamespace(text="baseline", token_ids=[20])],
            hook_results={},
        )
        engine = SimpleNamespace(run_one=AsyncMock(return_value=final))

        def submit(coro):
            return SimpleNamespace(result=lambda: asyncio.run(coro))

        engine.submit = submit
        backend.get_engine = Mock(return_value=engine)
        backend.prepare_hooks = Mock()
        layout = PromptLayout((10, 11, 12), ((1, 2),), ())

        with patch("utils.vllm_lens_wrapper.make_span_hook") as make_hook:
            texts = backend.generate_steered(
                "allenai/Olmo-3-7B-Instruct",
                [(layout, 0.0)],
                torch.ones(1, 3),
                [0],
                steering_target="latest_user",
            )

        self.assertEqual(texts, ["baseline"])
        make_hook.assert_not_called()


if __name__ == "__main__":
    unittest.main()
