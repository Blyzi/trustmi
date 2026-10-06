from __future__ import annotations

import unittest

from jinja2.exceptions import TemplateError

from utils.steering import build_steered_prompt
from utils.steering_policy import (
    resolve_steering_ranges,
    training_span_for_target,
    validate_steering_target,
)


class CharacterTokenizer:
    """Tiny concatenative chat template whose token ids are character ids."""

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking=False,
        tools=None,
    ):
        assert tokenize is False
        rendered = ""
        if tools:
            rendered += "<tools>lookup</tools>"
        for message in messages:
            role = message["role"]
            rendered += f"<{role}>{message.get('content', '')}</{role}>"
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def __call__(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}

    @staticmethod
    def decode(token_ids) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class GroupedToolTokenizer(CharacterTokenizer):
    """Template whose earlier tool wrapper changes when another tool follows."""

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking=False,
        tools=None,
    ):
        assert tokenize is False
        rendered = ""
        for index, message in enumerate(messages):
            role = message["role"]
            if role != "tool":
                rendered += f"<{role}>{message.get('content', '')}</{role}>"
                continue
            previous_is_tool = index > 0 and messages[index - 1]["role"] == "tool"
            next_is_tool = (
                index + 1 < len(messages)
                and messages[index + 1]["role"] == "tool"
            )
            if not previous_is_tool:
                rendered += "<user>"
            rendered += f"<tool>{message.get('content', '')}</tool>"
            if not next_is_tool:
                rendered += "</user>"
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


class ToolCallTokenizer(CharacterTokenizer):
    """Template that renders every assistant tool call, several per turn."""

    max_calls_per_turn: int | None = None

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking=False,
        tools=None,
    ):
        assert tokenize is False
        rendered = ""
        for message in messages:
            calls = message.get("tool_calls") or []
            if (
                self.max_calls_per_turn is not None
                and len(calls) > self.max_calls_per_turn
            ):
                raise TemplateError("This model only supports single tool-calls at once!")
            role = message["role"]
            names = "".join(f"<call>{call['function']['name']}</call>" for call in calls)
            rendered += f"<{role}>{message.get('content', '')}{names}</{role}>"
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


class SingleToolCallTokenizer(ToolCallTokenizer):
    """Refuses parallel tool calls, as Llama 3.1's template does."""

    max_calls_per_turn = 1


def _call(name: str, call_id: str) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": {}}}


class SteeringTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharacterTokenizer()
        self.messages = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "first request"},
            {"role": "assistant", "content": "calling a tool"},
            {"role": "tool", "content": "old tool result"},
            {"role": "user", "content": "second request"},
            {"role": "assistant", "content": "calling another tool"},
            {"role": "tool", "content": "latest tool result"},
        ]
        self.layout = build_steered_prompt(
            self.tokenizer,
            self.messages,
            system_prompt=None,
            tools=[{"type": "function"}],
        )

    def user_contents(self) -> list[str]:
        return [
            self.tokenizer.decode(self.layout.token_ids[start:end])
            for start, end in self.layout.user_spans
        ]

    def tool_contents(self) -> list[str]:
        return [
            self.tokenizer.decode(self.layout.token_ids[start:end])
            for start, end in self.layout.tool_spans
        ]

    def test_prompt_layout_finds_every_user_content_span(self) -> None:
        self.assertEqual(self.user_contents(), ["first request", "second request"])
        self.assertEqual(
            self.tool_contents(), ["old tool result", "latest tool result"]
        )
        self.assertTrue(
            self.tokenizer.decode(self.layout.token_ids).endswith("<assistant>")
        )

    def test_prompt_shows_the_model_its_tools(self) -> None:
        # The steered engine generates from exactly these ids, so a tool
        # definition missing here is one the model never sees.
        self.assertTrue(
            self.tokenizer.decode(self.layout.token_ids).startswith(
                "<tools>lookup</tools>"
            )
        )

    def test_latest_user_selects_only_the_final_user_span(self) -> None:
        self.assertEqual(
            resolve_steering_ranges(
                "latest_user",
                len(self.layout.token_ids),
                self.layout.user_spans,
                self.layout.tool_spans,
            ),
            [self.layout.user_spans[-1]],
        )

    def test_latest_user_and_tools_selects_following_tool_results(self) -> None:
        self.assertEqual(
            resolve_steering_ranges(
                "latest_user_and_tools",
                len(self.layout.token_ids),
                self.layout.user_spans,
                self.layout.tool_spans,
            ),
            [self.layout.user_spans[-1], self.layout.tool_spans[-1]],
        )

    def test_latest_user_and_tools_without_results_selects_latest_user(self) -> None:
        self.assertEqual(
            resolve_steering_ranges(
                "latest_user_and_tools",
                len(self.layout.token_ids),
                self.layout.user_spans,
                (),
            ),
            [self.layout.user_spans[-1]],
        )

    def test_all_users_selects_every_user_span(self) -> None:
        self.assertEqual(
            resolve_steering_ranges(
                "all_users",
                len(self.layout.token_ids),
                self.layout.user_spans,
                self.layout.tool_spans,
            ),
            list(self.layout.user_spans),
        )

    def test_all_users_and_tools_selects_all_input_content_spans(self) -> None:
        self.assertEqual(
            resolve_steering_ranges(
                "all_users_and_tools",
                len(self.layout.token_ids),
                self.layout.user_spans,
                self.layout.tool_spans,
            ),
            sorted((*self.layout.user_spans, *self.layout.tool_spans)),
        )

    def test_spans_index_final_prompt_when_tool_turns_are_grouped(self) -> None:
        tokenizer = GroupedToolTokenizer()
        messages = [
            {"role": "user", "content": "request"},
            {"role": "assistant", "content": "calling tools"},
            {"role": "tool", "content": "first result"},
            {"role": "tool", "content": "second result"},
        ]
        layout = build_steered_prompt(
            tokenizer,
            messages,
            system_prompt=None,
        )

        self.assertEqual(
            [
                tokenizer.decode(layout.token_ids[start:end])
                for start, end in layout.tool_spans
            ],
            ["first result", "second result"],
        )
        self.assertEqual(
            resolve_steering_ranges(
                "all_users_and_tools",
                len(layout.token_ids),
                layout.user_spans,
                layout.tool_spans,
            ),
            sorted((*layout.user_spans, *layout.tool_spans)),
        )

    def test_generated_assistant_starts_after_the_prompt(self) -> None:
        self.assertEqual(
            resolve_steering_ranges(
                "generated_assistant",
                len(self.layout.token_ids),
                self.layout.user_spans,
                self.layout.tool_spans,
            ),
            [(len(self.layout.token_ids), None)],
        )

    def test_runtime_targets_map_to_their_training_span(self) -> None:
        self.assertEqual(training_span_for_target("latest_user"), "user")
        self.assertEqual(training_span_for_target("latest_user_and_tools"), "user")
        self.assertEqual(training_span_for_target("all_users"), "user")
        self.assertEqual(training_span_for_target("all_users_and_tools"), "user")
        self.assertEqual(training_span_for_target("generated_assistant"), "assistant")

    def test_unknown_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown steering target"):
            validate_steering_target("user")

    def test_invalid_or_overlapping_spans_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid or overlapping"):
            resolve_steering_ranges("all_users", 8, [(1, 4)], [(3, 6)])

    def test_prompt_without_user_message_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no user turn"):
            build_steered_prompt(
                self.tokenizer,
                [{"role": "system", "content": "policy"}],
                system_prompt=None,
            )


class ParallelToolCallTests(unittest.TestCase):
    messages = [
        {"role": "user", "content": "check both"},
        {
            "role": "assistant",
            "content": "on it",
            "tool_calls": [_call("a", "1"), _call("b", "2")],
        },
        {"role": "tool", "content": "result a", "tool_call_id": "1"},
        {"role": "tool", "content": "result b", "tool_call_id": "2"},
    ]

    def render(self, tokenizer):
        layout = build_steered_prompt(tokenizer, self.messages, system_prompt=None)
        return layout, tokenizer.decode(layout.token_ids)

    def test_templates_that_accept_parallel_calls_render_them_unchanged(self) -> None:
        _, text = self.render(ToolCallTokenizer())
        self.assertIn(
            "<assistant>on it<call>a</call><call>b</call></assistant>"
            "<tool>result a</tool><tool>result b</tool>",
            text,
        )

    def test_single_call_templates_see_each_call_in_its_own_turn(self) -> None:
        tokenizer = SingleToolCallTokenizer()
        layout, text = self.render(tokenizer)
        self.assertIn(
            "<assistant>on it<call>a</call></assistant><tool>result a</tool>"
            "<assistant><call>b</call></assistant><tool>result b</tool>",
            text,
        )
        self.assertEqual(
            [tokenizer.decode(layout.token_ids[s:e]) for s, e in layout.tool_spans],
            ["result a", "result b"],
        )


if __name__ == "__main__":
    unittest.main()
