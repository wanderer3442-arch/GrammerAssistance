import unittest
from unittest.mock import patch

from app import engine


class EngineTests(unittest.TestCase):
    def test_sentence_splitter_keeps_terminal_punctuation(self):
        self.assertEqual(
            engine._split_sentences("Why he work here? He bought a apple."),
            ["Why he work here?", "He bought a apple."],
        )

    def test_polish_preserves_blank_line_paragraph_boundaries(self):
        class FakeTokenizer:
            def __call__(self, *_args, **_kwargs):
                return {}

            def decode(self, *_args, **_kwargs):
                return "fixed."

        class FakeModel:
            def generate(self, **_kwargs):
                return [[0]]

        source = "First paragraph.\n\nSecond paragraph."
        with patch.object(engine, "_load", return_value=(FakeTokenizer(), FakeModel())):
            result = engine.polish(source)

        self.assertEqual(result, "fixed.\n\nfixed.")
