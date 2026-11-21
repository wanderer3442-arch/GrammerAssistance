import unittest

from app.analysis import spell_check, style_report


class AnalysisTests(unittest.TestCase):
    def test_spell_suggestions_are_common_and_deterministic(self):
        hits = spell_check("I recieved it tommorow.")

        self.assertEqual(
            [(hit.word, hit.suggestions[0]) for hit in hits],
            [("recieved", "received"), ("tommorow", "tomorrow")],
        )

    def test_capitalized_word_after_newline_is_checked_at_sentence_boundary(self):
        hits = spell_check("First line.\nTommorow is soon.")

        self.assertEqual(
            [(hit.word, hit.suggestions[0]) for hit in hits],
            [("Tommorow", "Tomorrow")],
        )

    def test_style_report_handles_punctuation_without_fabricating_a_word(self):
        report = style_report("...")

        self.assertEqual(report.words, 0)
        self.assertEqual(report.sentences, 0)
        self.assertEqual(report.avg_sentence_len, 0.0)
        self.assertEqual(report.flesch_label, "—")
