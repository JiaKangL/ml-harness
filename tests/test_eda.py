"""Acceptance tests for the stored EDA (P1b).

The report is a deliverable, so the tests assert the two properties that make it
trustworthy: it agrees exactly with the prompt-sized profile the agent actually
reads, and it contains no label-derived statistic for the test split.

    ./.venv/bin/python -m unittest tests.test_eda -v
"""
from __future__ import annotations

import json
import unittest

from harness import config as C
from harness import eda
from harness.data_guard import DataAPI


class TestEDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = DataAPI()
        cls.eda = eda.build_eda(cls.api)
        cls.md = eda.render_markdown(cls.eda)

    def test_report_is_substantial(self):
        self.assertGreater(len(self.md), 2048)
        for heading in ("within-user variance", "play_time_ms", "Item quality"):
            self.assertIn(heading, self.md)

    def test_no_label_derived_statistic_for_test(self):
        """Test may contribute feature-side statistics (group sizes, churn) and
        nothing else. Checked structurally rather than by reading the prose."""
        test_split = self.eda["profile"]["splits"]["test"]
        for label_derived in ("positive_rate", "user_composition"):
            self.assertNotIn(label_derived, test_split)
        for section in ("long_view_by_tab", "long_view_by_duration"):
            self.assertNotIn("test", self.eda[section])
        # The label-mechanics and popularity sections are train/valid by construction.
        self.assertEqual(self.eda["label_mechanics"]["_what"].split(",")[-1].strip(),
                         "train only")

    def test_shared_figures_match_the_profile_exactly(self):
        """`data_profile.json` is the compression of this report. If the two ever
        disagree, the agent is reasoning from numbers the write-up does not defend."""
        on_disk = json.loads(C.DATA_PROFILE_JSON.read_text())
        self.assertEqual(self.eda["profile"], on_disk)

    def test_known_value_duplicate_pairs_on_test(self):
        """Reproduces the published 3.06% duplicate (user, video) rate on test."""
        self.assertAlmostEqual(
            self.eda["profile"]["splits"]["test"]["duplicate_user_video_pct"],
            3.06,
            places=2,
        )

    def test_popularity_only_is_below_fm_but_far_above_random(self):
        """The headroom claim in the write-up. If item popularity alone matched FM,
        the whole personalisation premise of the project would be wrong."""
        pop = self.eda["popularity_only_ceiling"]
        self.assertLess(pop["primary"], C.EXPECTED["fm_valid_primary"])
        self.assertGreater(pop["primary"], C.EXPECTED["random_valid_primary"])
        self.assertGreater(pop["personalisation_worth_gauc"], 0.0)

    def test_label_mechanics_separate_positives_from_negatives(self):
        """The visual proof behind the leakage firewall: if the play/duration ratio
        did NOT separate the classes, firewalling play_time_ms would be superstition."""
        lab = self.eda["label_mechanics"]
        self.assertGreater(lab["positives"]["p50"], 0.5)
        self.assertLess(lab["negatives"]["p50"], 0.2)
        self.assertGreater(lab["threshold_rule_agreement"], 0.8)


if __name__ == "__main__":
    unittest.main()
