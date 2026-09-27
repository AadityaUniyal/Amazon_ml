import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.normalization import normalize_record
from src.blocking import MultiChannelBlocker

class TestBlocking(unittest.TestCase):
    def setUp(self):
        self.blocker = MultiChannelBlocker(max_candidates=10)
        # Create small target pool
        self.targets = {
            "S2-101": normalize_record("S2-101", "Acme Logistics Inc", "123 Main St, New York", "US"),
            "S3-202": normalize_record("S3-202", "Acme Logistics", "123 Main Street, NY", "US"),
            "S2-303": normalize_record("S2-303", "Global Health Corp", "456 Oak Rd, Chicago", "US")
        }
        self.blocker.fit_targets(self.targets)

    def test_candidate_retrieval(self):
        s1 = normalize_record("S1-001", "Acme Logistics LLC", "123 Main St, New York", "US")
        cands = self.blocker.retrieve_candidates(s1)
        self.assertIn("S2-101", cands)
        self.assertIn("S3-202", cands)
        self.assertNotIn("S1-001", cands)  # No self matches

    def test_singleton_dynamic_pruning(self):
        # Entity with completely disjoint tokens should produce zero candidates (singleton)
        s1_singleton = normalize_record("S1-999", "Zilch Nonexistent Enterprises", "9999 Faraway Blvd, Nowhere", "US")
        cands = self.blocker.retrieve_candidates(s1_singleton)
        self.assertEqual(len(cands), 0)

if __name__ == '__main__':
    unittest.main()
