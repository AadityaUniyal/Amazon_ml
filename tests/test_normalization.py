import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.normalization import normalize_text, normalize_business_name, normalize_business_address

class TestNormalization(unittest.TestCase):
    def test_unicode_and_casing(self):
        text = "Léarning   Center,  LLC"
        norm = normalize_text(text)
        self.assertEqual(norm, "learning center llc")

    def test_suffix_stripping(self):
        res = normalize_business_name("Amazon India Private Limited")
        self.assertEqual(res['name_norm_nosuffix'], "amazon india")
        self.assertEqual(res['name_compact'], "amazonindia")

    def test_prefix_removal(self):
        text = "M/s Reliance Industries Ltd"
        norm = normalize_text(text)
        self.assertNotIn("m/s", norm)

    def test_address_number_preservation(self):
        res = normalize_business_address("Plot No. 160/2, Near SBI ATM, Sector-5, Noida")
        self.assertTrue(any('160' in n for n in res['numbers']))
        self.assertTrue(any('5' in n for n in res['numbers']))

    def test_indic_transliteration(self):
        text_odia = "ଶକ୍ତି ଆଗ୍ରୋ ଲିମିଟେଡ୍"
        norm = normalize_text(text_odia)
        self.assertIn("shkti", norm)
        self.assertIn("limited", norm)

if __name__ == '__main__':
    unittest.main()

