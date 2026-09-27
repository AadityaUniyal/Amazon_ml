"""
Pairwise Feature Engineering Module (Role B).
Computes 20 discriminative signals across:
- Name exact, edit distance, token-level Jaccard, token sort/set, Jaro-Winkler
- Address exact, token-level Jaccard, token sort/set, length differences
- Alphanumeric building & plot number overlap, symmetric difference, conflict flags
- Cross-feature interactions (strong name + strong address, exact name + conflict address)
- Open-set country agreement flag
"""
from rapidfuzz import fuzz, distance

def compute_jaccard(tokens1, tokens2):
    s1, s2 = set(tokens1), set(tokens2)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / float(len(s1 | s2))

def compute_overlap_ratio(tokens1, tokens2):
    s1, s2 = set(tokens1), set(tokens2)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / float(min(len(s1), len(s2)))

def extract_pair_features(s1_rec, cand_rec, cand_score=0.0):
    """
    Extracts 27 pairwise numerical features between S1 reference record and target candidate.
    """
    n1 = s1_rec['name_norm']
    n2 = cand_rec['name_norm']
    a1 = s1_rec['address_norm']
    a2 = cand_rec['address_norm']
    
    # 1-5. Name similarity signals
    name_ratio = fuzz.ratio(n1, n2) / 100.0
    name_partial = fuzz.partial_ratio(n1, n2) / 100.0
    name_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
    name_set = fuzz.token_set_ratio(n1, n2) / 100.0
    name_jw = distance.JaroWinkler.similarity(n1, n2)
    
    # 6-8. Name token overlap, Jaccard, and No-suffix similarity
    name_jaccard = compute_jaccard(s1_rec['name_tokens'], cand_rec['name_tokens'])
    name_overlap_ratio = compute_overlap_ratio(s1_rec['name_tokens'], cand_rec['name_tokens'])
    name_nosuf_ratio = fuzz.ratio(s1_rec.get('name_norm_nosuffix', ''), cand_rec.get('name_norm_nosuffix', '')) / 100.0
    
    # 9. First token brand anchor ratio
    toks1 = s1_rec.get('name_tokens', [])
    toks2 = cand_rec.get('name_tokens', [])
    first_tok1 = toks1[0] if toks1 else ""
    first_tok2 = toks2[0] if toks2 else ""
    name_first_tok_ratio = fuzz.ratio(first_tok1, first_tok2) / 100.0 if (first_tok1 and first_tok2) else 0.0
    
    # 10-13. Address similarity signals
    addr_ratio = fuzz.ratio(a1, a2) / 100.0
    addr_sort = fuzz.token_sort_ratio(a1, a2) / 100.0
    addr_set = fuzz.token_set_ratio(a1, a2) / 100.0
    addr_jaccard = compute_jaccard(s1_rec['address_tokens'], cand_rec['address_tokens'])
    
    # 14-16. Number overlap and conflict signals
    nums1 = s1_rec['numbers']
    nums2 = cand_rec['numbers']
    common_nums = float(len(nums1 & nums2))
    diff_nums = float(len(nums1 ^ nums2))
    has_conflict = 1.0 if (len(nums1) > 0 and len(nums2) > 0 and common_nums == 0) else 0.0
    
    # 17. Postal code match (1.0 = match, -1.0 = conflict, 0.0 = missing)
    p1 = s1_rec.get('postal_code', '')
    p2 = cand_rec.get('postal_code', '')
    if p1 and p2:
        postal_match = 1.0 if p1 == p2 else -1.0
    else:
        postal_match = 0.0
        
    # 18. Distinctive address signature overlap
    addr_sig_overlap = compute_overlap_ratio(s1_rec.get('address_sig', []), cand_rec.get('address_sig', []))
    
    # 19-20. Length differences
    len_diff_name = abs(len(n1) - len(n2)) / (max(len(n1), len(n2)) + 1)
    len_diff_addr = abs(len(a1) - len(a2)) / (max(len(a1), len(a2)) + 1)
    
    # 21. Combined text token set ratio
    comb1 = n1 + " " + a1
    comb2 = n2 + " " + a2
    comb_set = fuzz.token_set_ratio(comb1, comb2) / 100.0
    
    # 22. Country agreement (open-set comparison)
    country_match = 1.0 if s1_rec.get('country', '').lower() == cand_rec.get('country', '').lower() else 0.0
    
    # 23. Interaction: Strong name + Strong address
    strong_both = 1.0 if (name_jw >= 0.85 and addr_set >= 0.75) else 0.0
    
    # 24. Interaction: Exact name + Conflicting address (Chain/Branch penalty)
    exact_name_conflict = 1.0 if (name_ratio >= 0.90 and has_conflict == 1.0) else 0.0
    
    # 25. High address exact location (rescues Trade Name / DBA with identical address)
    high_addr_location = 1.0 if (addr_sort >= 0.88 and common_nums >= 1.0 and has_conflict == 0.0) else 0.0
    
    # 26. Empty address flag
    both_empty_addr = 1.0 if (not a1 and not a2) else 0.0
    
    # 27. Blocker score prior
    prior_score = float(cand_score)
    
    return [
        name_ratio, name_partial, name_sort, name_set, name_jw,
        name_jaccard, name_overlap_ratio, name_nosuf_ratio, name_first_tok_ratio,
        addr_ratio, addr_sort, addr_set, addr_jaccard,
        common_nums, diff_nums, has_conflict, postal_match, addr_sig_overlap,
        len_diff_name, len_diff_addr, comb_set,
        country_match, strong_both, exact_name_conflict,
        high_addr_location, both_empty_addr,
        prior_score
    ]

FEATURE_NAMES = [
    'name_ratio', 'name_partial', 'name_sort', 'name_set', 'name_jw',
    'name_jaccard', 'name_overlap_ratio', 'name_nosuf_ratio', 'name_first_tok_ratio',
    'addr_ratio', 'addr_sort', 'addr_set', 'addr_jaccard',
    'common_nums', 'diff_nums', 'has_conflict', 'postal_match', 'addr_sig_overlap',
    'len_diff_name', 'len_diff_addr', 'comb_set',
    'country_match', 'strong_both', 'exact_name_conflict',
    'high_addr_location', 'both_empty_addr',
    'prior_score'
]
