"""
Advanced Multi-Channel Blocking Engine (Role A).
Implements the full 7-channel retrieval strategy:
1. Exact normalized name
2. Suffix-normalized name
3. Compact name
4. Character n-gram shingle retrieval (soft character similarity)
5. Distinctive token inverted index (IDF-weighted token retrieval)
6. Token overlap retrieval
7. Address-aware alphanumeric anchor retrieval
Treats country as an open-set label.
"""
from collections import defaultdict, Counter

def get_char_ngrams(text, n=3):
    """Generates character n-grams from normalized text."""
    if len(text) < n:
        return [text] if text else []
    return [text[i:i+n] for i in range(len(text) - n + 1)]

class MultiChannelBlocker:
    def __init__(self, df_cap=3000, max_candidates=25, use_char_ngrams=False):
        self.df_cap = df_cap
        self.max_candidates = max_candidates
        self.use_char_ngrams = use_char_ngrams
        
        # Inverted index channels
        self.name_exact_idx = defaultdict(list)
        self.name_nosuffix_idx = defaultdict(list)
        self.name_compact_idx = defaultdict(list)
        self.name_sorted_idx = defaultdict(list)
        self.name_token_idx = defaultdict(list)
        self.name_char_ngram_idx = defaultdict(list) if use_char_ngrams else None
        self.addr_token_idx = defaultdict(list)
        self.num_word_idx = defaultdict(list)
        
        self.token_df = Counter()
        self.char_ngram_df = Counter() if use_char_ngrams else None
        self.indexed_targets = {}

    def fit_targets(self, targets):
        """
        Indexes target records (S2 and S3) for a specific country or open set.
        """
        self.indexed_targets = targets
        
        # 1. Estimate document frequencies for tokens
        for tid, rec in targets.items():
            toks = rec.get('name_tokens_nosuffix') or rec.get('name_tokens', [])
            for w in set(toks):
                self.token_df[('n', w)] += 1
            for w in set(rec.get('address_tokens', [])):
                self.token_df[('a', w)] += 1
                
            if self.use_char_ngrams:
                fcn = rec.get('name_norm_nosuffix', '')
                if fcn:
                    ngrams = set(get_char_ngrams(fcn, 3))
                    for ng in ngrams:
                        self.char_ngram_df[ng] += 1

        # 2. Build multi-channel inverted indexes
        for tid, rec in targets.items():
            n_norm = rec.get('name_norm', '')
            n_nosuf = rec.get('name_norm_nosuffix', '')
            n_compact = rec.get('name_compact', '')
            n_sorted = rec.get('name_sorted', '')
            
            # Channel 1: Exact clean name
            if n_norm:
                self.name_exact_idx[n_norm].append(tid)
            # Channel 2: No-suffix name
            if n_nosuf:
                self.name_nosuffix_idx[n_nosuf].append(tid)
            # Channel 3: Compact name
            if n_compact:
                self.name_compact_idx[n_compact].append(tid)
            # Channel 4: Sorted token name
            if n_sorted:
                self.name_sorted_idx[n_sorted].append(tid)
                
            # Channel 5: Distinctive name tokens
            toks = rec.get('name_tokens_nosuffix') or rec.get('name_tokens', [])
            for w in toks:
                if self.token_df[('n', w)] <= self.df_cap:
                    self.name_token_idx[w].append(tid)
                    
            # Channel 6: Character n-grams (optional)
            if self.use_char_ngrams and n_nosuf:
                for ng in set(get_char_ngrams(n_nosuf, 3)):
                    if 5 <= self.char_ngram_df[ng] <= 1500:
                        self.name_char_ngram_idx[ng].append(tid)
                        
            # Channel 7: Address tokens & alphanumeric building anchors
            for w in rec.get('address_tokens', []):
                if self.token_df[('a', w)] <= self.df_cap:
                    self.addr_token_idx[w].append(tid)
                    
            # Channel 8: Composite (Number, First Word) anchor
            nums = rec.get('numbers', set())
            if nums and toks:
                first_w = toks[0]
                if self.token_df[('n', first_w)] <= self.df_cap:
                    for num in nums:
                        self.num_word_idx[(num, first_w)].append(tid)

    def retrieve_candidates_with_scores(self, s1_rec):
        """
        Retrieves top candidate (ID, score) pairs for an S1 entity by unioning all channels.
        """
        cand_scores = Counter()
        s1_id = s1_rec.get('entity_id', '')

        # Pass 1: Exact normalized name (High precision anchor)
        n_norm = s1_rec.get('name_norm', '')
        if n_norm and n_norm in self.name_exact_idx:
            for tid in self.name_exact_idx[n_norm]:
                if tid != s1_id:
                    cand_scores[tid] += 20

        # Pass 2: No-suffix normalized name
        n_nosuf = s1_rec.get('name_norm_nosuffix', '')
        if n_nosuf and n_nosuf in self.name_nosuffix_idx:
            for tid in self.name_nosuffix_idx[n_nosuf]:
                if tid != s1_id:
                    cand_scores[tid] += 16

        # Pass 3: Compact name
        n_compact = s1_rec.get('name_compact', '')
        if n_compact and n_compact in self.name_compact_idx:
            for tid in self.name_compact_idx[n_compact]:
                if tid != s1_id:
                    cand_scores[tid] += 12

        # Pass 4: Sorted token name
        n_sorted = s1_rec.get('name_sorted', '')
        if n_sorted and n_sorted in self.name_sorted_idx:
            for tid in self.name_sorted_idx[n_sorted]:
                if tid != s1_id:
                    cand_scores[tid] += 12

        # Pass 5: Distinctive name tokens (IDF-weighted)
        toks = s1_rec.get('name_tokens_nosuffix') or s1_rec.get('name_tokens', [])
        for w in toks:
            df = self.token_df.get(('n', w), 0)
            if 0 < df <= self.df_cap:
                weight = 5 if df <= 30 else (3 if df <= 200 else 1)
                for tid in self.name_token_idx[w]:
                    if tid != s1_id:
                        cand_scores[tid] += weight

        # Pass 6: Character 3-gram retrieval (for spelling noise and OCR)
        if self.use_char_ngrams and n_nosuf:
            for ng in set(get_char_ngrams(n_nosuf, 3)):
                df = self.char_ngram_df.get(ng, 0)
                if 5 <= df <= 1500:
                    weight = 2 if df <= 100 else 1
                    for tid in self.name_char_ngram_idx[ng]:
                        if tid != s1_id:
                            cand_scores[tid] += weight

        # Pass 7: Composite (Number, First Word) anchor
        nums = s1_rec.get('numbers', set())
        if nums and toks:
            first_w = toks[0]
            for num in nums:
                key = (num, first_w)
                if key in self.num_word_idx:
                    for tid in self.num_word_idx[key]:
                        if tid != s1_id:
                            cand_scores[tid] += 10

        # Pass 8: Address tokens & alphanumeric building anchors (IDF-weighted)
        for w in s1_rec.get('address_tokens', []):
            df = self.token_df.get(('a', w), 0)
            if 0 < df <= self.df_cap:
                weight = 5 if any(c.isdigit() for c in w) else (3 if df <= 100 else 1)
                for tid in self.addr_token_idx[w]:
                    if tid != s1_id:
                        cand_scores[tid] += weight

        return cand_scores.most_common(self.max_candidates)

    def retrieve_candidates(self, s1_rec):
        """
        Retrieves top candidate IDs for an S1 entity.
        """
        return [tid for tid, _ in self.retrieve_candidates_with_scores(s1_rec)]
