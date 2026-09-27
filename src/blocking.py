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
    def __init__(self, df_cap=3000, max_candidates=25, use_char_ngrams=False, min_score=3, relative_margin=0.35, dynamic_pruning=True):
        self.df_cap = df_cap
        self.max_candidates = max_candidates
        self.use_char_ngrams = use_char_ngrams
        self.min_score = min_score
        self.relative_margin = relative_margin
        self.dynamic_pruning = dynamic_pruning
        
        # Inverted index channels
        self.name_exact_idx = defaultdict(list)
        self.name_nosuffix_idx = defaultdict(list)
        self.name_compact_idx = defaultdict(list)
        self.name_sorted_idx = defaultdict(list)
        self.name_token_idx = defaultdict(list)
        self.name_char_ngram_idx = defaultdict(list) if use_char_ngrams else None
        self.addr_token_idx = defaultdict(list)
        self.num_word_idx = defaultdict(list)
        self.addr_num_word_idx = defaultdict(list)
        self.rare_addr_word_idx = defaultdict(list)
        
        self.char_ngram_df = Counter() if use_char_ngrams else None

    def add_target(self, tid, rec):
        """
        Incrementally indexes a single target record into sparse inverted channels.
        Automatically caps hub tokens for zero-overhead streaming indexing.
        """
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
            l = self.name_token_idx[w]
            if len(l) < self.df_cap:
                l.append(tid)
                
        # Channel 6: Character n-grams (optional)
        if self.use_char_ngrams and n_nosuf:
            for ng in set(get_char_ngrams(n_nosuf, 3)):
                l = self.name_char_ngram_idx[ng]
                if len(l) < 1500:
                    l.append(tid)
                    
        # Channel 7: Numeric address tokens & building/PIN anchors
        for w in rec.get('address_tokens', []):
            if any(c.isdigit() for c in w):
                l = self.addr_token_idx[w]
                if len(l) < self.df_cap:
                    l.append(tid)
                
        # Channel 8: Composite (Number, First Word) anchor
        nums = rec.get('numbers', set())
        if nums and toks:
            first_w = toks[0]
            for num in nums:
                l = self.num_word_idx[(num, first_w)]
                if len(l) < 500:
                    l.append(tid)

        # Channel 9: Address-Dominant Composite Channel (Number, Address Word)
        # Rescues DBA and trade names sharing building/shop/suite
        sig_addr = rec.get('address_sig', [])
        if nums and sig_addr:
            for num in nums:
                for aw in sig_addr[:3]:
                    l = self.addr_num_word_idx[(num, aw)]
                    if len(l) < 200:
                        l.append(tid)

        # Channel 10: Rare address tokens (Plaza, locality, estate names)
        for aw in sig_addr:
            l = self.rare_addr_word_idx[aw]
            if len(l) < 150:
                l.append(tid)

    def fit_targets(self, targets):
        """
        Indexes target records (S2 and S3) for a specific country or open set.
        """
        for tid, rec in targets.items():
            self.add_target(tid, rec)

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
            if w in self.name_token_idx:
                df = len(self.name_token_idx[w])
                if 0 < df <= self.df_cap:
                    weight = 5 if df <= 30 else (3 if df <= 200 else 1)
                    for tid in self.name_token_idx[w]:
                        if tid != s1_id:
                            cand_scores[tid] += weight

        # Pass 6: Character 3-gram retrieval (for spelling noise and OCR)
        if self.use_char_ngrams and n_nosuf:
            for ng in set(get_char_ngrams(n_nosuf, 3)):
                if ng in self.name_char_ngram_idx:
                    df = len(self.name_char_ngram_idx[ng])
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

        # Pass 8: Numeric address tokens & building/PIN anchors (IDF-weighted)
        for w in s1_rec.get('address_tokens', []):
            if any(c.isdigit() for c in w) and w in self.addr_token_idx:
                df = len(self.addr_token_idx[w])
                if 0 < df <= self.df_cap:
                    weight = 5 if df <= 30 else 3
                    for tid in self.addr_token_idx[w]:
                        if tid != s1_id:
                            cand_scores[tid] += weight

        # Pass 9: Address-Dominant Composite Channel (Number, Address Word)
        sig_addr = s1_rec.get('address_sig', [])
        if nums and sig_addr:
            for num in nums:
                for aw in sig_addr[:3]:
                    key = (num, aw)
                    if key in self.addr_num_word_idx:
                        for tid in self.addr_num_word_idx[key]:
                            if tid != s1_id:
                                cand_scores[tid] += 12

        # Pass 10: Rare address locality words
        for aw in sig_addr:
            if aw in self.rare_addr_word_idx:
                df = len(self.rare_addr_word_idx[aw])
                if 0 < df <= 50:
                    for tid in self.rare_addr_word_idx[aw]:
                        if tid != s1_id:
                            cand_scores[tid] += 3

        top_candidates = cand_scores.most_common(self.max_candidates)
        if not self.dynamic_pruning or not top_candidates:
            return top_candidates

        max_score = top_candidates[0][1]
        if max_score < self.min_score:
            return []

        threshold = max(self.min_score, max_score * self.relative_margin)
        filtered = [(tid, s) for tid, s in top_candidates if s >= threshold]
        
        # Tie-breaker candidate minimization:
        # If top candidate is dominant (score >= 28) and second candidate is weak (< 0.40 * max_score), truncate
        if len(filtered) > 2 and max_score >= 28 and filtered[1][1] < 0.40 * max_score:
            return filtered[:2]

        return filtered

    def retrieve_candidates(self, s1_rec):
        """
        Retrieves top candidate IDs for an S1 entity.
        """
        return [tid for tid, _ in self.retrieve_candidates_with_scores(s1_rec)]
