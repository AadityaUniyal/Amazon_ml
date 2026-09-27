"""
Multi-View Normalization Engine for Business Entity Resolution.
Preserves raw values while exposing multiple canonical views:
- normalized (case, punctuation, whitespace, unicode)
- without legal suffixes (Pvt Ltd, LLC, Corp, SARL, SAS)
- compact (no whitespace, character-level comparison)
- token sets (with numbers and distinctive words preserved)
"""
import re
import unicodedata

LEGAL_TERMS = {
    'limited', 'ltd', 'private', 'pvt', 'llc', 'inc', 'corp', 'corporation',
    'llp', 'co', 'company', 'services', 'enterprises', 'holding', 'holdings',
    'sarl', 'sas', 'sasu', 'eurl', 'sci', 'sa', 'cie', 'lnc', 'd b a', 'dba',
    'associates', 'solutions', 'consultancy', 'agency', 'agencies', 'group',
    'pc', 'pllc'
}

COMMON_ADDR_TERMS = {
    'road', 'street', 'drive', 'avenue', 'lane', 'floor', 'city', 'unit',
    'block', 'near', 'behind', 'beside', 'opposite', 'cross', 'main', 'rd',
    'st', 'ave', 'dr', 'pl', 'blvd', 'rue', 'de', 'du', 'la', 'des', 'north',
    'south', 'east', 'west', 'state', 'district', 'nagar', 'colony'
}

def collapse_initials(text):
    tokens = text.split()
    res = []
    i = 0
    while i < len(tokens):
        if len(tokens[i]) == 1 and tokens[i].isalnum():
            seq = [tokens[i]]
            j = i + 1
            while j < len(tokens) and len(tokens[j]) == 1 and tokens[j].isalnum():
                seq.append(tokens[j])
                j += 1
            if len(seq) > 1:
                res.append(''.join(seq))
                i = j
                continue
        res.append(tokens[i])
        i += 1
    return ' '.join(res)

def normalize_text(text):
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+|\.(com|in|org|net|fr|co)\b', '', text)
    text = re.sub(r'\bm/s\b|\bm\s+s\b|\bd/b/a\b|\bd\s+b\s+a\b|\bdba\b', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = collapse_initials(text)
    return " ".join(text.split())

def normalize_business_name(raw_name):
    norm = normalize_text(raw_name)
    tokens = norm.split()
    no_suffix_tokens = [w for w in tokens if w not in LEGAL_TERMS and len(w) >= 2]
    norm_nosuffix = " ".join(no_suffix_tokens) if no_suffix_tokens else norm
    compact = re.sub(r'\s+', '', norm_nosuffix)
    sorted_name = " ".join(sorted(no_suffix_tokens)) if no_suffix_tokens else norm
    return {
        'name_raw': raw_name or "",
        'name_norm': norm,
        'name_norm_nosuffix': norm_nosuffix,
        'name_tokens': tokens,
        'name_tokens_nosuffix': no_suffix_tokens,
        'name_compact': compact,
        'name_sorted': sorted_name
    }

def normalize_business_address(raw_addr):
    norm = normalize_text(raw_addr)
    tokens = norm.split()
    addr_tokens = [w for w in tokens if w not in COMMON_ADDR_TERMS and (len(w) >= 3 or any(c.isdigit() for c in w))]
    compact = re.sub(r'\s+', '', norm)
    numbers = set(re.findall(r'\b\d+[a-z]?\b', norm))
    return {
        'address_raw': raw_addr or "",
        'address_norm': norm,
        'address_tokens': addr_tokens,
        'address_compact': compact,
        'numbers': numbers
    }

def normalize_country(raw_country):
    """Normalizes country string as an open-set label (no whitelist)."""
    if not raw_country:
        return "unknown"
    return str(raw_country).strip().lower()

def normalize_record(entity_id, name, addr, country):
    n_views = normalize_business_name(name)
    a_views = normalize_business_address(addr)
    c_norm = normalize_country(country)
    return {
        'entity_id': entity_id,
        'country': c_norm,
        **n_views,
        **a_views
    }
