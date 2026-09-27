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
    'pc', 'pllc', 'federation', 'industries', 'international'
}

COMMON_ADDR_TERMS = {
    'road', 'street', 'drive', 'avenue', 'lane', 'floor', 'city', 'unit',
    'block', 'near', 'behind', 'beside', 'opposite', 'cross', 'main', 'rd',
    'st', 'ave', 'dr', 'pl', 'blvd', 'rue', 'de', 'du', 'la', 'des', 'north',
    'south', 'east', 'west', 'state', 'district', 'nagar', 'colony', 'shop',
    'bldg', 'building', 'ground', 'po', 'box', 'suite', 'ste', 'apt', 'c', 'o'
}

INDIC_OFFSET_MAP = {
    0x02: 'm', 0x03: 'h',
    0x05: 'a', 0x06: 'aa', 0x07: 'i', 0x08: 'ee', 0x09: 'u', 0x0A: 'oo',
    0x0B: 'ri', 0x0E: 'e', 0x0F: 'e', 0x10: 'ai', 0x12: 'o', 0x13: 'o', 0x14: 'au',
    0x15: 'k', 0x16: 'kh', 0x17: 'g', 0x18: 'gh', 0x19: 'ng',
    0x1A: 'ch', 0x1B: 'chh', 0x1C: 'j', 0x1D: 'jh', 0x1E: 'ny',
    0x1F: 't', 0x20: 'th', 0x21: 'd', 0x22: 'dh', 0x23: 'n',
    0x24: 't', 0x25: 'th', 0x26: 'd', 0x27: 'dh', 0x28: 'n',
    0x2A: 'p', 0x2B: 'ph', 0x2C: 'b', 0x2D: 'bh', 0x2E: 'm',
    0x2F: 'y', 0x30: 'r', 0x31: 'r', 0x32: 'l', 0x33: 'l', 0x35: 'v',
    0x36: 'sh', 0x37: 'sh', 0x38: 's', 0x39: 'h',
    0x3E: 'aa', 0x3F: 'i', 0x40: 'ee', 0x41: 'u', 0x42: 'oo', 0x43: 'ri',
    0x46: 'e', 0x47: 'e', 0x48: 'ai', 0x4A: 'o', 0x4B: 'o', 0x4C: 'au',
    0x4D: '',
}

def transliterate_indic(text):
    if not text:
        return ""
    s = str(text)
    if s.isascii():
        return s
    out = []
    for ch in s:
        code = ord(ch)
        if 0x0900 <= code <= 0x0D7F:
            base = 0x0900 + ((code - 0x0900) // 0x80) * 0x80
            offset = code - base
            out.append(INDIC_OFFSET_MAP.get(offset, ''))
        else:
            out.append(ch)
    return ''.join(out)

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
    text = transliterate_indic(str(text))
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
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
    addr_sig = [w for w in tokens if w not in COMMON_ADDR_TERMS and len(w) >= 4 and not w.isdigit()]
    compact = re.sub(r'\s+', '', norm)
    numbers = set(re.findall(r'\b\d+[a-z]?\b', norm))
    
    # Extract postal code (6 digits in India, 5 digits in US/France)
    postal_matches = re.findall(r'\b\d{5,6}\b', norm)
    postal_code = postal_matches[0] if postal_matches else ""
    
    return {
        'address_raw': raw_addr or "",
        'address_norm': norm,
        'address_tokens': addr_tokens,
        'address_sig': addr_sig,
        'address_compact': compact,
        'numbers': numbers,
        'postal_code': postal_code
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
