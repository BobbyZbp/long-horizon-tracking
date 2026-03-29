import re
from typing import Dict, List

def load_spacy(model: str):
    import spacy
    return spacy.load(model)

def clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def normalize_entity(text: str) -> str:
    t = clean_spaces(text)
    t = re.sub(r"^(the|a|an|my|his|her|their)\s+", "", t, flags=re.I)
    t = re.sub(r"[^\w\s\-']+", "", t).strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t

def extract_mentions_spacy(doc, source: str) -> List[Dict]:
    recs = []
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ not in ("NOUN", "PROPN"):
            continue
        ent_norm = normalize_entity(chunk.text)
        if not ent_norm:
            continue
        sent = chunk.sent
        sent_text = clean_spaces(sent.text)
        verbs = sorted({t.lemma_.lower() for t in sent if t.pos_ in ("VERB", "AUX") and t.lemma_.isalpha()})
        co_persons = sorted({clean_spaces(e.text).upper() for e in sent.ents if e.label_ == "PERSON"})
        recs.append({
            "ent_norm": ent_norm,
            "ent_surface": chunk.text.strip(),
            "sent_text": sent_text,
            "verbs": verbs,
            "co_persons": co_persons,
            "source": source,
        })
    return recs