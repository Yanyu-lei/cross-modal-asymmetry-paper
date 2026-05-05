"""Text corruptions, applied at the WORD level on raw caption strings.

Word-level (rather than BPE-token-level) keeps the corruption model-agnostic:
the same severity-2 mask produces the same string for every model. Each model
then BPE-tokenizes the corrupted string with its own tokenizer.

Conventions:
- "mask"    : delete K random words (drop them, condense whitespace).
- "shuffle" : permute K random word positions (preserves word multiset).
- "replace" : replace K random words with random words from a fixed English
              wordlist (so the corruption is reproducible and model-agnostic).

K = round(severity_fraction * n_words), with a floor of 1 if severity > 0 and
n_words >= 1, so even short captions feel each severity step.
"""
from __future__ import annotations

import random
from typing import List

from .severity import lookup

# Fixed English wordlist for "replace". Curated common nouns/verbs/adjectives;
# scope-bounded so the corruption produces caption-shaped (but unrelated) text.
_REPLACE_VOCAB: tuple[str, ...] = (
    "table", "river", "valley", "engine", "garden", "ladder", "harbor", "village",
    "kitten", "quartz", "ribbon", "thunder", "marble", "feather", "lantern", "saddle",
    "mountain", "pavement", "kettle", "crater", "orchard", "trumpet", "satellite",
    "blanket", "biscuit", "compass", "elephant", "diamond", "anchor", "harvest",
    "running", "shouting", "carving", "drifting", "glowing", "humming", "pouring",
    "racing", "swirling", "trembling", "chasing", "leaping", "sleeping", "vanishing",
    "ancient", "brittle", "crooked", "dusty", "elegant", "frozen", "golden", "hollow",
    "jagged", "lonely", "muddy", "narrow", "quiet", "ragged", "sparse", "tangled",
    "bright", "calm", "dim", "empty", "fierce", "gentle", "humid", "icy", "joyful",
    "blue", "scarlet", "amber", "violet", "olive", "ivory", "indigo", "rust",
    "cottage", "subway", "harbor", "library", "stadium", "plaza", "alley", "rooftop",
    "tiger", "salmon", "raven", "rabbit", "lobster", "buffalo", "owl", "lizard",
    "carpet", "mirror", "candle", "window", "doorway", "balcony", "fountain",
    "pebble", "boulder", "clay", "shale", "moss", "ivy", "fern", "thistle",
    "umbrella", "sandal", "scarf", "jacket", "necklace", "bracelet", "watch",
    "radio", "guitar", "violin", "piano", "drum", "harp", "banjo", "flute",
    "captain", "doctor", "farmer", "painter", "soldier", "teacher", "weaver",
    "swiftly", "loudly", "quietly", "firmly", "gently", "suddenly", "rarely",
    "above", "below", "beyond", "beside", "between", "around", "inside", "outside",
    "yesterday", "tomorrow", "tonight", "midnight", "morning", "afternoon", "dusk",
    "smoke", "steam", "fog", "mist", "rain", "snow", "hail", "frost", "dew",
    "shadow", "sparkle", "glimmer", "shimmer", "ripple", "echo", "rumble", "whisper",
)


def _split_words(text: str) -> List[str]:
    """Whitespace split. Punctuation stays glued to the adjacent word, which is
    fine for our purposes (we're corrupting semantics, not parsing syntax)."""
    return text.split()


def _join_words(words: List[str]) -> str:
    return " ".join(w for w in words if w)


def _k_for(severity: int, n_words: int, frac: float) -> int:
    if n_words == 0 or severity == 0:
        return 0
    return max(1, min(n_words, round(frac * n_words)))


def mask_text(text: str, severity: int, *, rng: random.Random) -> str:
    if severity == 0:
        return text
    frac = lookup("text", "mask", severity)
    words = _split_words(text)
    k = _k_for(severity, len(words), frac)
    if k == 0:
        return text
    drop = set(rng.sample(range(len(words)), k))
    return _join_words([w for i, w in enumerate(words) if i not in drop])


def shuffle_text(text: str, severity: int, *, rng: random.Random) -> str:
    if severity == 0:
        return text
    frac = lookup("text", "shuffle", severity)
    words = _split_words(text)
    if len(words) < 2:
        return text
    # Shuffle requires at least 2 positions to do anything; round-down severities
    # produce k=1 on short captions, which is a no-op. Lift the floor to 2 so
    # the operation is always non-trivial when severity > 0 and the caption has
    # at least two words.
    k = max(2, _k_for(severity, len(words), frac))
    k = min(k, len(words))
    positions = rng.sample(range(len(words)), k)
    pool = [words[i] for i in positions]
    rng.shuffle(pool)
    out = list(words)
    for i, w in zip(positions, pool):
        out[i] = w
    return _join_words(out)


def replace_text(text: str, severity: int, *, rng: random.Random) -> str:
    if severity == 0:
        return text
    frac = lookup("text", "replace", severity)
    words = _split_words(text)
    k = _k_for(severity, len(words), frac)
    if k == 0:
        return text
    positions = rng.sample(range(len(words)), k)
    out = list(words)
    for i in positions:
        out[i] = rng.choice(_REPLACE_VOCAB)
    return _join_words(out)


CORRUPTORS = {
    "mask": mask_text,
    "shuffle": shuffle_text,
    "replace": replace_text,
}


def apply_text_corruption(
    text: str,
    corruption_type: str,
    severity: int,
    *,
    rng: random.Random,
) -> str:
    if corruption_type == "none" or severity == 0:
        return text
    if corruption_type not in CORRUPTORS:
        raise KeyError(f"unknown text corruption {corruption_type!r}")
    return CORRUPTORS[corruption_type](text, severity, rng=rng)
