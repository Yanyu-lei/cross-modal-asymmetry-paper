"""Apply seeded image and word corruptions using configs/corruption_severities.yaml."""
from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import yaml
from PIL import Image, ImageFilter

from src import file_paths

with open(file_paths.CORRUPTION_SEVERITIES) as f:
    _STRENGTHS = yaml.safe_load(f)


def strength(side: str, kind: str, severity: int) -> float:
    """Return the parameter for a corruption at the given severity."""
    if severity == 0:
        raise ValueError("severity 0 is the clean input and has no strength")
    if side == "text":
        return float(_STRENGTHS["text_word_share"][severity])
    return float(_STRENGTHS["image"][kind][severity])


# Keep these formulas fixed so runs, damage replays and pooling use the same corruptions.
# Seeds formed with bit shifts can collide, as for seed 1's first pair and seed 0's 257th.
# Changing the constants to remove these collisions would change the stored results.
def single_image_rng(seed: int, pair_index: int, severity: int) -> random.Random:
    """Return one pair's image-corruption generator for a fidelity-retention run."""
    return random.Random((seed << 16) ^ (pair_index << 8) ^ severity)


def single_text_rng(seed: int, pair_index: int, severity: int) -> random.Random:
    """Return the caption-corruption generator shared by fidelity and pooling runs."""
    return random.Random((seed << 16) ^ (pair_index << 8) ^ severity ^ 0xCAFE)


def joint_grid_rngs(seed: int, pair_index: int, image_severity: int, text_severity: int) -> tuple[random.Random, random.Random]:
    """Return the image and caption generators for one noise-mask grid cell."""
    grid_seed = (seed << 24) ^ (pair_index << 12) ^ (image_severity << 6) ^ text_severity
    return random.Random(grid_seed ^ 0xA1), random.Random(grid_seed ^ 0xB2)


def retrieval_rngs(seed: int, severity: int, n: int, side: str) -> list[random.Random]:
    """Return query generators whose initial sequence is independent of the gallery size.

    Nested galleries therefore use identical corruptions for their shared queries.
    """
    master = random.Random(seed * (31 if side == "image" else 53) + severity)
    return [random.Random(master.randint(0, 2**31)) for _ in range(n)]


def pair_rngs(run_kind: str, seed: int, severity: int, n: int, side: str) -> list[random.Random]:
    """Return one generator per pair for the given run type, modality and severity."""
    if run_kind == "retrieval":
        return retrieval_rngs(seed, severity, n, side)
    make = single_image_rng if side == "image" else single_text_rng
    return [make(seed, i, severity) for i in range(n)]


# Image corruptions preserve the input dimensions.
def gaussian_noise(img: Image.Image, severity: int, *, rng: random.Random) -> Image.Image:
    sigma = strength("image", "gaussian_noise", severity)
    pixels = np.asarray(img, dtype=np.float32)
    noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0.0, sigma, size=pixels.shape)
    return Image.fromarray(np.clip(pixels + noise.astype(np.float32), 0, 255).astype(np.uint8))


def gaussian_blur(img: Image.Image, severity: int, *, rng: random.Random) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=strength("image", "gaussian_blur", severity)))


def cutout(img: Image.Image, severity: int, *, rng: random.Random) -> Image.Image:
    """Black out a randomly placed rectangle with the requested target area.

    Capping its height without widening it can undershoot the target on wide images.
    """
    w, h = img.size
    area = max(1, int(w * h * strength("image", "cutout", severity)))
    cw = max(1, min(w, int(np.sqrt(area))))
    ch = max(1, min(h, max(1, area // cw)))
    x0 = rng.randint(0, max(0, w - cw))
    y0 = rng.randint(0, max(0, h - ch))
    pixels = np.asarray(img).copy()
    pixels[y0:y0 + ch, x0:x0 + cw, :] = 0
    return Image.fromarray(pixels)


def object_mask(segmentation, height: int, width: int) -> np.ndarray:
    """Decode COCO polygons or run-length counts into a boolean mask of shape (height, width).

    Use the reference rasterizer because Pillow includes different boundary pixels.
    """
    from pycocotools import mask as coco_mask

    if isinstance(segmentation, list):
        polygons = [p for p in segmentation if len(p) >= 6]
        if not polygons:
            return np.zeros((height, width), dtype=bool)
        rle = coco_mask.merge(coco_mask.frPyObjects(polygons, height, width))
    elif isinstance(segmentation["counts"], list):
        rle = coco_mask.frPyObjects(segmentation, height, width)
    else:
        # Compressed RLE can be decoded directly.
        counts = segmentation["counts"]
        rle = {"size": segmentation["size"], "counts": counts.encode() if isinstance(counts, str) else counts}
    return coco_mask.decode(rle).astype(bool)


def object_cutout_mask(width: int, height: int, severity: int, *, rng: random.Random, objects: Sequence) -> np.ndarray:
    """Combine whole-object masks in random order until the target is reached or no objects remain.

    The targets use the rectangle cutout's nominal area shares. Whole objects can
    overshoot the target or leave it unmet, so area matching is approximate.
    """
    target = strength("image", "object_cutout", severity) * width * height
    order = list(range(len(objects)))
    rng.shuffle(order)
    covered = np.zeros((height, width), dtype=bool)
    for i in order:
        if covered.sum() >= target:
            break
        covered |= object_mask(objects[i], height, width)
    return covered


def object_cutout(img: Image.Image, severity: int, *, rng: random.Random, objects: Sequence) -> Image.Image:
    """Black out the pixels selected by object_cutout_mask."""
    covered = object_cutout_mask(*img.size, severity, rng=rng, objects=objects)
    pixels = np.asarray(img).copy()
    pixels[covered] = 0
    return Image.fromarray(pixels)


_IMAGE_CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
    "cutout": cutout,
    "object_cutout": object_cutout,
}


def corrupt_image(img: Image.Image, kind: str, severity: int, *, rng: random.Random, objects=None) -> Image.Image:
    """Apply an image corruption, using object outlines when object cutout is requested."""
    if severity == 0:
        return img.copy()
    if kind == "object_cutout":
        return object_cutout(img, severity, rng=rng, objects=objects)
    return _IMAGE_CORRUPTIONS[kind](img, severity, rng=rng)


# Corrupt whitespace-separated words before each model tokenizes the caption.
# Selection need not change a word, since shuffle and replacement can leave it unchanged.
# Replacements need not remain grammatical or meaningful. Keep the duplicate "harbor",
# since removing it changes the seeded draws.
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

# Each key lists neighboring QWERTY keys but excludes itself, so substitutions change it.
_QWERTY_NEIGHBORS: dict[str, str] = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "ryfg", "y": "tugh",
    "u": "yihj", "i": "uojk", "o": "ipkl", "p": "ol",
    "a": "qwsz", "s": "wedxza", "d": "erfcxs", "f": "rtgvcd", "g": "tyhbvf",
    "h": "yujnbg", "j": "uikmnh", "k": "iolmj", "l": "opk",
    "z": "asx", "x": "sdzc", "c": "dfxv", "v": "fgcb", "b": "ghvn",
    "n": "hjbm", "m": "jkn",
}


def _n_words_changed(severity: int, n_words: int) -> int:
    if n_words == 0:
        return 0
    return max(1, min(n_words, round(strength("text", "", severity) * n_words)))


def _chosen_words(text: str, severity: int, rng: random.Random) -> tuple[list[str], list[int]]:
    """Return the caption's whitespace-separated words and the sampled positions."""
    words = text.split()
    k = _n_words_changed(severity, len(words))
    return words, (rng.sample(range(len(words)), k) if k else [])


def mask_text(text: str, severity: int, *, rng: random.Random) -> str:
    """Delete the chosen words."""
    words, positions = _chosen_words(text, severity, rng)
    if not positions:
        return text
    dropped = set(positions)
    return " ".join(w for i, w in enumerate(words) if i not in dropped)


def replace_text(text: str, severity: int, *, rng: random.Random) -> str:
    """Replace each selected word with a draw from the fixed vocabulary."""
    words, positions = _chosen_words(text, severity, rng)
    if not positions:
        return text
    for i in positions:
        words[i] = rng.choice(_REPLACE_VOCAB)
    return " ".join(words)


def _misspell(word: str, rng: random.Random) -> str:
    """Apply an adjacent-key substitution or insertion, a deletion, or an adjacent swap.

    Substitutions preserve case; words without supported letters stay unchanged.
    """
    letters = [i for i, ch in enumerate(word) if ch.lower() in _QWERTY_NEIGHBORS]
    if not letters:
        return word
    i = rng.choice(letters)
    ch = word[i]
    sub = rng.choice(_QWERTY_NEIGHBORS[ch.lower()])
    if ch.isupper():
        sub = sub.upper()
    op = rng.choice(["substitute", "insert"] + (["delete", "transpose"] if len(word) >= 2 else []))
    if op == "substitute":
        out = word[:i] + sub + word[i + 1:]
    elif op == "insert":
        out = word[:i + 1] + sub.lower() + word[i + 1:]
    elif op == "delete":
        out = word[:i] + word[i + 1:]
    else:
        j = i + 1 if i + 1 < len(word) else i - 1
        a, b = min(i, j), max(i, j)
        out = word[:a] + word[b] + word[a] + word[b + 1:]
    if out == word:
        # Swapping two identical letters changes nothing, so fall back to a substitution.
        out = word[:i] + sub + word[i + 1:]
    return out


def typo_text(text: str, severity: int, *, rng: random.Random) -> str:
    """Misspell each chosen word once."""
    words, positions = _chosen_words(text, severity, rng)
    if not positions:
        return text
    for i in positions:
        words[i] = _misspell(words[i], rng)
    return " ".join(words)


def shuffle_text(text: str, severity: int, *, rng: random.Random) -> str:
    """Shuffle the words at selected positions, choosing at least two when available.

    A random permutation may leave the caption unchanged.
    """
    words = text.split()
    if len(words) < 2:
        return text
    k = min(len(words), max(2, _n_words_changed(severity, len(words))))
    positions = rng.sample(range(len(words)), k)
    moved = [words[i] for i in positions]
    rng.shuffle(moved)
    for i, w in zip(positions, moved):
        words[i] = w
    return " ".join(words)


_TEXT_CORRUPTIONS = {
    "mask": mask_text,
    "shuffle": shuffle_text,
    "replace": replace_text,
    "typo": typo_text,
}


def corrupt_caption(text: str, kind: str, severity: int, *, rng: random.Random) -> str:
    if severity == 0:
        return text
    return _TEXT_CORRUPTIONS[kind](text, severity, rng=rng)
