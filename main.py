import hashlib
import logging
import math
import os
import random
import struct
import sys
import time
from pathlib import Path
from typing import Callable, Generator, Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend for file output
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# SECTION 1 – HASH FUNCTIONS
# ===========================================================================


def _murmurhash3_32(data: bytes, seed: int = 0) -> int:
    """MurmurHash3 32-bit implementation for fast, high-quality hashing.

    Args:
        data: Raw bytes to hash.
        seed: Seed value for independent hash families.

    Returns:
        A 32-bit unsigned integer hash value.
    """
    # MurmurHash3 constants
    c1, c2 = 0xCC9E2D51, 0x1B873593
    h1 = seed & 0xFFFFFFFF
    length = len(data)

    # Process 4-byte blocks
    nblocks = length // 4
    for block_start in range(0, nblocks * 4, 4):
        k1 = struct.unpack_from("<I", data, block_start)[0]
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF  # ROTL32(k1,15)
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1
        h1 = ((h1 << 13) | (h1 >> 19)) & 0xFFFFFFFF  # ROTL32(h1,13)
        h1 = ((h1 * 5) + 0xE6546B64) & 0xFFFFFFFF

    # Handle remaining bytes (tail)
    tail_start = nblocks * 4
    tail = data[tail_start:]
    k1 = 0
    tail_size = length & 3
    if tail_size >= 3:
        k1 ^= tail[2] << 16
    if tail_size >= 2:
        k1 ^= tail[1] << 8
    if tail_size >= 1:
        k1 ^= tail[0]
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1

    # Final mix (avalanche)
    h1 ^= length
    h1 ^= h1 >> 16
    h1 = (h1 * 0x85EBCA6B) & 0xFFFFFFFF
    h1 ^= h1 >> 13
    h1 = (h1 * 0xC2B2AE35) & 0xFFFFFFFF
    h1 ^= h1 >> 16
    return h1


def _fnv1a_32(data: bytes, seed: int = 0) -> int:
    """FNV-1a 32-bit hash — fast alternative with good distribution.

    Args:
        data: Raw bytes to hash.
        seed: XOR'd into the initial basis for seeding.

    Returns:
        A 32-bit unsigned integer hash value.
    """
    FNV_PRIME = 0x01000193
    FNV_OFFSET = 0x811C9DC5 ^ (seed & 0xFFFF)
    h = FNV_OFFSET & 0xFFFFFFFF
    for byte in data:
        h ^= byte
        h = (h * FNV_PRIME) & 0xFFFFFFFF
    return h


def _sha256_hash(data: bytes, seed: int = 0) -> int:
    """SHA-256 based hash — cryptographically strong but slower.

    Uses only the first 4 bytes of the digest as a 32-bit value.

    Args:
        data: Raw bytes to hash.
        seed: Appended to data as a seed discriminator.

    Returns:
        A 32-bit unsigned integer hash value.
    """
    seeded = data + seed.to_bytes(4, "little")
    digest = hashlib.sha256(seeded).digest()
    return struct.unpack_from("<I", digest, 0)[0]


# Registry: maps strategy name → hash callable
HASH_STRATEGIES: dict[str, Callable[[bytes, int], int]] = {
    "murmur3": _murmurhash3_32,
    "fnv1a": _fnv1a_32,
    "sha256": _sha256_hash,
}


def get_hash_positions(
    item: str,
    bit_size: int,
    k: int,
    hash_fn: Callable[[bytes, int], int],
) -> list[int]:
    """Compute k independent bit positions for a given item.

    Uses the double-hashing technique:  pos_i = (h1 + i * h2) % m
    This avoids calling k distinct hash functions while maintaining
    statistical independence.

    Args:
        item: The string element to hash.
        bit_size: Total number of bits in the filter (m).
        k: Number of hash functions (positions) to compute.
        hash_fn: The underlying hash primitive to use.

    Returns:
        A list of k bit indices in the range [0, bit_size).

    Raises:
        ValueError: If bit_size or k are non-positive.
    """
    if bit_size <= 0:
        raise ValueError(f"bit_size must be positive, got {bit_size}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    encoded = item.encode("utf-8", errors="replace")
    h1 = hash_fn(encoded, seed=0)
    h2 = hash_fn(encoded, seed=1)

    # Double-hashing: each position uses a linear combination of h1 and h2
    return [(h1 + i * h2) % bit_size for i in range(k)]


# ===========================================================================
# SECTION 2 – BIT ARRAY
# ===========================================================================


def create_bit_array(size: int) -> bytearray:
    """Allocate a zero-initialised bit array of the requested size.

    Args:
        size: Number of bits required.

    Returns:
        A bytearray large enough to hold `size` bits, all set to 0.

    Raises:
        ValueError: If size is not a positive integer.
        MemoryError: If the system cannot allocate the required memory.
    """
    if size <= 0:
        raise ValueError(f"Bit array size must be positive, got {size}")
    byte_count = (size + 7) // 8
    try:
        return bytearray(byte_count)
    except MemoryError as exc:
        raise MemoryError(
            f"Cannot allocate {byte_count / 1e6:.1f} MB for bit array"
        ) from exc


def set_bit(bit_array: bytearray, index: int) -> None:
    """Set a single bit to 1 at the given index.

    Args:
        bit_array: The mutable bytearray backing the filter.
        index: Zero-based bit index to set.
    """
    bit_array[index >> 3] |= 1 << (index & 7)


def get_bit(bit_array: bytearray, index: int) -> bool:
    """Read a single bit at the given index.

    Args:
        bit_array: The bytearray backing the filter.
        index: Zero-based bit index to read.

    Returns:
        True if the bit is set, False otherwise.
    """
    return bool(bit_array[index >> 3] & (1 << (index & 7)))


def count_set_bits(bit_array: bytearray) -> int:
    """Count all set bits using the popcount approach.

    Uses Python's built-in `bin().count('1')` which delegates to the
    CPU's POPCNT instruction on supported platforms via CPython internals.

    Args:
        bit_array: The bytearray to count set bits in.

    Returns:
        Total number of bits set to 1.
    """
    return sum(bin(byte).count("1") for byte in bit_array)


# ===========================================================================
# SECTION 3 – BLOOM FILTER CORE
# ===========================================================================


def optimal_parameters(n: int, fpr: float) -> tuple[int, int]:
    """Calculate the theoretically optimal Bloom filter dimensions.

    Formulas derived from information-theoretic analysis:
      m = -n * ln(p) / (ln(2)^2)
      k = (m / n) * ln(2)

    Args:
        n: Expected number of elements to insert.
        fpr: Desired false positive rate (0 < fpr < 1).

    Returns:
        A tuple (m, k) where m is the bit array size and k is the
        number of hash functions.

    Raises:
        ValueError: If n <= 0 or fpr is outside (0, 1).
    """
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")
    if not (0 < fpr < 1):
        raise ValueError(f"fpr must be in (0, 1), got {fpr}")

    m = math.ceil(-n * math.log(fpr) / (math.log(2) ** 2))
    k = max(1, round((m / n) * math.log(2)))
    return m, k


def create_bloom_filter(
    capacity: int,
    fpr: float = 0.01,
    hash_strategy: str = "murmur3",
) -> dict:
    """Initialise a new Bloom filter as a plain dictionary (no class needed).

    The filter state is a dict so it can be freely inspected, serialised,
    and passed around without OOP overhead.

    Args:
        capacity: Maximum number of elements to support (n).
        fpr: Target false positive rate.
        hash_strategy: One of 'murmur3', 'fnv1a', or 'sha256'.

    Returns:
        A dict containing:
            - bit_array: bytearray backing store
            - m: bit array size
            - k: number of hash positions
            - capacity: maximum expected elements
            - fpr_target: desired false positive rate
            - hash_fn: the hash callable
            - hash_strategy: strategy name string
            - count: number of inserted elements (0)
            - insert_times: list of per-insert durations (ns)
            - lookup_times: list of per-lookup durations (ns)

    Raises:
        ValueError: For invalid capacity, fpr, or hash_strategy.
    """
    if hash_strategy not in HASH_STRATEGIES:
        raise ValueError(
            f"Unknown hash strategy '{hash_strategy}'. "
            f"Choose from: {list(HASH_STRATEGIES)}"
        )
    m, k = optimal_parameters(capacity, fpr)
    logger.info(
        "Bloom filter created — capacity=%d, fpr=%.4f, m=%d bits (%.2f MB), k=%d",
        capacity,
        fpr,
        m,
        m / 8e6,
        k,
    )
    return {
        "bit_array": create_bit_array(m),
        "m": m,
        "k": k,
        "capacity": capacity,
        "fpr_target": fpr,
        "hash_fn": HASH_STRATEGIES[hash_strategy],
        "hash_strategy": hash_strategy,
        "count": 0,
        "insert_times": [],  # nanosecond timings for each insert
        "lookup_times": [],  # nanosecond timings for each lookup
    }


def bloom_add(bf: dict, item: str) -> None:
    """Insert an item into the Bloom filter.

    Args:
        bf: Bloom filter state dict created by `create_bloom_filter`.
        item: String element to insert.

    Raises:
        TypeError: If item is not a string.
        ValueError: If item is empty.
    """
    if not isinstance(item, str):
        raise TypeError(f"item must be str, got {type(item).__name__}")
    if not item:
        raise ValueError("item must be a non-empty string")

    t0 = time.perf_counter_ns()
    positions = get_hash_positions(item, bf["m"], bf["k"], bf["hash_fn"])
    for pos in positions:
        set_bit(bf["bit_array"], pos)
    elapsed = time.perf_counter_ns() - t0

    bf["count"] += 1
    bf["insert_times"].append(elapsed)


def bloom_contains(bf: dict, item: str) -> bool:
    """Test whether an item is possibly in the Bloom filter.

    A `True` result means the item *might* be present (could be a false
    positive).  A `False` result is a definitive *not present*.

    Args:
        bf: Bloom filter state dict.
        item: String element to look up.

    Returns:
        True if all k bit positions are set (possible member).
        False if at least one position is unset (definite non-member).

    Raises:
        TypeError: If item is not a string.
        ValueError: If item is empty.
    """
    if not isinstance(item, str):
        raise TypeError(f"item must be str, got {type(item).__name__}")
    if not item:
        raise ValueError("item must be a non-empty string")

    t0 = time.perf_counter_ns()
    positions = get_hash_positions(item, bf["m"], bf["k"], bf["hash_fn"])
    result = all(get_bit(bf["bit_array"], pos) for pos in positions)
    elapsed = time.perf_counter_ns() - t0

    bf["lookup_times"].append(elapsed)
    return result


def bloom_load_factor(bf: dict) -> float:
    """Fraction of bits currently set in the filter.

    A load factor near 0.5 is theoretically optimal for false positive rate.

    Args:
        bf: Bloom filter state dict.

    Returns:
        A float in [0.0, 1.0].
    """
    set_count = count_set_bits(bf["bit_array"])
    return set_count / bf["m"]


def bloom_estimated_fpr(bf: dict) -> float:
    """Estimate the current empirical false positive rate.

    Based on the approximation: fpr ≈ (1 - e^(-k*n/m))^k

    Args:
        bf: Bloom filter state dict.

    Returns:
        Estimated false positive rate as a float.
    """
    k = bf["k"]
    n = bf["count"]
    m = bf["m"]
    # Guard against log-of-zero when the filter is empty
    if n == 0:
        return 0.0
    return (1 - math.exp(-k * n / m)) ** k


# ===========================================================================
# SECTION 4 – DATASET LOADING
# ===========================================================================


def stream_rockyou(
    path: str,
    max_lines: Optional[int] = None,
    min_length: int = 1,
    max_length: int = 128,
) -> Generator[str, None, None]:
    """Lazily stream valid password entries from rockyou.txt.

    The file uses latin-1 encoding (original breach dump).  Lines that
    cannot be decoded or fall outside the length bounds are silently skipped.

    Args:
        path: Filesystem path to rockyou.txt (plain or .gz not supported).
        max_lines: Cap on the number of lines yielded; None = unlimited.
        min_length: Minimum password length to include (inclusive).
        max_length: Maximum password length to include (inclusive).

    Yields:
        One cleaned password string at a time.

    Raises:
        FileNotFoundError: If `path` does not exist.
        PermissionError: If the file cannot be read.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {path}")

    yielded = 0
    with p.open("rb") as fh:
        for raw_line in fh:
            try:
                # rockyou.txt uses latin-1; strip CRLF/LF
                line = raw_line.rstrip(b"\r\n").decode("latin-1")
            except (UnicodeDecodeError, ValueError):
                continue  # skip undecodable lines
            if min_length <= len(line) <= max_length:
                yield line
                yielded += 1
                if max_lines and yielded >= max_lines:
                    return


def load_rockyou_sample(
    path: str,
    n_train: int = 500_000,
    n_test_pos: int = 10_000,
    n_test_neg: int = 10_000,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Load train + test splits from rockyou.txt.

    Positive test samples are drawn from the training set (must be found).
    Negative test samples are synthetically generated (should not be found).

    Args:
        path: Path to rockyou.txt.
        n_train: Number of passwords to insert into the filter.
        n_test_pos: Number of known-positive test queries.
        n_test_neg: Number of known-negative test queries.
        seed: Random seed for reproducibility.

    Returns:
        A tuple (train_set, test_positive, test_negative).

    Raises:
        FileNotFoundError: If the dataset is missing.
        ValueError: If the file has fewer lines than n_train.
    """
    rng = random.Random(seed)
    logger.info("Loading rockyou.txt — reading %d passwords …", n_train)

    # Load more than needed so we can sample test positives without overlap
    load_n = n_train + n_test_pos
    passwords = list(stream_rockyou(path, max_lines=load_n))

    if len(passwords) < n_train:
        raise ValueError(
            f"Dataset has only {len(passwords)} valid entries, "
            f"but n_train={n_train} was requested."
        )

    rng.shuffle(passwords)
    train_set = passwords[:n_train]
    # Positives: passwords that ARE in the training set
    test_positive = rng.sample(train_set, min(n_test_pos, len(train_set)))

    # Negatives: randomly generated strings very unlikely to collide with any
    # real password in rockyou.  Using UUID-style hex strings.
    test_negative = [
        "SYNTHETIC_" + hashlib.md5(str(rng.random()).encode()).hexdigest()
        for _ in range(n_test_neg)
    ]

    logger.info(
        "Split — train=%d | test_pos=%d | test_neg=%d",
        len(train_set),
        len(test_positive),
        len(test_negative),
    )
    return train_set, test_positive, test_negative


# ===========================================================================
# SECTION 5 – METRICS COMPUTATION
# ===========================================================================


def compute_metrics(
    bf: dict,
    test_positive: list[str],
    test_negative: list[str],
) -> dict:
    """Evaluate Bloom filter accuracy and performance metrics.

    Args:
        bf: A populated Bloom filter state dict.
        test_positive: Items known to be in the filter (true members).
        test_negative: Items known NOT to be in the filter (non-members).

    Returns:
        A dict with keys:
            - tp, fn, tn, fp: confusion matrix counts
            - tpr: true positive rate (recall)
            - fpr_empirical: measured false positive rate
            - fpr_theoretical: formula-based estimate
            - fpr_target: the fpr the filter was designed for
            - precision, f1: classification metrics
            - memory_bytes, memory_mb: storage footprint
            - bits_per_element: memory efficiency
            - avg_insert_ns, avg_lookup_ns: latency statistics
            - throughput_insert, throughput_lookup: ops per second
            - load_factor: fraction of bits set
            - bit_array_size: total bits (m)
            - num_hash_functions: k value
            - inserted_count: elements in filter
    """
    # ---- Confusion matrix -------------------------------------------------
    tp = sum(1 for pw in test_positive if bloom_contains(bf, pw))
    fn = len(test_positive) - tp  # should be 0 for a correct filter
    fp = sum(1 for pw in test_negative if bloom_contains(bf, pw))
    tn = len(test_negative) - fp

    # ---- Rates ------------------------------------------------------------
    tpr = tp / len(test_positive) if test_positive else 0.0
    fpr_emp = fp / len(test_negative) if test_negative else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # ---- Memory -----------------------------------------------------------
    memory_bytes = len(bf["bit_array"])
    bits_per_element = bf["m"] / max(bf["count"], 1)

    # ---- Latency ----------------------------------------------------------
    ins_times = bf["insert_times"]
    lkp_times = bf["lookup_times"]
    avg_ins = float(np.mean(ins_times)) if ins_times else 0.0
    avg_lkp = float(np.mean(lkp_times)) if lkp_times else 0.0

    # Throughput in operations per second
    tp_ins = 1e9 / avg_ins if avg_ins > 0 else 0.0
    tp_lkp = 1e9 / avg_lkp if avg_lkp > 0 else 0.0

    return {
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "tpr": tpr,
        "fpr_empirical": fpr_emp,
        "fpr_theoretical": bloom_estimated_fpr(bf),
        "fpr_target": bf["fpr_target"],
        "precision": precision,
        "f1": f1,
        "memory_bytes": memory_bytes,
        "memory_mb": memory_bytes / 1e6,
        "bits_per_element": bits_per_element,
        "avg_insert_ns": avg_ins,
        "avg_lookup_ns": avg_lkp,
        "throughput_insert": tp_ins,
        "throughput_lookup": tp_lkp,
        "load_factor": bloom_load_factor(bf),
        "bit_array_size": bf["m"],
        "num_hash_functions": bf["k"],
        "inserted_count": bf["count"],
    }


# ===========================================================================
# SECTION 6 – BENCHMARKING ACROSS CONFIGURATIONS
# ===========================================================================


def benchmark_fpr_vs_n(
    all_passwords: list[str],
    fpr_target: float = 0.01,
    sizes: Optional[list[int]] = None,
) -> list[dict]:
    """Measure empirical FPR as n (inserted elements) increases.

    Args:
        all_passwords: Full list of passwords available for testing.
        fpr_target: False positive rate target for filter construction.
        sizes: List of n values to test; defaults to 10 log-spaced points.

    Returns:
        List of dicts with keys: n, fpr_empirical, fpr_theoretical,
        load_factor, memory_mb.
    """
    if sizes is None:
        # 10 evenly log-spaced sample points between 1k and min(500k, len)
        max_n = min(500_000, len(all_passwords))
        sizes = [int(x) for x in np.logspace(3, math.log10(max_n), 10)]

    results = []
    n_neg = 5000  # synthetic negatives per measurement point
    rng = random.Random(7)

    for n in sizes:
        sample = all_passwords[:n]
        bf = create_bloom_filter(n, fpr=fpr_target, hash_strategy="murmur3")
        for pw in sample:
            bloom_add(bf, pw)

        # Generate synthetic negatives (not in sample)
        neg = [
            "BM_" + hashlib.md5(str(rng.random()).encode()).hexdigest()
            for _ in range(n_neg)
        ]
        fp_count = sum(1 for x in neg if bloom_contains(bf, x))
        results.append(
            {
                "n": n,
                "fpr_empirical": fp_count / n_neg,
                "fpr_theoretical": bloom_estimated_fpr(bf),
                "load_factor": bloom_load_factor(bf),
                "memory_mb": len(bf["bit_array"]) / 1e6,
            }
        )
        logger.info(
            "  n=%7d | fpr_emp=%.4f | load=%.3f",
            n,
            results[-1]["fpr_empirical"],
            results[-1]["load_factor"],
        )
    return results


def benchmark_hash_strategies(
    train: list[str],
    test_neg: list[str],
    fpr_target: float = 0.01,
) -> list[dict]:
    """Compare all hash strategies on the same dataset.

    Args:
        train: Passwords to insert.
        test_neg: Known-negative queries for FPR measurement.
        fpr_target: Desired false positive rate for filter sizing.

    Returns:
        List of dicts with keys: strategy, fpr_empirical, avg_insert_ns,
        avg_lookup_ns, throughput_insert, throughput_lookup, load_factor.
    """
    results = []
    for strategy in HASH_STRATEGIES:
        bf = create_bloom_filter(len(train), fpr=fpr_target, hash_strategy=strategy)
        for pw in train:
            bloom_add(bf, pw)

        fp = sum(1 for x in test_neg if bloom_contains(bf, x))
        fpr_emp = fp / len(test_neg) if test_neg else 0.0
        avg_ins = float(np.mean(bf["insert_times"])) if bf["insert_times"] else 0
        avg_lkp = float(np.mean(bf["lookup_times"])) if bf["lookup_times"] else 0

        results.append(
            {
                "strategy": strategy,
                "fpr_empirical": fpr_emp,
                "avg_insert_ns": avg_ins,
                "avg_lookup_ns": avg_lkp,
                "throughput_insert": 1e9 / avg_ins if avg_ins > 0 else 0,
                "throughput_lookup": 1e9 / avg_lkp if avg_lkp > 0 else 0,
                "load_factor": bloom_load_factor(bf),
            }
        )
        logger.info(
            "  strategy=%-8s | fpr=%.4f | ins=%.0f ns", strategy, fpr_emp, avg_ins
        )
    return results


def benchmark_fpr_targets(
    train: list[str],
    test_neg: list[str],
    targets: Optional[list[float]] = None,
) -> list[dict]:
    """Sweep over multiple FPR targets and record size / accuracy trade-offs.

    Args:
        train: Training passwords.
        test_neg: Known-negative queries.
        targets: List of fpr values to evaluate; defaults to [0.001 … 0.1].

    Returns:
        List of dicts with keys: fpr_target, fpr_empirical, memory_mb,
        bits_per_element, k.
    """
    if targets is None:
        targets = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]

    results = []
    for fpr_t in targets:
        bf = create_bloom_filter(len(train), fpr=fpr_t, hash_strategy="murmur3")
        for pw in train:
            bloom_add(bf, pw)
        fp = sum(1 for x in test_neg if bloom_contains(bf, x))
        results.append(
            {
                "fpr_target": fpr_t,
                "fpr_empirical": fp / len(test_neg) if test_neg else 0,
                "memory_mb": len(bf["bit_array"]) / 1e6,
                "bits_per_element": bf["m"] / bf["count"],
                "k": bf["k"],
            }
        )
    return results


# ===========================================================================
# SECTION 7 – TERMINAL REPORT
# ===========================================================================


def _bar(value: float, max_value: float = 1.0, width: int = 30) -> str:
    """Render a simple ASCII progress bar.

    Args:
        value: Current value.
        max_value: Value that corresponds to a full bar.
        width: Character width of the bar.

    Returns:
        A string like '████████░░░░░░ 53.3%'.
    """
    pct = min(value / max_value, 1.0) if max_value > 0 else 0
    filled = int(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {pct * 100:.1f}%"


def print_banner() -> None:
    """Print the ASCII-art project banner to stdout."""
    banner = r"""
╔══════════════════════════════════════════════════════════════╗
║   ██████╗ ██╗      ██████╗  ██████╗ ███╗   ███╗            ║
║   ██╔══██╗██║     ██╔═══██╗██╔═══██╗████╗ ████║            ║
║   ██████╔╝██║     ██║   ██║██║   ██║██╔████╔██║            ║
║   ██╔══██╗██║     ██║   ██║██║   ██║██║╚██╔╝██║            ║
║   ██████╔╝███████╗╚██████╔╝╚██████╔╝██║ ╚═╝ ██║            ║
║   ╚═════╝ ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝     ╚═╝            ║
║              F I L T E R   A N A L Y S I S                  ║
║                  rockyou.txt  ·  Python 3.9+                 ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_metrics_report(metrics: dict, bf: dict) -> None:
    """Print a detailed human-readable metrics report to stdout.

    Args:
        metrics: Dict returned by `compute_metrics`.
        bf: Bloom filter state dict (for config display).
    """
    W = 65  # report width
    sep = "─" * W

    print(f"\n{'═' * W}")
    print(f"  BLOOM FILTER  ·  METRICS REPORT".center(W))
    print(f"{'═' * W}")

    # Configuration
    print(f"\n  {'CONFIGURATION':}")
    print(f"  {sep}")
    print(f"  Hash strategy : {bf['hash_strategy']}")
    print(f"  Capacity (n)  : {bf['capacity']:,}")
    print(f"  Bit array (m) : {metrics['bit_array_size']:,} bits")
    print(f"  Hash funcs (k): {metrics['num_hash_functions']}")
    print(f"  Inserted (n)  : {metrics['inserted_count']:,}")
    print(f"  Memory        : {metrics['memory_mb']:.2f} MB")
    print(f"  Bits/element  : {metrics['bits_per_element']:.2f}")

    # Accuracy
    print(f"\n  {'ACCURACY':}")
    print(f"  {sep}")
    print(f"  True Positives  (TP): {metrics['tp']:,}")
    print(f"  False Negatives (FN): {metrics['fn']:,}  ← must be 0")
    print(f"  True Negatives  (TN): {metrics['tn']:,}")
    print(f"  False Positives (FP): {metrics['fp']:,}")
    print()
    print(f"  True Positive Rate  : {metrics['tpr']:.6f}  {_bar(metrics['tpr'])}")
    print(
        f"  FPR (empirical)     : {metrics['fpr_empirical']:.6f}  {_bar(metrics['fpr_empirical'], 0.1)}"
    )
    print(f"  FPR (theoretical)   : {metrics['fpr_theoretical']:.6f}")
    print(f"  FPR (target)        : {metrics['fpr_target']:.6f}")
    print(f"  Precision           : {metrics['precision']:.6f}")
    print(f"  F1 Score            : {metrics['f1']:.6f}")

    # Load
    print(f"\n  {'LOAD & SATURATION':}")
    print(f"  {sep}")
    print(
        f"  Load factor: {metrics['load_factor']:.4f}  {_bar(metrics['load_factor'])}"
    )
    optimal_note = " ← near optimal!" if 0.4 < metrics["load_factor"] < 0.6 else ""
    print(f"             (optimal ≈ 0.5){optimal_note}")

    # Performance
    print(f"\n  {'PERFORMANCE':}")
    print(f"  {sep}")
    print(f"  Avg insert latency : {metrics['avg_insert_ns']:,.0f} ns")
    print(f"  Avg lookup latency : {metrics['avg_lookup_ns']:,.0f} ns")
    print(f"  Insert throughput  : {metrics['throughput_insert']:>12,.0f} ops/s")
    print(f"  Lookup throughput  : {metrics['throughput_lookup']:>12,.0f} ops/s")

    print(f"\n{'═' * W}\n")


# ===========================================================================
# SECTION 8 – MATPLOTLIB VISUALISATIONS
# ===========================================================================

# Colour palette — consistent across all charts
_PALETTE = {
    "primary": "#4F8EF7",  # bright blue
    "accent": "#F7754F",  # coral
    "green": "#4FD4A0",  # teal-green
    "yellow": "#F7C94F",  # warm yellow
    "purple": "#A07AF7",  # violet
    "bg": "#0D1117",  # near-black background
    "surface": "#161B22",  # card surface
    "border": "#30363D",  # subtle border
    "text": "#E6EDF3",  # primary text
    "muted": "#8B949E",  # secondary text
}


def _apply_dark_theme(fig: plt.Figure, axes: list) -> None:
    """Apply consistent dark theme to figure and axes.

    Args:
        fig: The matplotlib Figure object.
        axes: List of Axes objects to style.
    """
    fig.patch.set_facecolor(_PALETTE["bg"])
    for ax in axes:
        ax.set_facecolor(_PALETTE["surface"])
        ax.tick_params(colors=_PALETTE["muted"], labelsize=9)
        ax.xaxis.label.set_color(_PALETTE["text"])
        ax.yaxis.label.set_color(_PALETTE["text"])
        ax.title.set_color(_PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(_PALETTE["border"])
        ax.grid(color=_PALETTE["border"], linestyle="--", linewidth=0.5, alpha=0.6)


def plot_dashboard(
    metrics: dict,
    bf: dict,
    fpr_vs_n: list[dict],
    hash_bench: list[dict],
    fpr_sweep: list[dict],
    output_path: str = "bloom_dashboard.png",
) -> None:
    """Render a 6-panel analytics dashboard to a PNG file.

    Panels:
        1. Confusion matrix heatmap
        2. FPR empirical vs theoretical vs n
        3. Load factor vs n
        4. Hash strategy throughput comparison
        5. Memory (MB) vs FPR target
        6. Latency distribution (insert vs lookup)

    Args:
        metrics: Output of `compute_metrics`.
        bf: Bloom filter state dict.
        fpr_vs_n: Output of `benchmark_fpr_vs_n`.
        hash_bench: Output of `benchmark_hash_strategies`.
        fpr_sweep: Output of `benchmark_fpr_targets`.
        output_path: File path for the PNG output.
    """
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "Bloom Filter Analytics Dashboard  ·  rockyou.txt",
        fontsize=17,
        fontweight="bold",
        color=_PALETTE["text"],
        y=0.98,
    )
    gs = GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])  # Confusion matrix
    ax2 = fig.add_subplot(gs[0, 1])  # FPR vs n
    ax3 = fig.add_subplot(gs[0, 2])  # Load factor vs n
    ax4 = fig.add_subplot(gs[1, 0])  # Hash strategy throughput
    ax5 = fig.add_subplot(gs[1, 1])  # Memory vs FPR target
    ax6 = fig.add_subplot(gs[1, 2])  # FPR sweep (k values)
    ax7 = fig.add_subplot(gs[2, :])  # Latency distribution (full width)

    all_axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    _apply_dark_theme(fig, all_axes)

    # ------------------------------------------------------------------
    # Panel 1 – Confusion matrix
    # ------------------------------------------------------------------
    cm = np.array(
        [
            [metrics["tp"], metrics["fn"]],
            [metrics["fp"], metrics["tn"]],
        ],
        dtype=float,
    )
    # Normalise rows for readability
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    im = ax1.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(
        ["Predicted +", "Predicted −"], color=_PALETTE["text"], fontsize=9
    )
    ax1.set_yticklabels(["Actual +", "Actual −"], color=_PALETTE["text"], fontsize=9)
    ax1.set_title("Confusion Matrix (row-normalised)", fontsize=11, pad=8)
    labels = [["TP", "FN"], ["FP", "TN"]]
    counts = [[metrics["tp"], metrics["fn"]], [metrics["fp"], metrics["tn"]]]
    for i in range(2):
        for j in range(2):
            ax1.text(
                j,
                i,
                f"{labels[i][j]}\n{counts[i][j]:,}\n({cm_norm[i, j]:.3f})",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > 0.5 else _PALETTE["text"],
                fontsize=10,
                fontweight="bold",
            )

    # ------------------------------------------------------------------
    # Panel 2 – FPR vs n (empirical + theoretical)
    # ------------------------------------------------------------------
    ns = [r["n"] for r in fpr_vs_n]
    fpr_emp_vals = [r["fpr_empirical"] for r in fpr_vs_n]
    fpr_the_vals = [r["fpr_theoretical"] for r in fpr_vs_n]
    ax2.plot(
        ns,
        fpr_emp_vals,
        "o-",
        color=_PALETTE["accent"],
        lw=2,
        markersize=5,
        label="Empirical FPR",
    )
    ax2.plot(
        ns,
        fpr_the_vals,
        "s--",
        color=_PALETTE["primary"],
        lw=1.5,
        markersize=4,
        label="Theoretical FPR",
    )
    ax2.axhline(
        bf["fpr_target"],
        color=_PALETTE["yellow"],
        ls=":",
        lw=1.5,
        label=f"Target FPR = {bf['fpr_target']}",
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("Elements inserted (n)")
    ax2.set_ylabel("False Positive Rate")
    ax2.set_title("FPR vs Inserted Elements", fontsize=11)
    ax2.legend(
        fontsize=8,
        facecolor=_PALETTE["surface"],
        labelcolor=_PALETTE["text"],
        edgecolor=_PALETTE["border"],
    )

    # ------------------------------------------------------------------
    # Panel 3 – Load factor vs n
    # ------------------------------------------------------------------
    load_vals = [r["load_factor"] for r in fpr_vs_n]
    ax3.plot(ns, load_vals, "o-", color=_PALETTE["green"], lw=2, markersize=5)
    ax3.axhline(0.5, color=_PALETTE["yellow"], ls="--", lw=1.5, label="Optimal (0.5)")
    ax3.set_xscale("log")
    ax3.set_xlabel("Elements inserted (n)")
    ax3.set_ylabel("Load Factor (bits set / m)")
    ax3.set_title("Bit Array Load Factor vs n", fontsize=11)
    ax3.set_ylim(0, 1)
    ax3.legend(
        fontsize=8,
        facecolor=_PALETTE["surface"],
        labelcolor=_PALETTE["text"],
        edgecolor=_PALETTE["border"],
    )

    # ------------------------------------------------------------------
    # Panel 4 – Hash strategy throughput (grouped bar)
    # ------------------------------------------------------------------
    strategies = [r["strategy"] for r in hash_bench]
    ins_tps = [r["throughput_insert"] / 1e6 for r in hash_bench]  # ops/s → M ops/s
    lkp_tps = [r["throughput_lookup"] / 1e6 for r in hash_bench]
    x = np.arange(len(strategies))
    w = 0.35
    ax4.bar(x - w / 2, ins_tps, w, label="Insert", color=_PALETTE["primary"], alpha=0.9)
    ax4.bar(x + w / 2, lkp_tps, w, label="Lookup", color=_PALETTE["accent"], alpha=0.9)
    ax4.set_xticks(x)
    ax4.set_xticklabels(strategies, color=_PALETTE["text"])
    ax4.set_ylabel("Throughput (M ops/s)")
    ax4.set_title("Hash Strategy Throughput", fontsize=11)
    ax4.legend(
        fontsize=8,
        facecolor=_PALETTE["surface"],
        labelcolor=_PALETTE["text"],
        edgecolor=_PALETTE["border"],
    )

    # ------------------------------------------------------------------
    # Panel 5 – Memory vs FPR target
    # ------------------------------------------------------------------
    fpr_targets = [r["fpr_target"] for r in fpr_sweep]
    memories = [r["memory_mb"] for r in fpr_sweep]
    k_vals = [r["k"] for r in fpr_sweep]
    color_map = plt.cm.plasma(np.linspace(0.3, 0.9, len(fpr_targets)))
    bars = ax5.bar(
        [f"{t * 100:.1f}%" for t in fpr_targets],
        memories,
        color=color_map,
        alpha=0.9,
        edgecolor=_PALETTE["border"],
    )
    # Annotate each bar with k value
    for bar, k in zip(bars, k_vals):
        ax5.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"k={k}",
            ha="center",
            va="bottom",
            color=_PALETTE["muted"],
            fontsize=8,
        )
    ax5.set_xlabel("FPR Target")
    ax5.set_ylabel("Memory (MB)")
    ax5.set_title("Memory vs FPR Target", fontsize=11)
    ax5.tick_params(axis="x", labelrotation=30)

    # ------------------------------------------------------------------
    # Panel 6 – FPR empirical vs target (scatter with ideal line)
    # ------------------------------------------------------------------
    fpr_emp_sweep = [r["fpr_empirical"] for r in fpr_sweep]
    ax6.scatter(
        fpr_targets,
        fpr_emp_sweep,
        color=_PALETTE["accent"],
        s=70,
        zorder=5,
        label="Empirical",
    )
    ideal = np.linspace(min(fpr_targets), max(fpr_targets), 100)
    ax6.plot(
        ideal, ideal, "--", color=_PALETTE["muted"], lw=1, label="Ideal (emp = target)"
    )
    ax6.set_xlabel("FPR Target")
    ax6.set_ylabel("FPR Empirical")
    ax6.set_title("Empirical vs Target FPR", fontsize=11)
    ax6.legend(
        fontsize=8,
        facecolor=_PALETTE["surface"],
        labelcolor=_PALETTE["text"],
        edgecolor=_PALETTE["border"],
    )

    # ------------------------------------------------------------------
    # Panel 7 – Latency distribution (insert + lookup histograms)
    # ------------------------------------------------------------------
    ins_ns = np.array(bf["insert_times"])
    lkp_ns = np.array(bf["lookup_times"])

    # Clip extreme outliers for readability (99th percentile)
    p99_ins = np.percentile(ins_ns, 99)
    p99_lkp = np.percentile(lkp_ns, 99)
    ins_clipped = ins_ns[ins_ns <= p99_ins]
    lkp_clipped = lkp_ns[lkp_ns <= p99_lkp]

    bins = 80
    ax7.hist(
        ins_clipped,
        bins=bins,
        alpha=0.65,
        color=_PALETTE["primary"],
        label=f"Insert (avg={np.mean(ins_ns):.0f} ns)",
        density=True,
    )
    ax7.hist(
        lkp_clipped,
        bins=bins,
        alpha=0.65,
        color=_PALETTE["accent"],
        label=f"Lookup (avg={np.mean(lkp_ns):.0f} ns)",
        density=True,
    )
    ax7.axvline(np.mean(ins_ns), color=_PALETTE["primary"], ls="--", lw=1.5)
    ax7.axvline(np.mean(lkp_ns), color=_PALETTE["accent"], ls="--", lw=1.5)
    ax7.set_xlabel("Latency (ns)")
    ax7.set_ylabel("Density")
    ax7.set_title("Operation Latency Distribution (clipped at 99th pct)", fontsize=11)
    ax7.legend(
        fontsize=9,
        facecolor=_PALETTE["surface"],
        labelcolor=_PALETTE["text"],
        edgecolor=_PALETTE["border"],
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    fig.subplots_adjust(top=0.94)
    plt.savefig(output_path, dpi=150, facecolor=_PALETTE["bg"])
    logger.info("Dashboard saved → %s", output_path)
    plt.close(fig)


def plot_bit_density_heatmap(
    bf: dict,
    output_path: str = "bloom_bit_density.png",
    grid_size: int = 64,
) -> None:
    """Visualise the spatial bit density of the filter as a heatmap.

    Divides the bit array into a grid_size × grid_size grid and colours
    each cell by the fraction of bits set within that region.

    Args:
        bf: Bloom filter state dict.
        grid_size: Number of rows and columns in the grid (default 64).
        output_path: File path for PNG output.
    """
    m = bf["m"]
    bits_per_cell = m // (grid_size * grid_size)
    if bits_per_cell < 1:
        logger.warning("Filter too small for %dx%d density grid", grid_size, grid_size)
        return

    density = np.zeros((grid_size, grid_size), dtype=float)
    bit_arr = bf["bit_array"]

    # Flatten the bit array into per-cell densities
    for row in range(grid_size):
        for col in range(grid_size):
            cell_idx = (row * grid_size + col) * bits_per_cell
            cell_end = cell_idx + bits_per_cell
            set_count = 0
            for bit_pos in range(cell_idx, min(cell_end, m)):
                if get_bit(bit_arr, bit_pos):
                    set_count += 1
            density[row, col] = set_count / bits_per_cell

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor(_PALETTE["bg"])
    ax.set_facecolor(_PALETTE["bg"])

    im = ax.imshow(density, cmap="plasma", vmin=0, vmax=1, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Bit Density (fraction set)", color=_PALETTE["text"])
    cbar.ax.yaxis.set_tick_params(color=_PALETTE["muted"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_PALETTE["muted"])

    ax.set_title(
        f"Bloom Filter Bit Density  ·  load={bloom_load_factor(bf):.3f}",
        color=_PALETTE["text"],
        fontsize=13,
        pad=12,
    )
    ax.set_xlabel("Bit block column", color=_PALETTE["text"])
    ax.set_ylabel("Bit block row", color=_PALETTE["text"])
    ax.tick_params(colors=_PALETTE["muted"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_PALETTE["bg"])
    logger.info("Bit density heatmap saved → %s", output_path)
    plt.close(fig)


# ===========================================================================
# SECTION 9 – INFERENCE ENGINE
# ===========================================================================


def run_inference(bf: dict, train_set_size: int) -> None:
    """Interactive CLI inference loop for real-time password lookups.

    Allows the user to type passwords and receive instant membership
    verdicts from the Bloom filter, with detailed statistics per query.

    Args:
        bf: A populated Bloom filter state dict.
        train_set_size: Number of items originally inserted (for display).
    """
    fpr = bloom_estimated_fpr(bf)
    print("\n" + "═" * 65)
    print("  BLOOM FILTER  ·  INTERACTIVE INFERENCE ENGINE")
    print("═" * 65)
    print(f"  Filter contains approx. {train_set_size:,} passwords")
    print(f"  Estimated FPR: {fpr:.4%}")
    print(f"  Type a password to query (or 'quit' / 'exit' to stop)")
    print("═" * 65 + "\n")

    while True:
        try:
            query = input("  🔍 Query: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Exiting inference engine.")
            break

        if query.lower() in {"quit", "exit", "q"}:
            print("  Goodbye!")
            break

        if not query:
            print("  ⚠  Empty input, please enter a password.\n")
            continue

        # Time the individual lookup
        t0 = time.perf_counter_ns()
        found = bloom_contains(bf, query)
        elapsed_ns = time.perf_counter_ns() - t0

        positions = get_hash_positions(query, bf["m"], bf["k"], bf["hash_fn"])
        bits_set = sum(1 for p in positions if get_bit(bf["bit_array"], p))

        print(f"\n  ┌─ Result for: '{query}' {'─' * max(0, 40 - len(query))}┐")
        if found:
            print(f"  │  ✅ POSSIBLY IN FILTER  (all {bf['k']} hash positions set)")
            print(f"  │  ⚠  Note: could be a false positive (FPR ≈ {fpr:.4%})")
        else:
            print(
                f"  │  ❌ DEFINITELY NOT IN FILTER  ({bits_set}/{bf['k']} positions set)"
            )
        print(f"  │  Hash positions : {positions}")
        print(f"  │  Lookup latency : {elapsed_ns} ns")
        print(f"  └{'─' * 53}┘\n")


# ===========================================================================
# SECTION 10 – MAIN ORCHESTRATOR
# ===========================================================================


def run_full_analysis(
    dataset_path: str,
    n_train: int = 500_000,
    n_test_pos: int = 10_000,
    n_test_neg: int = 10_000,
    fpr_target: float = 0.01,
    hash_strategy: str = "murmur3",
    output_dir: str = ".",
    interactive: bool = True,
) -> None:
    """Run the complete Bloom Filter analysis pipeline.

    Steps:
        1. Load and split rockyou.txt dataset
        2. Build and populate the Bloom filter
        3. Compute accuracy + performance metrics
        4. Run benchmark suites (FPR vs n, hash strategies, FPR sweep)
        5. Render dashboard and bit-density heatmap
        6. Print terminal report
        7. Launch interactive inference (if interactive=True)

    Args:
        dataset_path: Path to rockyou.txt.
        n_train: Passwords to insert.
        n_test_pos: Known-positive test samples.
        n_test_neg: Known-negative test samples.
        fpr_target: Desired false positive rate.
        hash_strategy: Hash function family ('murmur3', 'fnv1a', 'sha256').
        output_dir: Directory for output PNG files.
        interactive: Launch the inference CLI after analysis.

    Raises:
        FileNotFoundError: If dataset_path does not exist.
        ValueError: For invalid parameter combinations.
    """
    print_banner()

    # Validate output directory
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load dataset ───────────────────────────────────────────────
    train_set, test_positive, test_negative = load_rockyou_sample(
        dataset_path, n_train, n_test_pos, n_test_neg
    )

    # ── Step 2: Build Bloom filter ─────────────────────────────────────────
    logger.info(
        "Building Bloom filter (strategy=%s, fpr=%.3f) …", hash_strategy, fpr_target
    )
    bf = create_bloom_filter(n_train, fpr=fpr_target, hash_strategy=hash_strategy)

    t_insert_start = time.perf_counter()
    for pw in train_set:
        bloom_add(bf, pw)
    t_insert_total = time.perf_counter() - t_insert_start
    logger.info(
        "Inserted %d passwords in %.2f s (%.0f/s)",
        len(train_set),
        t_insert_total,
        len(train_set) / t_insert_total,
    )

    # ── Step 3: Metrics ────────────────────────────────────────────────────
    logger.info("Computing metrics …")
    metrics = compute_metrics(bf, test_positive, test_negative)

    # ── Step 4: Benchmark suites ───────────────────────────────────────────
    logger.info("Benchmark — FPR vs n …")
    fpr_vs_n = benchmark_fpr_vs_n(train_set, fpr_target=fpr_target)

    logger.info("Benchmark — hash strategies …")
    # Use a smaller subset for strategy comparison to keep runtime reasonable
    bench_subset = train_set[:50_000]
    hash_bench = benchmark_hash_strategies(
        bench_subset, test_negative[:5000], fpr_target
    )

    logger.info("Benchmark — FPR target sweep …")
    fpr_sweep = benchmark_fpr_targets(bench_subset, test_negative[:5000])

    # ── Step 5: Visualisations ─────────────────────────────────────────────
    dashboard_path = str(out / "bloom_dashboard.png")
    density_path = str(out / "bloom_bit_density.png")
    logger.info("Rendering dashboard …")
    plot_dashboard(metrics, bf, fpr_vs_n, hash_bench, fpr_sweep, dashboard_path)
    logger.info("Rendering bit density heatmap …")
    plot_bit_density_heatmap(bf, density_path)

    # ── Step 6: Terminal report ────────────────────────────────────────────
    print_metrics_report(metrics, bf)
    print(f"  📊 Dashboard saved to  : {dashboard_path}")
    print(f"  🗺  Bit density map    : {density_path}\n")

    # ── Step 7: Interactive inference ──────────────────────────────────────
    if interactive:
        run_inference(bf, len(train_set))


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Bloom Filter Analysis on rockyou.txt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="rockyou.txt",
        help="Path to rockyou.txt password dataset",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=500_000,
        help="Number of passwords to insert into the filter",
    )
    parser.add_argument(
        "--n-test-pos",
        type=int,
        default=10_000,
        help="Number of positive test samples (known members)",
    )
    parser.add_argument(
        "--n-test-neg",
        type=int,
        default=10_000,
        help="Number of negative test samples (non-members)",
    )
    parser.add_argument(
        "--fpr",
        type=float,
        default=0.01,
        help="Target false positive rate (e.g. 0.01 = 1%%)",
    )
    parser.add_argument(
        "--hash",
        choices=list(HASH_STRATEGIES),
        default="murmur3",
        help="Hash function strategy",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save PNG visualisations",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip the interactive inference CLI",
    )

    args = parser.parse_args()

    # Validate FPR range before entering the pipeline
    if not (0 < args.fpr < 1):
        parser.error(f"--fpr must be between 0 and 1 (exclusive), got {args.fpr}")

    try:
        run_full_analysis(
            dataset_path=args.dataset,
            n_train=args.n_train,
            n_test_pos=args.n_test_pos,
            n_test_neg=args.n_test_neg,
            fpr_target=args.fpr,
            hash_strategy=args.hash,
            output_dir=args.output_dir,
            interactive=not args.no_interactive,
        )
    except FileNotFoundError as e:
        logger.error("%s", e)
        logger.error(
            "Download rockyou.txt from https://github.com/brannondorsey/naive-hashcat"
            "/releases/download/data/rockyou.txt  and pass its path as the first argument."
        )
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
