import math
from typing import List, Dict, Any
import numpy as np
# from itertools import combinations

def expected_jaccard_random(population_size: int, subset_size: int) -> float:
    """
    E[J] for two uniformly random 'subset_size'-subsets (A, B) of a universe of size 'population_size'.
    https://math.stackexchange.com/questions/1769655/computing-the-expected-value-of-the-jaccard-similarity-of-two-random-sets
    """
    ps = population_size
    ss = subset_size

    assert ps > 0 and ss > 0, \
        f"population_size and subset_size must be > 0, got {ps}, {ss}"
    assert ss <= ps, \
        f"subset_size must be <= population_size, got {ss} > {ps}"

    EJ = 0.0
    for overlap_size in range(0, ss + 1):
        os = overlap_size
        # Hypergeometric PMF
        pmf = (math.comb(ss, os) * math.comb(ps - ss, ss - os)) / math.comb(ps, ss)
        union_size = 2*ss - os
        EJ += pmf * (os / union_size)
    return EJ

# def normalized_entropy(freqs: np.ndarray) -> float:
#     total = freqs.sum()
#     if total == 0:
#         return 0.0
#     p = freqs / total
#     p = p[p > 0]
#     H = -(p * np.log(p)).sum()
#     Hmax = math.log(len(freqs)) if len(freqs) > 0 else 1.0
#     return float(H / (Hmax + 1e-12))

# def gini_coefficient(freqs: np.ndarray) -> float:
#     total = freqs.sum()
#     if total == 0:
#         return 0.0
#     x = np.sort(freqs.astype(np.float64))
#     n = x.size
#     cumx = np.cumsum(x)
#     gini = (n + 1 - 2 * (cumx.sum() / cumx[-1])) / n
#     return float(gini)

def mean_pairwise_overlap_between_sets(unstable_sets: List[set], total_heads: int) -> float:
    assert len(unstable_sets) > 1, "Need at least 2 sets"

    # sum_intersections = 0
    # num_pairs = 0
    # for A, B in combinations(unstable_sets, 2):
    #     sum_intersections += len(A & B)  # size of intersection
    #     num_pairs += 1
    # return sum_intersections / num_pairs

    # Faster method using frequency counts. Equivalent to above.
    # If a head appears in f sets, it contributes fC2 to the total intersection count.

    head_freq = np.zeros(total_heads, dtype=np.int64)
    for unstable_set in unstable_sets:
        for h in unstable_set:
            head_freq[h] += 1

    sum_intersections = int((head_freq*(head_freq-1)//2).sum())

    num_sets = len(unstable_sets)
    num_pairs = (num_sets * (num_sets-1)) // 2
    return sum_intersections / num_pairs

def pairwise_jaccard_stats(unstable_sets: List[set], M: int, total_heads: int) -> Dict[str, Any]:
    num_sets  = len(unstable_sets)
    num_pairs = (num_sets * (num_sets-1)) // 2

    # WikiPedia: The hypergeometric distribution describes the probability of k successes
    # in n draws, without replacement, from a finite population of size N that contains exactly
    # K objects with that feature, wherein each draw is either a success or a failure.
    # In contrast, the binomial distribution describes the probability of k successes in n draws with replacement.

    # We need to estimate the expected overlap between two sets of M unstable heads if each set was formed
    # independently and randomly by drawing M heads without replacement from the set of total_heads.
    # This can be estimated using the hypergeometric distribution. Suppose we have generated set A and now
    # we are generating set B. So, for generating set B, we have a hypergeometric distribution with
    # population size N=total_heads, number of success states in the population K=M (the heads in set A),
    # number of draws n=M (we want to select M heads for set B).
    # Now, the expected overlap is the mean of this hypergeometric distribution.
    # Mean of Hypergeometric(N, K, n) = n * (K/N) = M * (M/total_heads)
    # Variance of Hypergeometric(N, K, n) = n * (K/N) * (1 - K/N) * ((N - n)/(N - 1))
    #                                     = M * (M/total_heads) * (1 - M/total_heads) * ((total_heads - M)/(total_heads - 1))

    mean_overlap = mean_pairwise_overlap_between_sets(unstable_sets, total_heads)

    expected_random_overlap = M * (M / total_heads)
    variance_random_overlap = M * (M / total_heads) * (1 - M / total_heads) * ((total_heads - M) / (total_heads - 1))
    stddev_random_overlap = math.sqrt(variance_random_overlap)

    z_score = (mean_overlap - expected_random_overlap) / stddev_random_overlap

    num_sets = len(unstable_sets)
    js_sum   = 0.0
    js_cnt   = 0

    for i in range(num_sets):
        A = unstable_sets[i]
        for j in range(i + 1, num_sets):
            B = unstable_sets[j]
            assert len(A) == M and len(B) == M, f"Set sizes must be {M}"
            inter = len(A & B)
            union = 2 * M - inter  # since |A|=|B|=M
            js_sum += inter / union
            js_cnt += 1

    avg_j = js_sum / js_cnt

    return {
        "avg_jaccard": float(avg_j),
        "exp_jaccard_random": expected_jaccard_random(total_heads, M),
        "avg_intersection": float(mean_overlap),
        "exp_intersection_random": expected_random_overlap,
        "avg_intersection_z": float(z_score),
        "num_sets": int(num_sets),
        "num_pairs_total": int(num_pairs),
    }

def head_frequency(unstable_sets: List[set], total_heads: int) -> np.ndarray:
    counts = np.zeros(total_heads, dtype=np.int64)
    for s in unstable_sets:
        for h in s:
            counts[h] += 1
    return counts