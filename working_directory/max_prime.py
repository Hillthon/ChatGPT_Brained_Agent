"""使用欧拉筛和分段筛计算不超过上限的最大素数。"""

import argparse
import math


def euler_sieve(limit: int) -> list[int]:
    """欧拉筛：返回不超过 limit 的全部素数，每个合数只被标记一次。"""
    is_composite = bytearray(limit + 1)
    primes: list[int] = []

    for number in range(2, limit + 1):
        if not is_composite[number]:
            primes.append(number)
        for prime in primes:
            multiple = number * prime
            if multiple > limit:
                break
            is_composite[multiple] = 1
            if number % prime == 0:
                break
    return primes


def largest_prime_at_most(limit: int, segment_size: int = 1_000_000) -> int:
    """返回不超过 limit 的最大素数。

    先用欧拉筛生成 sqrt(limit) 以内的素数，再从 limit 向下分段筛选。
    因此不需要为整个 [0, limit] 区间分配内存。
    """
    if limit < 2:
        raise ValueError("limit 必须至少为 2")
    if segment_size < 1:
        raise ValueError("segment_size 必须为正数")

    base_primes = euler_sieve(math.isqrt(limit))
    high = limit if limit % 2 else limit - 1

    while high >= 3:
        low = max(3, high - segment_size + 1)
        if low % 2 == 0:
            low += 1

        # 只保存奇数候选数：candidate = low + 2 * index。
        is_composite = bytearray((high - low) // 2 + 1)
        for prime in base_primes:
            if prime == 2:
                continue
            if prime * prime > high:
                break

            first = max(prime * prime, ((low + prime - 1) // prime) * prime)
            if first % 2 == 0:
                first += prime
            for index in range((first - low) // 2, len(is_composite), prime):
                is_composite[index] = 1

        for index in range(len(is_composite) - 1, -1, -1):
            if not is_composite[index]:
                return low + 2 * index
        high = low - 2

    return 2


def main() -> None:
    parser = argparse.ArgumentParser(description="查找不超过上限的最大素数")
    parser.add_argument("limit", nargs="?", type=int, default=10**12)
    parser.add_argument(
        "--segment-size",
        type=int,
        default=1_000_000,
        help="每次分段处理的候选范围大小，默认 1000000",
    )
    args = parser.parse_args()
    print(largest_prime_at_most(args.limit, args.segment_size))


if __name__ == "__main__":
    main()
