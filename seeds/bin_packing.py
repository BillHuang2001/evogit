def bin_packing(items: list[float]) -> list[int]:
    """Given a list of items, pack them into bins with capacity 1.0, minimizing the number of bins used.
    The function should returns a packing plan,
    where each item is assigned to a bin encoded as an integer from 0 to n-1,
    and n is the total number of bins used.
    The time complexity of the function should be less than O(n^2).

    For example:
    >>> items = [0.8, 0.01, 0.2, 0.99]
    >>> bin_packing(items)
    [0, 1, 0, 1]

    The packing plan above means that the first and third items are packed into the first bin, and the second and fourth items are packed into the second bin.
    """
    return list(range(len(items)))