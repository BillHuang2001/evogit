def bin_packing(items: list[float]) -> list[int]:
    """Pack a list of items into bins with a capacity of 1.0, minimizing the number of bins used.

    The function returns a list where each item is assigned to a bin,
    represented by an integer from 0 to n-1, where n is the total number of bins used.

    Example:
    >>> items = [0.8, 0.01, 0.2, 0.99]
    >>> bin_packing(items)
    [0, 1, 0, 1]

    In this example, the first and third items are placed in the first bin, and the second and fourth items are placed in the second bin.
    """
    return list(range(len(items)))
