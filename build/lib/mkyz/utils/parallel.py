# mkyz/utils/parallel.py
"""Parallel processing utilities for MKYZ library."""

import concurrent.futures
from typing import Any, Callable, List, Optional, TypeVar, Iterator
import multiprocessing

T = TypeVar('T')
R = TypeVar('R')


def parallel_map(func: Callable[[T], R],
                items: List[T],
                n_jobs: int = -1,
                use_threads: bool = True,
                show_progress: bool = False) -> List[R]:
    """Apply a function to items in parallel.
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        n_jobs: Number of parallel workers (-1 for all CPUs)
        use_threads: If True, use threads; if False, use processes
        show_progress: Whether to show progress bar (requires rich)
        
    Returns:
        List of results in the same order as input items
    
    Examples:
        >>> from mkyz.utils import parallel_map
        >>> def square(x):
        ...     return x ** 2
        >>> parallel_map(square, [1, 2, 3, 4], n_jobs=2)
        [1, 4, 9, 16]
    """
    if not items:
        return []
    
    # Determine number of workers
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    elif n_jobs < 1:
        n_jobs = 1
    
    # For small lists, don't bother with parallelism
    if len(items) <= 2 or n_jobs == 1:
        return [func(item) for item in items]
    
    # Choose executor type
    Executor = (concurrent.futures.ThreadPoolExecutor if use_threads 
                else concurrent.futures.ProcessPoolExecutor)
    
    results = [None] * len(items)
    
    if show_progress:
        try:
            from rich.progress import Progress, TaskID
            with Progress() as progress:
                task = progress.add_task("Processing...", total=len(items))
                with Executor(max_workers=n_jobs) as executor:
                    future_to_idx = {
                        executor.submit(func, item): idx 
                        for idx, item in enumerate(items)
                    }
                    for future in concurrent.futures.as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        results[idx] = future.result()
                        progress.advance(task)
        except ImportError:
            # Fall back to no progress bar
            show_progress = False
    
    if not show_progress:
        with Executor(max_workers=n_jobs) as executor:
            future_to_idx = {
                executor.submit(func, item): idx 
                for idx, item in enumerate(items)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
    
    return results


def chunk_data(data: List[T], 
               chunk_size: Optional[int] = None,
               n_chunks: Optional[int] = None) -> Iterator[List[T]]:
    """Split data into chunks for batch processing.
    
    Args:
        data: List of items to chunk
        chunk_size: Size of each chunk (mutually exclusive with n_chunks)
        n_chunks: Number of chunks to create (mutually exclusive with chunk_size)
        
    Yields:
        Chunks of data
    
    Examples:
        >>> list(chunk_data([1, 2, 3, 4, 5], chunk_size=2))
        [[1, 2], [3, 4], [5]]
        
        >>> list(chunk_data([1, 2, 3, 4, 5, 6], n_chunks=3))
        [[1, 2], [3, 4], [5, 6]]
    """
    if not data:
        return
    
    if chunk_size is None and n_chunks is None:
        chunk_size = max(1, len(data) // multiprocessing.cpu_count())
    elif n_chunks is not None:
        chunk_size = max(1, len(data) // n_chunks)
    
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


def get_optimal_workers(data_size: int, 
                       min_items_per_worker: int = 100) -> int:
    """Calculate optimal number of workers based on data size.
    
    Args:
        data_size: Number of items to process
        min_items_per_worker: Minimum items per worker for efficiency
        
    Returns:
        Recommended number of workers
    """
    cpu_count = multiprocessing.cpu_count()
    
    # Don't spawn more workers than we need
    max_needed = max(1, data_size // min_items_per_worker)
    
    return min(cpu_count, max_needed)
