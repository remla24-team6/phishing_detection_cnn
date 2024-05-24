"""
memory.py
"""

import os
import psutil


def process_memory():
    """
    Get the memory usage of the current process.

    Returns:
    int: The memory usage in bytes.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

# decorator function


def profile(func):
    """
    Decorator function to profile memory usage before and after a function call.

    Args:
    func (function): The function to be profiled.

    Returns:
    function: The wrapper function.
    """

    def wrapper(*args, **kwargs):

        mem_before = process_memory()
        _ = func(*args, **kwargs)
        mem_after = process_memory()
        print(f"{ func.__name__}:\nconsumed memory - before: {mem_before}\nafter: {mem_after}")

        return mem_after - mem_before
    return wrapper


@profile
def train_with_memory(func):
    """
    Profile memory usage before and after calling the provided function.

    Args:
    func (function): The function to be called.
    """
    func()
