from datasets import IterableDataset
import inspect

# Check the IterableDataset class
print("=== IterableDataset attributes ===")
for attr in dir(IterableDataset):
    if not attr.startswith('__'):
        print(f"  {attr}")

# Check if there's a _ex_iterable property
print("\n=== _ex_iterable ===")
if hasattr(IterableDataset, '_ex_iterable'):
    print(inspect.getsource(IterableDataset._ex_iterable.fget) if hasattr(IterableDataset._ex_iterable, 'fget') else IterableDataset._ex_iterable)
else:
    print("Not found as class attribute")

# Check the iter method
print("\n=== __iter__ ===")
print(inspect.getsource(IterableDataset.__iter__))