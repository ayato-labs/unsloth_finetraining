from datasets import IterableDataset, Features, Sequence, Value

def gen():
    for i in range(10):
        yield {"input_ids": [i, i + 1]}

ds = IterableDataset.from_generator(gen, features=Features({"input_ids": Sequence(feature=Value("int32"))}))
print("column_names:", ds.column_names)
print("iter one:", next(iter(ds))["input_ids"])