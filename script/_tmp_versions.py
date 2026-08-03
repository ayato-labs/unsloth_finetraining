import os
import trl, transformers, datasets
print("trl", trl.__version__)
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print(os.path.dirname(trl.__file__))
