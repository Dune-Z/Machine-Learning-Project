#!/usr/bin/env python3
'''
Generate submission file for the project
'''

student_id = 'PB20061254'
data_path = "./data/test_data_sample.json"


def main():
    import sys, inspect
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location('test', '../src/test.py')
    sys.path.append('../src')
    test = module_from_spec(spec)
    spec.loader.exec_module(test)
    lines = inspect.getsource(test.main).splitlines()
    lines = ['    ' + line for line in lines[1:-1]]
    lines = '\n'.join(lines)

    src = f'''#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

DATA_PATH = "{data_path}"

## for TA's test
## you need to modify the class name to your student id.
## you also need to implement the predict function, which reads the .json file,
## calls your trained model and returns predict results as an ndarray

class {student_id}():
    def predict(self, data_path): 
        #### This function is generated by `scripts/generate.py`
        import sys
        sys.path.append('./src')
        class args:
            in_dir = './models'
            data = data_path
        #### Predict function begins here
{lines} 
        # Our OrdinalEncoder maps the labels to 0, 1, 2, so we need to plus 1
        return y_pred + 1


## for local validation
if __name__ == '__main__':
    with open(DATA_PATH, "r") as f:
        test_data_list = json.load(f)
    true = np.array([int(data["fit"]) for data in test_data_list])
    bot = {student_id}()
    pred = bot.predict(DATA_PATH)

    macro_f1 = f1_score(y_true=true, y_pred=pred, average="macro")
    print(macro_f1)
'''

    with open(f'../{student_id}.py', 'w') as f:
        f.write(src)


if __name__ == '__main__':
    main()