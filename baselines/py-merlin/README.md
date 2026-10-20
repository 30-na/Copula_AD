<!--
SPDX-FileCopyrightText: 2022 German Aerospace Center (DLR e.V.), Ferdinand Rewicki

SPDX-License-Identifier: Apache-2.0
-->

# py-merlin [![pipeline status](https://gitlab.com/dlr-dw/py-merlin/badges/main/pipeline.svg)](https://gitlab.com/dlr-dw/py-merlin/-/commits/main) [![coverage report](https://gitlab.com/dlr-dw/py-merlin/badges/main/coverage.svg)](https://gitlab.com/dlr-dw/py-merlin/-/commits/main)

A python implementation of the MERLIN anomaly detection algorithm published by [Nakamura et al. 2020](#nakamura-et-al-2020):
```text
T. Nakamura, M. Imamura, R. Mercer and E. Keogh
"MERLIN: Parameter-Free Discovery of Arbitrary Length Anomalies in Massive Time Series Archives"
2020 IEEE International Conference on Data Mining (ICDM), 
2020, pp. 1190-1195, doi: 10.1109/ICDM50108.2020.00147.
```

# Dependencies
* Python >= v3.9

# Installation
This package can be installed standalone via `git-clone` or as a depencdency via `pip`.
### Standalone
To install the package standalone:
```bash
git clone https://gitlab.com/dlr-dw/py-merlin.git
cd py-merlin
make install
```

### Dependency
To include the package as a dependency into a python project:
```bash
pip install merlin@git+https://gitlab.com/dlr-dw/py-merlin.git
```

### Development
To install the package for development:
```bash
git clone https://gitlab.com/dlr-dw/py-merlin.git
cd py-merlin
make dev
```

# Usage
The following example demonstrates the usage on a univariate timeseries taken from the UCR anomaly archive (see:  [Wu and Keogh 2021](#wu-and-keogh-2021)).
```python
import numpy as np

import merlin
import merlin.utils as utils

TS_NAME = "../tests/fixtures/004_UCR_Anomaly_DISTORTEDBIDMC1_2500_5400_5600.txt"

# 1. Set the range of subsequence lengths to be searched
L_MIN = 170
L_MAX = 190

# 2. load timeseries from file
ts = np.loadtxt(TS_NAME, delimiter=",", dtype=float)

# 3. run MERLIN
discords, distances, lengths = merlin.merlin(ts, lmin=L_MIN, lmax=L_MAX)

# 4. plot result
name_parts = TS_NAME.split("_")
# extract ground truth label from file name
ground_truth = (int(name_parts[5]), int(name_parts[6].split(".")[0]))
# use a minimal tolerance of 100 points
tolerance = max(ground_truth[1] - ground_truth[0], 100)
# create plot
fig, ax = utils.plot(ts, discords, lengths, anomalies=[ground_truth], name=TS_NAME.split(".")[0], tol=tolerance)
fig.show()
```

Another example can be fone in the jupyter notebook [./examples/demo.ipynb](./examples/demo.ipynb).


# Tests
To run the tests execute
```bash
make test
```
The test fixtures are taken from the UCR anomaly archive (see: [Wu and Keogh 2021](#wu-and-keogh-2021)).

To generate the coverage report, run:
```bash
make coverage
```

# Documentation
To generate the documentation, run:
```bash
make docu
```

# Citation
If you use this software, please cite it using the data provided in `CITATION.cff`. Please also consider citing the original publication from [Nakamura et al. 2020](#nakamura-et-al-2020). 

# References

#### Nakamura et al. 2020 
> T. Nakamura, M. Imamura, R. Mercer and E. Keogh, "MERLIN: Parameter-Free Discovery of Arbitrary Length Anomalies in Massive Time Series Archives," 2020 IEEE International Conference on Data Mining (ICDM), 2020, pp. 1190-1195, doi: 10.1109/ICDM50108.2020.00147.

#### Wu and Keogh 2021
> R. Wu and E. Keogh, "Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress", IEEE Transactions on Knowledge and Data Engineering, 2021, Institute of Electrical and Electronics Engineers (IEEE), doi: 10.1109/tkde.2021.3112126.

# License
Please refer to the `LICENSE.md` file for further information about how the content is licensed. 
