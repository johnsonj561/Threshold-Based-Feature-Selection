# Threshold Based Feature Selection

Python implementation of [TBFS](https://digitalcommons.wku.edu/cgi/viewcontent.cgi?article=1005&context=comp_sci) feature ranking algorithm.

## Installation

```
pip install git+https://github.com/johnsonj561/Threshold-Based-Feature-Selection
```

## Usage

See [notebook](example-usage.ipynb) for example usage.

```python
from tbfs.ranker import TBFSRanker, metrics
import pandas as pd

# prepare data features and labels
df = pd.read_csv('sample-data.csv')
y, x = df['class'], df.drop(columns=['class'])
y = np.where(y == 'ACL', 1, 0)

# fit TBFS ranker
tbfs = TBFSRanker(t_delta=0.01)
rankings = tbfs.fit(x, y)

# save the results for future experiments
tbfs.to_csv('tbfs-results.csv')

# take top K features for a metric
tbfs.top_k_features_by_metric(10, 'f-score')
```

```
['GENE1609X',
 'GENE1537X',
 'GENE493X',
 'GENE1616X',
 'GENE3945X',
 'GENE3258X',
 'GENE3946X',
 'GENE384X',
 'GENE1296X',
 'GENE1620X']
```

```python
# load results and re-use
tbfs2 = TBFSRanker()
tbfs2.from_csv('tbfs-results.csv')
tbfs2.top_k_features_by_metric(10, 'f-score')
```
