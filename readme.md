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
tbfs.fit(x, y)

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

### Saving / Loading Feature Rankings

We can save feature rankings for future experiments and re-use them as needed.

```python
# save results
tbfs.to_csv('tbfs-results.csv')

...

# load results and re-use
tbfs2 = TBFSRanker()
tbfs2.from_csv('tbfs-results.csv')
tbfs2.top_k_features_by_metric(10, 'f-score')
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

### Custom Threshold Ranges

Setting t_delta to 0.0001 would cause the TBFS ranker to enumerate 10,000 thresholds and will drastically increase run time.

We can use custom thresholds to decrease/increase t-delta in certain regions of the search space.

The below example reduces 10,000 thresholds to < 500 thresholds.

```python
thresholds = [
  *np.arange(0, 0.01, 0.0001),
  *np.arange(0.01, 0.1, 0.001),
  *np.arange(0.1, 0.9, 0.01),
  *np.arange(0.9, 0.99, 0.001),
  *np.arange(0.99, 1.0001, 0.0001)
]
tbfs = TBFSRanker(thresholds=)
tbfs.fit(x,y)
...
```
