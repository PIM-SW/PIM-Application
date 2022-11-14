# 1. Preprocess

**In case of amazon review datasets**,
`amazon_parse_divide_filter.py` : parse raw data into transactions, split into train/test, filter objects under given frequency (default=10).

```
*****************Must use python >= 3.6*****************
Usage: python3 amazon_parse_divide_filter.py [dataset_name] [optional: min_frequency]
[Home_and_Kitchen, Electronics, ...]
ex. python3 amazon_parse_divide_filter.py Electronics [10]
```

<br />
