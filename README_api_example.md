# API shape

The main API is simple:

```python
from skinnydiffing import diff

result = diff("source.parquet", "target.parquet", keys="id")

# Check how many cells changed
print(f"Found {result.n_diffs} cell differences.")

# Access the underlying Polars LazyFrame of differences
diff_df = result.diff.collect()

# Write all structural changes and cell differences to CSVs
result.write("./diff_output_directory")
```

Cell differences are always computed for rows present in both datasets. Rows that exist
only in the source or only in the target are reported separately on the result object.

Join keys are required by default. If you really want strict row-position alignment,
use the positional parameter:

```python
result = diff("source.parquet", "target.parquet", positional=True)
```

Supported inputs:

```python
# 1. Most normal: paths
result = diff("source.parquet", "target.parquet", keys="id")

# 2. Dataframe-native: Polars, pandas, PyArrow, etc. via Narwhals
result = diff(source_df, target_df, keys="id")

# 3. One-off infrastructure: zero-argument callables returning supported inputs
result = diff(lambda: load_source(), lambda: load_target(), keys="id")

# functools.partial also works here, just as long as the final callable needs no arguments
from functools import partial

result = diff(
    partial(load_table, spark=spark, name="prod.source_people"),
    partial(load_table, spark=spark, name="prod.target_people"),
    keys="id",
)

# 4. Config/repeated infrastructure: one-item loader calls
result = diff(
    {"spark": "prod.source_people"},
    {"spark": "prod.target_people"},
    keys="id",
    loaders={"spark": load_spark_table},
)
```

A loader receives the payload exactly as supplied and returns any supported input:

```python
def load_spark_table(table_name):
    return spark.table(table_name)
```

Payloads can be structured however the loader wants:

```python
def load_extract(args):
    return get_extract(
        dataset=args["dataset"],
        period=args["period"],
        version=args["side"],
    )

result = diff(
    {"extract": {"dataset": "people", "period": "2024-01", "side": "source"}},
    {"extract": {"dataset": "people", "period": "2024-01", "side": "target"}},
    keys="id",
    loaders={"extract": load_extract},
)
```

If the loader needs infrastructure, bind it yourself:

```python
from functools import partial

def load_spark_table(table_name, *, spark):
    return spark.table(table_name)

result = diff(
    {"spark": "prod.source_people"},
    {"spark": "prod.target_people"},
    keys="id",
    loaders={"spark": partial(load_spark_table, spark=spark)},
)
```

YAML config uses the same mechanism:

```yaml
defaults:
  keys: [id]
  normalise: float_strings
  output_dir: ./diff_output

comparisons:
  - name: people
    source:
      spark: prod.source_people
    target:
      spark: prod.target_people

  - name: structured_extract
    source:
      extract:
        dataset: people
        period: 2024-01
        side: source
    target:
      extract:
        dataset: people
        period: 2024-01
        side: target

  - name: positional_export
    source: source_export.csv
    target: target_export.csv
    keys: null
    positional: true
```

```python
from functools import partial

from skinnydiffing import run_config

run_config(
    "diff.yml",
    loaders={"spark": partial(load_spark_table, spark=spark)},
)
```
