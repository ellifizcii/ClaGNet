# Input data

Raw data are **not** distributed with this repository. Place the files below in this
directory (or pass `--data-dir /path/to/data`). `data/\\\*.csv` is git-ignored.

## File naming

For each dataset `<name>`, two CSV files are expected:

|File|Content|
|-|-|
|`data\\\_<name>.csv`|Expression matrix. **Genes are rows, cells are columns.** The first column holds gene identifiers; every remaining column is one cell.|
|`labs\\\_<name>.csv`|Cell-type labels, one row per cell, in the same order as the matrix columns. Must contain a `cell\\\_type` or `x` column. An optional first column (`Unnamed: 0` / `cell\\\_id`) holding integer cell indices `1..N` is used for matching.|

Expected `<name>` values: `li`, `zeisel`, `tasic`, `darmanis`, `xin`, `kolodziejczyk`, `baronm1`, `pollen`
(see `datasets.example.json`).

## Validation

```bash
python scripts/check\\\_data.py --data-dir data
```

The check confirms that all eight dataset pairs exist, that a label column is present, and that
the number of matrix columns (cells) equals the number of label rows.

## Sources

The datasets are public single-cell benchmark collections. Please obtain them from their original
publications/repositories and cite the original authors:



