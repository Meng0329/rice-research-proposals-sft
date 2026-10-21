# Rice breeding SFT analysis code

This repository contains the analysis script and derived input tables for the biological question atlas (Figure 2), gene and locus profile (Figure 3), and biological objective profile (Table 2) in the associated data descriptor.

## Reproduce the displays

Use Python 3.10 or later. From the repository root, install the dependencies and run:

```sh
python -m pip install -r requirements.txt
python code/manuscript/make_macro_biological_displays.py
```

The script writes the three displays as PDF, PNG and SVG files to `figures/`, and writes the table source CSV to `source_data/macro_biological_objective_atlas.csv`.

## Inputs

The CSV files in `source_data/biological_question_profile/` are derived record-level annotations and crosswalks used by the script. They contain no full-text source papers. The complete 6,670-record dataset and its release files are available from [BasicAgriData, record 649132100304896](https://hub.basicagridata.cn/#/dataset/detail/649132100304896).

## Data licence

The released dataset database and the derived tabular inputs are made available under the [Open Data Commons Attribution License (ODC-By) v1.0](https://opendatacommons.org/licenses/by/1-0/). The licence applies to database rights; rights in material originating from source papers, where applicable, remain with their respective holders.

The software source code is provided publicly in this repository. No separate software licence has been specified for the code.
