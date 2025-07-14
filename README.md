# GRN with Gradient Boosting

Utilities and runner scripts for generating gene regulatory networks with arboreto, XGBoost and Light GBM.

## Usage

Runner can be used to run either Arboretor, Sckit Learn Boosting, XG Boost or
light GBM using the AnnData h5ad file, CSV file as input or an yaml with the
specificed input parameters

Use one of csv, ad or yaml to select the type of input as follows:

    python -m gbr.cli
           [-h] {csv,ad,yaml} ...

           Generate GRNs w. XGB/lightgbm for Single Cell Data

           positional arguments:
              {csv,ad,yaml}  CSV file / AnnData H5AD file / YAML with the data locations
                csv          CSV File as Input
                ad           H5AD File as Input
                yaml         YAML File as Input

### Command-line parameters for csv file as input

Options to provide device, method, statistics output file and network output
file are mandatory. Option to provide csv file as input is as follows:

     python -m gbr.cli csv
            [-h] [--take_n TAKE_N]
            --csv_file CSV_FILE
            [--device {cpu,gpu}]
            [--method {xgb,stxgb,sgbr,stsgbr,lgb,stlgb,arb:default,arb:gbm,none}]
            [--rstats_out_file RSTATS_OUT_FILE]
            [--out_file OUT_FILE]

     options:
        -h, --help show this help message and exit
        --take_n TAKE_N
        --csv_file CSV_FILE
        --device, -c {cpu,gpu}
        --method, -m {xgb,stxgb,sgbr,stsgbr,lgb,stlgb,arb:default,arb:gbm,none}
        --rstats_out_file RSTATS_OUT_FILE
        --out_file OUT_FILE

### Command-line parameters for anndata h5ad file as input

Options to provide device, method, statistics output file and network output
file are mandatory. Option to provide AnnData h5ad file as input is as follows:

    python -m gbr.cli ad
           [-h]
           [--device {cpu,gpu}]
           [--method {xgb,stxgb,sgbr,stsgbr,lgb,stlgb,arb:default,arb:gbm,none}]
           [--ntop_genes NTOP_GENES]
           [--data {dense,sparse}]
           [--nsub_cells NSUB_CELLS]
           [--tf_file TF_FILE]
           [--h5ad_file H5AD_FILE]
           [--select_hvg SELECT_HVG]
           [--take_n TAKE_N]
           [--use_tqdm]
           [--rstats_out_file RSTATS_OUT_FILE]
           [--out_file OUT_FILE]

    options:
      -h, --help            show this help message and exit
      --device, -c {cpu,gpu}
      --method, -m {xgb,stxgb,sgbr,stsgbr,lgb,stlgb,arb:default,arb:gbm,none}
      --ntop_genes NTOP_GENES
      --data, -d {dense,sparse}
      --nsub_cells NSUB_CELLS
      --tf_file TF_FILE
      --h5ad_file H5AD_FILE
      --select_hvg SELECT_HVG
      --take_n TAKE_N
      --use_tqdm
      --rstats_out_file RSTATS_OUT_FILE
      --out_file OUT_FILE

### Command-line parameters for yaml file as input

Options for yaml file as input is as follows:

    python -m gbr.cli yaml [-h] yaml_file

    positional arguments:
        yaml_file Yaml Input file with a given configuration. Use 'schema' option to print schema

    options:
        -h, --help show this help message and exit

To output the schema run `python -m gbr.cli schema`. Schema and examples given in
data/config.
