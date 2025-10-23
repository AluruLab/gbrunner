import argparse
import pydantic
import logging
import os
import typing as t

import numpy as np

from . import (
    GBMethod,
    GBRArgs,
    STSGBMArgs,
    gbrunner_args,
    grnboost_model_weights_at,
    run_gradient_boosting,
)
from .comm_interface import default_comm
from .data import (
    NetworkGenes,
    DataArray,
    read_h5ad_submatrix,
    target_gene_matrix_h5ad,
)
from .util import parse_yaml


class MPIInputArgs(pydantic.BaseModel):
    method: GBMethod = 'stsgbr'
    data : t.Literal['sparse', 'dense'] = 'dense'
    regressor_args: GBRArgs = pydantic.Field(
        discriminator="regressor",
        default=STSGBMArgs()
    )
    tf_file: str =  "./data/meta/trrust_tf.txt"
    h5ad_file: str = "./data/pbmc800K/pbmc800K.20K.h5ad"
    yaml_file: str = "./config/preprocess/pp_pbmc800K_20K.yaml"
    net_genes: NetworkGenes | None = None


def _log():
    return logging.getLogger(__name__)


def getenv(env_name: str):
    if env_name in os.environ:
        return os.environ[env_name]
    return 'N/A'


def yaml_args(yaml_file: str):
    cfg_dict = parse_yaml(yaml_file)
    run_args = MPIInputArgs.model_validate(cfg_dict)
    if run_args.method != run_args.regressor_args.regressor:
        run_args.regressor_args = gbrunner_args(run_args.method)
    return run_args


def run_grnboost_model(
    rtf_mat: DataArray,
    tgt_value: DataArray,
    tf_index: int,
    ntfs: int,
    regressor_args: GBRArgs,
):
    #
    gbfit = run_gradient_boosting(rtf_mat, tgt_value, regressor_args)
    #
    if gbfit:
        gmodel, gstat = gbfit
        importance_, gstat = grnboost_model_weights_at(
            tf_index,
            ntfs,
            gmodel,
            gstat,
        )
        return importance_, gstat
    else:
        return None, None


def extract_tf_target_data(
    yargs: MPIInputArgs,
    tg_index: int,
    ndecimals: int | None = None
):
    if not yargs.net_genes:
        return None, None
    net_genes: NetworkGenes = yargs.net_genes 
    comm_ifx = default_comm()
    comm_ifx.log_at_root(_log(), logging.DEBUG, "Build TF Matrix")
    tf_mat  = read_h5ad_submatrix(
        yargs.h5ad_file,
        net_genes.tf_indices
    )
    comm_ifx.barrier()
    comm_ifx.log_at_root(_log(), logging.DEBUG, "Build Boosting Inputs")
    target_gene = net_genes.gene_list[tg_index]
    rtf_mat, tgt_value = target_gene_matrix_h5ad(
        yargs.h5ad_file,
        tf_mat,
        net_genes.gene_map,
        net_genes.tf_map,
        target_gene,
    )
    if ndecimals is not None and ndecimals > 0:
        rtf_mat = np.round(
            rtf_mat,
            ndecimals
        ) if rtf_mat is not None else rtf_mat
        tgt_value = np.round(
            tgt_value,
            ndecimals
        ) if tgt_value is not None else tgt_value
    comm_ifx.log_comm(
        _log(), logging.DEBUG, f"Complete {rtf_mat.shape};; {tgt_value.shape}"
    )
    return rtf_mat, tgt_value


def run_grad_boost_for(
    yargs: MPIInputArgs,
    tg_index: int,
    ndecimals: int | None = None
):
    yargs.net_genes = NetworkGenes.from_h5ad(yargs.h5ad_file, yargs.tf_file)
    net_genes: NetworkGenes = yargs.net_genes 
    tf_map = net_genes.tf_map
    target_gene = net_genes.gene_list[tg_index]
    #
    comm_ifx = default_comm()
    rtf_mat, tgt_value = extract_tf_target_data(yargs, tg_index, ndecimals)
    #
    importance_, gstat = run_grnboost_model(
        rtf_mat, 
        tgt_value,
        tf_map[target_gene] if target_gene in tf_map else -1,
        len(tf_map),
        yargs.regressor_args
    )
    result_str = "Failed"
    if importance_ is not None:
        nedges = np.sum(importance_ > 0)
        result_str = f"No. Edges : {nedges} ; Runtime : {gstat.fit_time}"
    comm_ifx.log_comm(_log(), logging.DEBUG, result_str)
    comm_ifx.barrier()
    return importance_, gstat


def run_yaml(
    yaml_file: str,
    tg_index: int | None = None,
    round_up: int | None = None,
):
    comm_ifx = default_comm()
    comm_ifx.log_at_root(_log(), logging.DEBUG, "Load genes/TF data")
    yargs : MPIInputArgs = yaml_args(yaml_file)
    if tg_index is not None and comm_ifx.size == 1:
        comm_ifx.log_at_root(
            _log(),
            logging.DEBUG,
            f"Running for target :: {tg_index};; round_up :: {str(round_up)}"
        )
        return run_grad_boost_for(yargs, tg_index, round_up)
    return run_grad_boost_for(yargs, comm_ifx.rank)


#
def main(
    yaml_file: str = "./config/networks/pp_pbmc800K_20K.yaml",
    tg_index: int | None = None,
    round_up: int | None = None,
):
    comm_ifx = default_comm()
    if comm_ifx.rank == 0: 
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.DEBUG)
    #
    comm_ifx.barrier()
    comm_ifx.log_at_root(
        _log(),
        logging.DEBUG,
        f"""ENV::
            OPENBLAS_NUM_THREADS:: {getenv('OPENBLAS_NUM_THREADS')}
            OMP_NUM_THREADS:: {getenv('OMP_NUM_THREADS')}
        """
    )
    #
    run_yaml(yaml_file, tg_index, round_up)


if __name__ == "__main__":
   parser = argparse.ArgumentParser(
       prog="gbr.cli", description="Generate GRNs w. XGB/lightgbm for Single Cell Data"
   )
   parser.add_argument(
       "yaml_file", 
       default="./config/networks/mpistsgb_pbmc800K_20K.yaml",
       help=f"Yaml Input file with a given configuration. Use 'schema' option to print schema"
   )
   parser.add_argument("--tg_index", type=int, default=None)
   parser.add_argument("--round_up", type=int, default=None)
   run_args = parser.parse_args()
   if default_comm().rank == 0:
       print("Run Arguments :: ", run_args)
   main(run_args.yaml_file, run_args.tg_index, run_args.round_up)
