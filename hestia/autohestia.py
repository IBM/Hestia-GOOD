import os
import os.path as osp
import shutil
import sys

from copy import deepcopy
from datetime import datetime
from itertools import product
from typing import Callable, Dict, List, Union

import logging
import numpy as np
import pandas as pd
import pickle
import polars as pl
import yaml

try:
    from autopeptideml.reps import PLMs, CLMs, FPs
    from autopeptideml.reps.fps import RepEngineFP
    from autopeptideml.reps.lms import RepEngineLM
    from autopeptideml.train.metrics import evaluate
except ImportError:
    raise ImportError(
        "autopeptideml package is required for AutoHestia. "
        "Please install it via pip: ``pip install autopeptideml``"
    )
from hestia.partition import ccpart, butina, umap_original
from hestia.similarity import (
    embedding_similarity, sequence_similarity_mmseqs, molecular_similarity
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from tqdm import tqdm

__version__ = '1.1.0'


AVAILABLE_ALGORITHMS = {
    'ccpart': ccpart,
    'butina': butina,
    'umap': umap_original
}
UMAP_FPS = {
    'molecule': ['ecfp-2', 'ecfp-3', 'ecfp-4', 'ecfp-6'],
    'peptide': ['ecfp-3', 'ecfp-4', 'ecfp-6', 'ecfp-8'],
    'sequence': []
}
AVAILABLE_METRICS = yaml.safe_load(
    open(osp.join(
        osp.dirname(osp.realpath(__file__)),
        'utils', 'auto-metrics.yml'
    ), 'r')
)

algorithm_list = '\n    - '.join(AVAILABLE_ALGORITHMS.keys())
metrics_list: str = '\n    - '.join(AVAILABLE_METRICS.keys())

MESSAGE_NO_DATA_TYPE = f"""
Data type: ``[data_type]`` not implemented.

Available data types are:
    ``
    - {metrics_list}
    ``

If none of this suit your use case, please feel free to open
an issue in the Github repository requesting the
data type you are interested in:
https://github.com/IBM/Hestia-GOOD/issues
"""
MESSAGE_NO_ALGORITHM = f"""
Algorithm: ``[algorithm]`` not implemented.

Available algorithms are:
    ``
    - {algorithm_list}
    ``
If none of them suit your use case, you can add custom algorithms,
through the ``add_custom_algorithm`` variable. It expects
a dictionary with the name of the method as key and as values:

    - A function that implements the new partitioning algorithm.
    The function has to accept at least the ``df`` as an argument
    and should output two np.ndarrays with the indices for training subset
    and testing subset.
    - A list with all desired hyperparameters for the custom method. For
    example the similarity threshold or the number of clusters.
"""


class TqdmHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)  # , file=sys.stderr)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            sys.exit(0)
            raise KeyboardInterrupt
        except:
            self.handleError(record)


def _define_mol_sim(
        rep: str,
        radius: int,
        nbits: int,
        sim_index: str,
        threads: int = -1
) -> Callable:
    def mol_sim(
        df_query: pd.DataFrame,
        field_name: str,
        threshold: float = 0.1
    ) -> pl.DataFrame:
        return molecular_similarity(
            df_query=df_query,
            field_name=field_name,
            rep=rep,
            radius=radius,
            nbits=nbits,
            sim_index=sim_index,
            threshold=threshold,
            verbose=0,
            threads=threads
        )
    return mol_sim


def _define_emb_sim(
        model: str,
        device: str,
        sim_index: str,
        threads: int = -1
) -> Callable:
    def emb_sim(
        df_query: pd.DataFrame,
        field_name: str,
        threshold: float = 0.1
    ) -> pl.DataFrame:
        re = RepEngineLM(
            model=model
        )
        re.move_to_device(device)
        x = re.compute_reps(
            df_query[field_name].tolist(),
            verbose=False
        )
        return embedding_similarity(
            query_embds=x,
            sim_function=sim_index,
            threshold=threshold,
            threads=threads,
            verbose=0
        )
    return emb_sim


def define_logger() -> logging.Logger:
    logger = logging.getLogger("AutoHestia")
    console_handler = logging.StreamHandler()
    logger_formatter = logging.Formatter(
        '{message}',
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(logger_formatter)
    # logger.addHandler(console_handler)
    logger.addHandler(TqdmHandler())
    return logger


def welcome():
    mssg = f"AutoHestia v.{__version__}\n"
    mssg += "By Raul Fernandez-Diaz"
    max_width = max([len(line) for line in mssg.split('\n')])
    out = "-" * (max_width + 4) + "\n"
    for line in mssg.split('\n'):
        out += "| " + line + " " * (max_width - len(line)) + " |\n"
    out += "-" * (max_width + 4) + "\n"
    return out


class AutoHestia:
    def __init__(
        self,
        df: pd.DataFrame,
        field_name: str,
        label_name: str,
        data_type: str = 'molecule',
        device: str = 'cpu',
        task_type: str = 'classification',
        algorithms: List[str] = ['ccpart', 'butina', 'umap'],
        # add_custom_metrics: Dict[str, Callable] = {},
        # add_custom_algorithm: Dict[str, Callable] = {},
        representation: Union[str, Callable] = 'ecfp-4',
        eval_model: str = 'svm',
        outdir: str = 'autohestia_experiment',
        n_jobs: int = -1,
        verbose_level: str = "info",
    ):
        config = deepcopy(locals())
        config['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        del config['self'], config['df']

        # Defining logger
        self.logger = define_logger()
        if verbose_level.lower() == 'debug':
            self.logger.setLevel(logging.DEBUG)
        elif verbose_level.lower() == 'info':
            self.logger.setLevel(logging.INFO)
        elif verbose_level.lower() == 'warning':
            self.logger.setLevel(logging.WARNING)
        else:
            self.logger.setLevel(logging.ERROR)

        self.logger.info(welcome())

        # Input validations
        if osp.isdir(outdir):
            self.logger.warning(
                f"WARNING: Output directory ``{outdir}``"
                " already exists. Results might be overwritten.\n"
            )
        if field_name not in df:
            raise ValueError(
                f"""Field name: ``{field_name}`` not present in dataframe.
                Please double check. Existing columns in df are:
                ``{', '.join(df.columns.tolist())}``"""
            )
        if data_type not in AVAILABLE_METRICS:
            raise NotImplementedError(
                MESSAGE_NO_DATA_TYPE.replace(
                    '[data_type]', data_type
                )
            )
        for algorithm in algorithms:
            if algorithm not in AVAILABLE_ALGORITHMS:
                raise NotImplementedError(
                    MESSAGE_NO_ALGORITHM.replace(
                        '[algorithm]', algorithm
                    )
                )

        # Setting attributes
        self.df = df
        self.field_name = field_name
        self.label_name = label_name
        self.data_type = data_type
        self.rep = representation
        self.device = device
        self.task_type = task_type
        self.eval_model = eval_model
        self.njobs = n_jobs if n_jobs != -1 else os.cpu_count()
        self.metrics = self._get_metrics(data_type)
        self.umap_metrics = UMAP_FPS[self.data_type]
        self.algorithms = algorithms
        self.outdir = outdir

        # Logging configuration
        self.logger.info("** AutoHestia configuration: **")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Data type: {self.data_type}")
        self.logger.info(f"Task type: {self.task_type}")
        self.logger.info(f"Algorithms: {', '.join(self.algorithms)}")
        self.logger.info(f"Representation: {self.rep}")
        self.logger.info(f"Evaluation model: {self.eval_model}")
        self.logger.info(f"Dataset size: {self.df.shape[0]:,} samples")
        self.logger.info(f"Number of threads: {self.njobs}")
        self.logger.info("")

        # Save metadata
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(osp.join(outdir, 'parts'), exist_ok=True)
        os.makedirs(osp.join(outdir, "metadata"), exist_ok=True)
        config['metrics'] = list(self.metrics.keys())
        if 'umap' in self.algorithms:
            config['umap-metrics'] = self.umap_metrics
        yaml.safe_dump(config, open(osp.join(outdir, "metadata",
                                             'experiment-config.yml'),
                                    'w'))
        df.to_csv(osp.join(outdir, "metadata", "dataset.csv"),
                  index=True)

    def run(self):
        self.logger.info("** Running AutoHestia **")
        self.logger.info("1 - Representing data")
        self.x = self._represent_data()
        self.y = self.df[self.label_name].to_numpy()

        self.logger.info("2 - Prepare and evaluate partitions")
        results = []

        if 'ccpart' in self.algorithms or 'butina' in self.algorithms:
            results.extend(self._eval_sim_parts())
        if 'umap' in self.algorithms:
            results.extend(self._eval_fp_parts())

        self.logger.info("3 - Save and interpret results")
        results_df = pd.DataFrame(results)
        metric = 'mcc' if self.task_type == 'classification' else "spcc"
        results_df = results_df.sort_values(metric).reset_index(drop=True)
        results_df.loc[0, 'best'] = "Y"
        results_df.loc[1:, 'best'] = "N"
        results_df.to_csv(osp.join(self.outdir, "parts-results.tsv"),
                          index=False, sep="\t")
        shutil.copy(osp.join(self.outdir, 'parts', f"{results_df.loc[0, 'part-alg']}-{results_df.loc[0, 'metric']}.pckl"),
                             osp.join(self.outdir, 'best-partition.pckl'))
        return pickle.load(open(osp.join(self.outdir, 'best-partition.pckl'), 'rb'))

    def run_good(self):
        self.logger.info("** Running AutoHestia **")
        self.logger.info("1 - Representing data")
        self.x = self._represent_data()
        self.y = self.df[self.label_name].to_numpy()

    def _eval_sim_parts(self) -> List[dict]:
        model = self._get_model()
        pbar = tqdm(self.metrics.items(), total=len(self.metrics),
                    desc="  Metric")
        results = []

        for metric_name, metric_func in pbar:
            pbar.set_description(f"  Sim Mtx - {metric_name}")
            sim_df = metric_func(
                df_query=self.df,
                field_name=self.field_name,
                threshold=0.1
            )

            # CCPart eval
            if 'ccpart' in self.algorithms:
                mdl = deepcopy(model)
                for th in range(10, 100, 10):
                    cparts = ccpart(
                        df=self.df,
                        sim_df=sim_df,
                        field_name=self.field_name,
                        label_name=self.label_name,
                        test_size=0.2,
                        threshold=th/100,
                    )
                    ccpart_config = {
                        'th': th / 100
                    }
                    cparts = {
                        'train': np.array(cparts[0]),
                        'test': np.array(cparts[1])
                    }
                    if len(cparts['test']) >= 0.185 * len(self.df):
                        break

                cparts_file = osp.join(self.outdir, 'parts',
                                       f'ccpart-{metric_name}.pckl')
                pickle.dump(cparts, open(cparts_file, 'wb'))
                mdl.fit(self.x[cparts['train']], self.y[cparts['train']])
                if self.task_type == 'classification':
                    preds = mdl.predict_proba(self.x[cparts['test']])
                else:
                    preds = mdl.predict(self.x[cparts['test']])
                result = evaluate(
                    preds, self.y[cparts['test']],
                    pred_task='reg' if 'reg' in self.task_type else 'class'
                )
                result['metric'] = metric_name
                result['part-alg'] = 'ccpart'
                result['part-alg-config'] = ccpart_config
                results.append(result)

            # Butina eval
            if 'butina' in self.algorithms:
                mdl = deepcopy(model)

                for th in range(10, 100, 10):
                    bparts = butina(
                        df=self.df,
                        sim_df=sim_df,
                        field_name=self.field_name,
                        label_name=self.label_name,
                        test_size=0.2,
                        threshold=th/100,
                    )
                    butina_config = {
                        'th': th / 100
                    }
                    bparts = {
                        'train': np.array(bparts[0]),
                        'test': np.array(bparts[1])
                    }
                    if len(bparts['test']) >= 0.185 * len(self.df):
                        break
                bparts_file = osp.join(self.outdir, 'parts',
                                       f'ccpart-{metric_name}.pckl')
                pickle.dump(bparts, open(bparts_file, 'wb'))
                mdl.fit(self.x[bparts['train']], self.y[bparts['train']])
                if self.task_type == 'classification':
                    preds = mdl.predict_proba(self.x[bparts['test']])
                else:
                    preds = mdl.predict(self.x[bparts['test']])
                result = evaluate(
                    preds, self.y[bparts['test']],
                    pred_task='reg' if 'reg' in self.task_type else 'class'
                )
                result['metric'] = metric_name
                result['part-alg'] = 'butina'
                result['part-alg-config'] = butina_config
                results.append(result)
        return results

    def _eval_fp_parts(self) -> List[dict]:
        model = self._get_model()

        pbar = tqdm(self.umap_metrics, desc="  FP")
        results = []

        for metric in pbar:
            mdl = deepcopy(model)
            pbar.set_description(f"  FP - {metric}")
            combs = product(list(range(10, 100, 10)),
                            list(range(10, 200, 20)),
                            list(range(10, 100, 10)))

            for (th, n_pcs, n_clus) in combs:
                parts = umap_original(
                    df=self.df,
                    field_name=self.field_name,
                    label_name=self.label_name,
                    test_size=0.2,
                    threshold=th/100,
                    n_clusters=n_clus,
                    n_pcs=n_pcs,
                    radius=int(metric.split('-')[1]),
                    verbose=0,
                    bits=1024
                )
                umap_config = {
                    'th': th/100,
                    'n_pcs': n_pcs,
                    'n_clus': n_clus
                }
                parts = {
                    'train': np.array(parts[0]),
                    'test': np.array(parts[1])
                }
                if len(parts['test']) >= 0.185 * len(self.df):
                    break

            parts_file = osp.join(self.outdir, 'parts', f'umap-{metric}.pckl')
            pickle.dump(parts, open(parts_file, 'wb'))
            mdl.fit(self.x[parts['train']], self.y[parts['train']])
            if self.task_type == 'classification':
                preds = mdl.predict_proba(self.x[parts['test']])
            else:
                preds = mdl.predict(self.x[parts['test']])
            result = evaluate(
                preds, self.y[parts['test']],
                pred_task='reg' if 'reg' in self.task_type else 'class'
            )
            result['metric'] = metric
            result['part-alg'] = 'umap'
            result['part-alg-config'] = umap_config
            results.append(result)
        return results

    def _get_metrics(self, data_type: str) -> Dict[str, Callable]:
        metrics = AVAILABLE_METRICS[data_type]
        for metric_name, conf in metrics.items():
            if conf['type'] == 'fp':
                metrics[metric_name] = _define_mol_sim(
                    rep=conf['rep']['rep'], radius=conf['rep']['radius'],
                    nbits=conf['rep']['nbits'], sim_index=conf['sim_index'],
                    threads=self.njobs
                )
            elif conf['type'] == 'lm':
                metrics[metric_name] = _define_emb_sim(
                    model=conf['rep']['model'], device=self.device,
                    sim_index=conf['sim_index'], threads=self.njobs
                )
            elif conf['type'] == 'sequence':
                if conf['method'] == 'mmseqs':
                    metrics[metric_name] = sequence_similarity_mmseqs
                else:
                    raise NotImplementedError(
                        f"Sequence similarity method "
                        f"``{conf['method']}`` not implemented."
                    )
            else:
                raise NotImplementedError(
                    f"Metric type: ``{conf['type']}`` not implemented."
                )
        return metrics

    def _represent_data(self):
        if callable(self.rep):
            x = self.rep(
                self.df[self.field_name].tolist()
            )
        elif self.rep in PLMs + CLMs:
            rep_engine = RepEngineLM(
                model_name=self.representation,
            )
            rep_engine.move_to_device(self.device)
            x = rep_engine.compute_reps(
                self.df[self.field_name].tolist(),
                verbose=self.logger.isEnabledFor(logging.INFO)
            )
        elif self.rep.split('-')[0] in FPs:
            if len(self.rep.split('-')) == 1:
                self.rep += '-2-1024'
            elif len(self.rep.split('-')) == 2:
                self.rep += '-1024'

            rep_engine = RepEngineFP(
                rep=self.rep.split('-')[0],
                nbits=int(self.rep.split('-')[2]),
                radius=int(self.rep.split('-')[1])
            )
            x = rep_engine.compute_reps(
                self.df[self.field_name].tolist()
            )
        else:
            raise ValueError(
                f"Representation: ``{self.rep}`` is not valid. "
                "It should be either a string "
                "indicating the fingerprint type or a callable "
                "function that takes as input the n elements in ``field_name``"
                "and outputs an n x m np.ndarray where m is the dimensions of "
                "the representation. "
                "Available fingerprints are: "
                f"``\n    - {'\n    - '.join([f+'-{radius}-{bits}' for f in FPs])}\n``."
            )
        return x

    def _get_model(self):
        if self.eval_model == 'svm':
            if self.task_type == 'classification':
                mdl = SVC(class_weight='balanced',
                          probability=True, kernel='linear')
            elif self.task_type == 'regression':
                mdl = SVR(kernel='linear')
        elif self.eval_model == 'rf':
            if self.task_type == 'classification':
                mdl = RandomForestClassifier(
                    n_jobs=-1, class_weight='balanced'
                )
            elif self.task_type == 'regression':
                mdl = RandomForestRegressor(n_jobs=-1)
        elif self.eval_model == 'knn':
            if self.task_type == 'classification':
                mdl = KNeighborsClassifier(
                    n_jobs=-1
                )
            elif self.task_type == 'regression':
                mdl = KNeighborsRegressor(
                    n_jobs=-1
                )
        else:
            raise NotImplementedError(
                f"Model: ``{self.eval_model}`` is not implemented."
                "Availabel models: ``svm``, ``rf``, and ``knn``"
            )
        return mdl


if __name__ == '__main__':
    df = pd.read_csv(osp.join(
            osp.dirname(osp.realpath(__file__)),
            '..', 'tests', 'biogen_logS.csv')
    )
    df = df[~df['SMILES'].isna()].reset_index(drop=True)
    hestia = AutoHestia(
        df=df.iloc[:100],
        field_name='SMILES',
        label_name='logS',
        task_type='regression',
        data_type='molecule',
        algorithms=['butina', 'umap'],
        representation='ecfp-4',
        verbose_level='debug'
    )
    out = hestia.run()

    print(out)