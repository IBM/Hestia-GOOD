import os.path as osp

import numpy as np
import pandas as pd

from hestia.autohestia import AutoHestia
from hestia.similarity import molecular_similarity


def test_autohestia():
    df = pd.read_csv(osp.join(
        osp.dirname(osp.realpath(__file__)), 'biogen_logS.csv')
    )
    df = df[~df['SMILES'].isna()].reset_index(drop=True).iloc[:100]

    # Pre-compute fingerprints (x) from SMILES
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    x = np.array([
        gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(smi))
        for smi in df['SMILES']
    ], dtype=np.float32)

    y = df['logS'].to_numpy()

    # Pre-compute similarity DataFrames
    sim_dfs = {
        'ecfp-4-t': molecular_similarity(
            df, field_name='SMILES',
            fingerprint='ecfp', sim_function='tanimoto',
            verbose=0
        ),
        'ecfp-6-t': molecular_similarity(
            df, field_name='SMILES',
            fingerprint='ecfp', sim_function='tanimoto',
            verbose=0, radius=3
        ),
        'ecfp-8-t': molecular_similarity(
            df, field_name='SMILES',
            fingerprint='ecfp', sim_function='tanimoto',
            verbose=0, radius=4
        ),
        'mapc-4-j': molecular_similarity(
            df, field_name='SMILES',
            fingerprint='mapc', sim_function='jaccard',
            verbose=0
        ),
    }

    save_dir = osp.join(osp.dirname(osp.realpath(__file__)), 'test_autohestia')

    hestia = AutoHestia(
        df=df,
        field_name='SMILES',
        x=x,
        y=y,
        sim_dfs=sim_dfs,
        verbose_level='debug',
    )
    out = hestia.best_guardrailed_splits(
        part_algs=['dissimilarity', 'ccpart', 'butina'],
        save_dir=save_dir, overwrite=True
    )

    assert isinstance(out, dict)
    assert 'raw-experiments' in out
    assert 'main-stats' in out
    assert 'after-guardrail' in out
    assert 'top-combination' in out
    assert 'best-parts' in out
    assert isinstance(out['best-parts'], dict)
    assert osp.exists(osp.join(save_dir, 'raw_experiments.csv'))
    assert osp.exists(osp.join(save_dir, 'main_stats.csv'))
