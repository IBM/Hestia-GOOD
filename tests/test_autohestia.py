import os.path as osp

import numpy as np
import pandas as pd

from hestia.autohestia import AutoHestia


def test_autohestia():
    df = pd.read_csv(osp.join(
        osp.dirname(osp.realpath(__file__)), 'biogen_logS.csv')
    )
    df = df[~df['SMILES'].isna()].reset_index(drop=True)
    hestia = AutoHestia(
        df=df.iloc[:100],
        field_name='SMILES',
        label_name='logS',
        task_type='regression',
        data_type='molecule',
        representation='ecfp-4',
        verbose_level='debug'
    )
    out = hestia.run()

    assert 'train' in out and 'test' in out
    assert isinstance(out, dict)
    assert isinstance(out['train'], np.ndarray)