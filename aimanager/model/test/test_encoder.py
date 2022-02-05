from aimanager.model.encoder import int_to_ordinal, ordinal_to_int, joined_encoder
import numpy as np
import pandas as pd


def test_ordinal_conversion():
    # testing the conversion

    integers = np.random.randint(0, 21, 30)

    integers = pd.Series(pd.Categorical(
        integers, categories=np.arange(21), ordered=True
    ))

    ordinal = int_to_ordinal(integers, n_levels=21)
    integers_reverse = ordinal_to_int(ordinal)

    np.testing.assert_array_equal(integers, integers_reverse)


def testing_encoding():

    encodings = [
        {'ordinal':True, 'column': 't1'},
        {'ordinal':False, 'column': 't2'},
        {'etype': 'interaction','a': {'ordinal':True, 'column': 't1'}, 'b': {'ordinal':False, 'column': 't2'}}
    ]

    t_df = pd.DataFrame({'t1': [0,1,0,1,2], 't2': [0,0,1,1,2]}, dtype='category')



    t_enc = joined_encoder(t_df, encodings)

    enc_a = np.array([[0,0],[1,0],[0,0],[1,0],[1,1]])
    enc_b = np.array([0,0,1,1,2])[:, np.newaxis]

    np.testing.assert_array_equal(t_enc[:,[0,1]], enc_a)
    np.testing.assert_array_equal(
        t_enc[:,[2]],
        enc_b
    )
    np.testing.assert_array_equal(
        t_enc[:,[3,4]],
        enc_a*enc_b
    )