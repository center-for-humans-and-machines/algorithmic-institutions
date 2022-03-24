from aimanager.generic.encoder import int_to_ordinal, ordinal_to_int, int_to_onehot, onehot_to_int, joined_encoder
import numpy as np
import pandas as pd


def test_int_to_ordinal():
    integers = np.array([0,3,2])
    ordinal = int_to_ordinal(integers, n_levels=4)

    expected = np.array([
        [0,0,0],
        [1,1,1],
        [1,1,0]
    ])

    np.testing.assert_array_equal(ordinal, expected)


def test_ordinal_to_int():
    ordinal = np.array([
        [0.3,0.2,0.6],
        [0.8,0.9,0.6],
        [0.6,0.7,0.4]
    ])
    expected = np.array([0,3,2])
    integers = ordinal_to_int(ordinal)

    np.testing.assert_array_equal(integers, expected)



def test_ordinal_conversion():
    # testing the conversion

    integers = np.random.randint(0, 21, 30)

    ordinal = int_to_ordinal(integers, n_levels=21)
    integers_reverse = ordinal_to_int(ordinal)

    np.testing.assert_array_equal(integers, integers_reverse)



def test_onehot_conversion():
    # testing the conversion

    integers = np.random.randint(0, 21, 30)

    onehot = int_to_onehot(integers, n_levels=21)

    assert (onehot.sum(-1) == 1).all()

    integers_reverse = onehot_to_int(onehot)

    np.testing.assert_array_equal(integers, integers_reverse)



def testing_encoding():

    encodings = [
        {'encoding': 'ordinal', 'column': 't1'},
        {'encoding': 'numeric', 'column': 't2'},
        {'etype': 'interaction','a': {'encoding': 'ordinal', 'column': 't1'}, 'b': {'encoding': 'numeric', 'column': 't2'}}
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