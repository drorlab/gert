"""Code for preparing a pairs dataset (filtering and splitting)."""
import numpy as np
import pandas as pd

import click

import atom3d.datasets.ppi.neighbors as nb
import atom3d.filters.pdb
import atom3d.filters.sequence
import atom3d.protein.scop as scop
import atom3d.protein.sequence
import atom3d.protein.sequence as seq
import atom3d.splits.sequence
import atom3d.filters.filters as filters
import atom3d.shard.shard as sh
import atom3d.shard.shard_ops as sho
import atom3d.util.file as fi
import atom3d.util.log as log

logger = log.get_logger('prepare')


def split(input_sharded, output_root, shuffle_buffer):
    """Split by sequence identity."""
    if input_sharded.get_keys() != ['ensemble']:
        raise RuntimeError('Can only apply to sharded by ensemble.')

    all_chain_sequences = []
    logger.info('Loading chain sequences')
    for _, shard in input_sharded.iter_shards():
        all_chain_sequences.extend(seq.get_all_chain_sequences_df(shard))

    logger.info('Splitting by cluster')
    train, val, test = atom3d.splits.sequence.cluster_split(all_chain_sequences, 30)

    # Will just look up ensembles.
    train = [x[0] for x in train]
    val = [x[0] for x in val]
    test = [x[0] for x in test]

    keys = input_sharded.get_keys()
    if keys != ['ensemble']:
        raise RuntimeError('Can only apply to sharded by ensemble.')
    prefix = sh.get_prefix(output_root)
    num_shards = sh.get_num_shards(output_root)
    train_sharded = sh.Sharded(f'{prefix:}_train@{num_shards:}', keys)
    val_sharded = sh.Sharded(f'{prefix:}_val@{num_shards:}', keys)
    test_sharded = sh.Sharded(f'{prefix:}_test@{num_shards:}', keys)

    logger.info('Writing sets')
    train_filter_fn = filters.form_filter_against_list(train, 'ensemble')
    val_filter_fn = filters.form_filter_against_list(val, 'ensemble')
    test_filter_fn = filters.form_filter_against_list(test, 'ensemble')

    sho.filter_sharded(
        input_sharded, train_sharded, train_filter_fn, shuffle_buffer)
    sho.filter_sharded(
        input_sharded, val_sharded, val_filter_fn, shuffle_buffer)
    sho.filter_sharded(
        input_sharded, test_sharded, test_filter_fn, shuffle_buffer)


def form_scop_pair_filter_against(sharded, level):
    """Remove pairs that have matching scop classes in both subunits."""

    scop_index = scop.get_scop_index()[level]

    scop_pairs = []
    for _, shard in sh.iter_shards(sharded):
        for e, ensemble in shard.groupby(['ensemble']):
            names, (bdf0, bdf1, udf0, udf1) = nb.get_subunits(ensemble)
            chains0 = bdf0[['structure', 'chain']].drop_duplicates()
            chains1 = bdf1[['structure', 'chain']].drop_duplicates()
            chains0['pdb_code'] = chains0['structure'].apply(
                lambda x: fi.get_pdb_code(x).lower())
            chains1['pdb_code'] = chains1['structure'].apply(
                lambda x: fi.get_pdb_code(x).lower())

            scop0, scop1 = [], []
            for (pc, c) in chains0[['pdb_code', 'chain']].to_numpy():
                if (pc, c) in scop_index:
                    scop0.append(scop_index.loc[(pc, c)].values)
            for (pc, c) in chains1[['pdb_code', 'chain']].to_numpy():
                if (pc, c) in scop_index:
                    scop1.append(scop_index.loc[(pc, c)].values)
            scop0 = list(np.unique(np.concatenate(scop0))) \
                if len(scop0) > 0 else []
            scop1 = list(np.unique(np.concatenate(scop1))) \
                if len(scop1) > 0 else []
            pairs = [tuple(sorted((a, b))) for a in scop0 for b in scop1]
            scop_pairs.extend(pairs)
    scop_pairs = set(scop_pairs)

    def filter_fn(df):
        to_keep = {}
        for e, ensemble in df.groupby(['ensemble']):
            names, (bdf0, bdf1, udf0, udf1) = nb.get_subunits(ensemble)
            chains0 = bdf0[['structure', 'chain']].drop_duplicates()
            chains1 = bdf1[['structure', 'chain']].drop_duplicates()
            chains0['pdb_code'] = chains0['structure'].apply(
                lambda x: fi.get_pdb_code(x).lower())
            chains1['pdb_code'] = chains1['structure'].apply(
                lambda x: fi.get_pdb_code(x).lower())

            scop0, scop1 = [], []
            for (pc, c) in chains0[['pdb_code', 'chain']].to_numpy():
                if (pc, c) in scop_index:
                    scop0.append(scop_index.loc[(pc, c)].values)
            for (pc, c) in chains1[['pdb_code', 'chain']].to_numpy():
                if (pc, c) in scop_index:
                    scop1.append(scop_index.loc[(pc, c)].values)
            scop0 = list(np.unique(np.concatenate(scop0))) \
                if len(scop0) > 0 else []
            scop1 = list(np.unique(np.concatenate(scop1))) \
                if len(scop1) > 0 else []
            pairs = [tuple(sorted((a, b))) for a in scop0 for b in scop1]

            to_keep[e] = True
            for p in pairs:
                if p in scop_pairs:
                    to_keep[e] = False
        to_keep = pd.Series(to_keep)[df['ensemble']]
        return df[to_keep.values]
    return filter_fn


def form_bsa_filter(bsa_path, min_area):
    """Filter by buried surface area (in Angstroms^2)."""
    bsa = pd.read_csv(bsa_path)
    bsa = bsa.set_index('ensemble')['bsa']

    def filter_fn(df):
        ensembles = df['ensemble'].unique()
        tmp = bsa.loc[ensembles] >= min_area
        to_keep = tmp[df['ensemble']]
        return df[to_keep.values]

    return filter_fn


@click.command(help='Filter pair dataset')
@click.argument('input_sharded_path', type=click.Path())
@click.argument('output_root', type=click.Path())
@click.option('-b', '--bsa', default=None,
              help='File to use for bsa filtering.')
@click.option('--against_path', default=None,
              help='Sharded dataset to filter against (for SCOP and seq)')
@click.option('--shuffle_buffer', type=int, default=10,
              help='How many shards to use in streaming shuffle. 0 means will '
              'not shuffle.')
def filter_pairs(input_sharded_path, output_root, bsa, against_path,
                 shuffle_buffer):
    input_sharded = sh.Sharded.load(input_sharded_path)
    keys = input_sharded.get_keys()
    if keys != ['ensemble']:
        raise RuntimeError('Can only apply to sharded by ensemble.')
    output_sharded = sh.Sharded(output_root, keys)
    # We form the combined filter by starting with the identity filter and
    # composing with further filters.
    filter_fn = filters.identity_filter

    filter_fn = filters.compose(
        atom3d.filters.pdb.form_molecule_type_filter(allowed=['prot']), filter_fn)
    filter_fn = filters.compose(
        filters.form_size_filter(min_size=50), filter_fn)
    filter_fn = filters.compose(
        atom3d.filters.pdb.form_resolution_filter(3.5), filter_fn)
    filter_fn = filters.compose(
        atom3d.filters.pdb.form_source_filter(allowed=['diffraction', 'EM']), filter_fn)
    if bsa is not None:
        filter_fn = filters.compose(
            form_bsa_filter(bsa, 500), filter_fn)
    if against_path is not None:
        against = sh.Sharded.load(against_path)
        filter_fn = filters.compose(
            atom3d.filters.sequence.form_seq_filter_against(against, 0.3), filter_fn)
        filter_fn = filters.compose(
            form_scop_pair_filter_against(against, 'superfamily'), filter_fn)

    sho.filter_sharded(input_sharded, output_sharded, filter_fn)
    split(output_sharded, output_root, shuffle_buffer)


if __name__ == "__main__":
    filter_pairs()
