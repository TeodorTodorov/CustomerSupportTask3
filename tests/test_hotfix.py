"""Regression tests for long-running semantic audit and opt-out behavior."""
import pandas as pd
import pytest

from klett_classifier.cli import build_parser
from klett_classifier.labels import CATEGORIES
from klett_classifier.leakage import GroupingSettings, LeakageError, build_leakage_groups
from klett_classifier.splitting import create_splits, validate_split_files


def test_semantic_not_invoked_if_provenance_already_oversized():
    frame = pd.DataFrame([
        {'text': f'Example {i}', 'label': CATEGORIES[i % 4],
         'pair_id': str(i), 'source': 'templates', 'family': 'same' if i < 5 else str(i)}
        for i in range(10)
    ])
    def should_not_call(*args):
        raise AssertionError('Semantic model should not load')
    with pytest.raises(LeakageError, match='after provenance: 5/10'):
        build_leakage_groups(frame, GroupingSettings(max_component_fraction=.2),
                             near_fn=should_not_call, semantic_fn=should_not_call)


def test_semantic_not_invoked_if_lexical_already_oversized():
    frame = pd.DataFrame([
        {'text': f'Example {i}', 'label': CATEGORIES[i % 4],
         'pair_id': str(i), 'source': 'templates', 'family': str(i)}
        for i in range(10)
    ])
    def should_not_call(*args):
        raise AssertionError('Semantic model should not load')
    with pytest.raises(LeakageError, match='after lexical: 4/10'):
        build_leakage_groups(frame, GroupingSettings(max_component_fraction=.2),
                             near_fn=lambda *args: [(0, 1, .99), (1, 2, .99), (2, 3, .99)],
                             semantic_fn=should_not_call)


def test_cli_opt_out_flags_parse():
    p = build_parser()
    args = p.parse_args(['split-data', '--data', 'x.csv', '--skip-semantic'])
    assert args.skip_semantic and not args.provenance_only
    args = p.parse_args(['split-data', '--data', 'x.csv', '--provenance-only'])
    assert args.provenance_only


def test_provenance_only_split_reports_disabled_checks(tmp_path):
    rows = []
    for label_idx, label in enumerate(CATEGORIES):
        for i in range(30):
            for lang in ('en', 'bg'):
                rows.append({'text': f'{lang} item {label_idx} number {i}', 'label': label,
                             'pair_id': f'{label_idx}-{i}', 'source': 'synthetic',
                             'family': f'family-{label_idx}-{i}', 'language': lang})
    data = tmp_path / 'source.csv'
    pd.DataFrame(rows).to_csv(data, index=False)
    output = tmp_path / 'splits'
    create_splits(data, output, near_fn=lambda *args: [],
                  semantic_fn=lambda *args: [],
                  checks={'lexical': False, 'semantic': False},
                  settings=GroupingSettings(max_component_fraction=.3))
    train, val, test, manifest = validate_split_files(output)
    assert manifest['similarity_checks_enabled'] == {'lexical': False, 'semantic': False}
    assert manifest['warning']
    assert set(train.pair_id).isdisjoint(val.pair_id)
    assert set(train.pair_id).isdisjoint(test.pair_id)
