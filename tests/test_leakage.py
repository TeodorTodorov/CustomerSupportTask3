import json

import pandas as pd
import pytest

from klett_classifier.leakage import GroupingSettings, LeakageError, build_leakage_groups
from klett_classifier.splitting import create_splits, validate_split_files
from klett_classifier.labels import CATEGORIES


def no_lexical(texts, threshold):
    return []


def no_semantic(texts, model_name, threshold, k):
    return []


def test_provenance_and_paraphrases_form_one_component():
    frame = pd.DataFrame([
        {'text': 'Where are the lessons?', 'label': CATEGORIES[2], 'pair_id': 'a', 'source':'gen', 'family': 'template-1'},
        {'text': 'Къде са уроците?', 'label': CATEGORIES[2], 'pair_id': 'a', 'source':'gen', 'family': 'template-1'},
        {'text': 'How do I find my lessons?', 'label': CATEGORIES[2], 'pair_id': 'b', 'source':'gen', 'family':'template-2'},
        {'text': 'Where are the mathematics lessons?', 'label': CATEGORIES[2], 'pair_id': 'c', 'source':'gen', 'family':'template-1'},
        {'text': 'Payment invoice please', 'label': CATEGORIES[3], 'pair_id':'d', 'source':'hf', 'family':'single'},
    ])
    frame, audit = build_leakage_groups(
        frame, GroupingSettings(max_component_fraction=1),
        near_fn=no_lexical,
        semantic_fn=lambda *args: [(0, 2, .96)],
    )
    assert frame.iloc[0].leakage_group == frame.iloc[1].leakage_group
    assert frame.iloc[0].leakage_group == frame.iloc[2].leakage_group
    assert frame.iloc[0].leakage_group == frame.iloc[3].leakage_group
    assert audit['cross_label_near_match_count'] == 0


def test_cross_label_near_match_is_not_silently_relabelled():
    data = pd.DataFrame([
        {'text':'Find the video', 'label':CATEGORIES[2]},
        {'text':'The video does not load', 'label':CATEGORIES[0]},
    ])
    _, audit = build_leakage_groups(
        data, GroupingSettings(max_component_fraction=1),
        near_fn=lambda *a: [(0, 1, .92)], semantic_fn=no_semantic,
    )
    assert audit['cross_label_near_match_count'] == 1


def test_three_way_split_and_tamper_guard(tmp_path):
    records=[]
    for category_index, label in enumerate(CATEGORIES):
        for i in range(30):
            for lang in ('en','bg'):
                records.append({'text':f'{lang} item {category_index} {i}', 'label':label,
                                'language':lang,'pair_id':f'p-{category_index}-{i}',
                                'source':'seed', 'family':f'f-{category_index}-{i}'})
    source=tmp_path/'source.csv'
    pd.DataFrame(records).to_csv(source,index=False,encoding='utf-8-sig')
    dest=tmp_path/'splits'
    manifest=create_splits(source,dest,near_fn=no_lexical,semantic_fn=no_semantic,
                           settings=GroupingSettings(max_component_fraction=.4))
    train,val,test,_=validate_split_files(dest)
    for part in (train,val,test):
        assert set(part.label)==set(CATEGORIES)
    pair_sets=[set(x.pair_id) for x in (train,val,test)]
    assert not pair_sets[0]&pair_sets[1]
    assert not pair_sets[0]&pair_sets[2]
    assert not pair_sets[1]&pair_sets[2]
    test.loc[0,'text']='changed text after audit'
    test.to_csv(dest/'test.csv',index=False,encoding='utf-8-sig')
    with pytest.raises(LeakageError, match='changed'):
        validate_split_files(dest)
