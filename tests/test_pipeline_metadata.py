import importlib.util
from pathlib import Path

import pandas as pd


def load_script(path):
    spec = importlib.util.spec_from_file_location('builder_for_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_translation_pipeline_retains_family_metadata(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'datasets_processing' / 'build_multilingual_dataset.py'
    module = load_script(path)
    src = tmp_path / 'source_english.csv'
    pd.DataFrame({'text':['Where is the lesson?', 'Why is the video broken?'],
                  'label':['Учебно съдържание','Технически проблем']}).to_csv(src,index=False)
    pd.DataFrame({'text':['Where is the lesson?', 'Why is the video broken?'],
                  'label':['Учебно съдържание','Технически проблем'],
                  'source':['synthetic-content','synthetic-tech'],
                  'family':['content:0','tech:0'],
                  'kind':['template','template']}).to_csv(tmp_path/'source_english_provenance.csv',index=False)
    english = module.read_english(src)
    assert list(english.family) == ['content:0','tech:0']
    english['bg_text'] = ['Къде е урокът?', 'Защо видеото е повредено?']
    module.make_outputs(english,tmp_path/'output',42)
    combined = pd.read_csv(tmp_path/'output'/'multilingual_with_pair_ids.csv')
    assert {'family','source','pair_id','language','text','label'}.issubset(combined.columns)
    assert combined.groupby('pair_id').family.nunique().max() == 1
