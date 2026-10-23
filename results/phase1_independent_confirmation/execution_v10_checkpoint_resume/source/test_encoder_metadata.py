import copy,json,pytest
from encoder_metadata import matches

def test_live_integer_label_keys_match_frozen_json_without_changing_values():
    live={'model_config':{'id2label':{0:'LABEL_0',1:'LABEL_1'}},'dtype':'float32','revision':'fixed','processor':{'size':[224,224]}}
    frozen=json.loads(json.dumps(live))
    assert live!=frozen and matches(live,frozen)
    for field,value in [('dtype','float16'),('revision','changed'),('processor',{'size':[96,96]})]:
        altered=copy.deepcopy(live);altered[field]=value;assert not matches(altered,frozen)
    live['model_config']['id2label'][0]='wrong';assert not matches(live,frozen)

def test_nonfinite_metadata_is_rejected():
    with pytest.raises(ValueError):matches({'x':float('nan')},{'x':None})
