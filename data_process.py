import json

def load_json(fname, mode="r", encoding="utf8"):
    if "b" in mode:
        encoding = None
    with open(fname, mode=mode, encoding=encoding) as f:
        return json.load(f)

def dump_json(obj, fname, indent=4, mode='w' ,encoding="utf8", ensure_ascii=False):
    """
    @param: ensure_ascii: `False`, 字符原样输出；`True`: 对于非 ASCII 字符进行转义
    """
    if "b" in mode:
        encoding = None
    with open(fname, "w", encoding=encoding) as f:
        return json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)

def get_webqsp_subset():
    original_data = load_json('data/webqsp_0107.test.json')
    subset_data = load_json('data/webqsp_test_0_200_linking.json')
    qid_list = [item['qid'] for item in subset_data]
    selected_examples = [item for item in original_data if item["qid"] in qid_list]
    assert len(selected_examples) == len(subset_data)
    dump_json(selected_examples, 'data/webqsp_0107.test.200.json')

def get_grailqa_subset():
    original_data = load_json('data/grailqa_v1.0_dev.json')
    subset_data = load_json('data/grailqa_v1.0_dev_0_200_linking.json')
    qid_list = [item['qid'] for item in subset_data]
    selected_examples = [item for item in original_data if item["qid"] in qid_list]
    assert len(selected_examples) == len(subset_data)
    dump_json(selected_examples, 'data/grailqa_v1.0_dev.200.json')

def get_cwq_subset():
    original_data = load_json('data/cwq_test_0_200_linking.json')
    new_data = list()
    for item in original_data:
        new_data.append({
            "id": item["qid"],
            "question": item["question"],
            "sparql_query": item["golden_sparql_query"],
            "s_expression": item["golden_s_expression"],
            "answer": [{"answer_argument": ans['mid']} for ans in item['answer']]
        })
    dump_json(new_data, 'data/cwq.test.200.json')

if __name__=='__main__':
    # get_webqsp_subset()
    # get_grailqa_subset()
    get_cwq_subset()