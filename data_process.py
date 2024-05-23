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
    subset_data = load_json('data/paper/subset/webqsp_test_0_1000_linking.json')
    qid_list = [item['qid'] for item in subset_data]
    selected_examples = [item for item in original_data if item["qid"] in qid_list]
    assert len(selected_examples) == len(subset_data)
    dump_json(selected_examples, 'data/paper/webqsp_0107.test.1000.json')

def get_grailqa_subset():
    original_data = load_json('data/grailqa_v1.0_dev.json')
    subset_data = load_json('data/paper/subset/grailqa_v1.0_dev_0_1000_linking.json')
    qid_list = [item['qid'] for item in subset_data]
    selected_examples = [item for item in original_data if item["qid"] in qid_list]
    assert len(selected_examples) == len(subset_data)
    dump_json(selected_examples, 'data/paper/grailqa_v1.0_dev.1000.json')

def get_cwq_subset():
    original_data = load_json('data/paper/subset/cwq_test_0_1000_linking.json')
    new_data = list()
    for item in original_data:
        nodes = list()
        for grounded_item in item['golden_grounded_items']:
            if grounded_item['type'] == 'entity':
                mid = grounded_item['mid']
                label = item['golden_item_to_label'][mid]
                nodes.append({
                    "node_type": "entity",
                    "id": mid,
                    "friendly_name": label
                })
        if len(nodes):
            new_data.append({
                "qid": item["qid"],
                "question": item["question"],
                "sparql_query": item["golden_sparql_query"],
                "s_expression": item["golden_s_expression"],
                "answer": [{"answer_argument": ans['mid']} for ans in item['answer']],
                "graph_query": {"nodes": nodes}
            })
        else:
            new_data.append({
                "qid": item["qid"],
                "question": item["question"],
                "sparql_query": item["golden_sparql_query"],
                "s_expression": item["golden_s_expression"],
                "answer": [{"answer_argument": ans['mid']} for ans in item['answer']],
                "graph_query": {"nodes": nodes},
                "topic_entity": 'null',
                "topic_entity_name": 'null' # 保证代码能运行，同时不造成什么影响
            })
    dump_json(new_data, 'data/paper/cwq.test.1000.json')

if __name__=='__main__':
    # get_webqsp_subset()
    get_grailqa_subset()
    # get_cwq_subset()