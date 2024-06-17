import json
import math
import os

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

def split_cwq():
    original_data = load_json('data/paper/cwq.test.1000.json')
    split_data = original_data[:50]
    dump_json(split_data, 'data/paper/cwq.test.0_50.json')

def split_webqsp():
    original_data = load_json('data/paper/webqsp_0107.test.1000.json')
    split_data = original_data[800:]
    dump_json(split_data, 'data/paper/webqsp_0107.test.800_1000.json')

def split_grailqa():
    original_data = load_json('data/paper/grailqa_v1.0_dev.1000.json')
    split_data = original_data[:300]
    dump_json(split_data, 'data/paper/grailqa_v1.0_dev.0_300.json')

def calculate_result(data_file):
    data = load_json(data_file)
    f1_list = list()
    time_list = list()
    for item in data:
        predicted_answer = item["predicted_answer"]
        if predicted_answer is None:
            predicted_answer = set()
        p, r, f1 = get_PRF1(predicted_answer, item["gold_answer"])
        f1_list.append(f1)
        time_list.append(item["exection_time"])
    print(f"average F1: {sum(f1_list) / len(f1_list)}")
    print(f"Answer sEM: {len([_f1 for _f1 in f1_list if math.isclose(_f1, 1.0)]) / len(f1_list)}")
    print(f"average time: {sum(time_list) / len(time_list)}")

def get_PRF1(pred_answer, golden_answer):
    pred_answer = set(pred_answer)
    golden_answer = set(golden_answer)
    if len(pred_answer)== 0:
        if len(golden_answer)==0:
            p=1
            r=1
            f=1
        else:
            p=0
            r=0
            f=0
    elif len(golden_answer)==0:
        p=0
        r=0
        f=0
    else:
        p = len(pred_answer & golden_answer)/ len(pred_answer)
        r = len(pred_answer & golden_answer)/ len(golden_answer)
        f = 2*(p*r)/(p+r) if p+r>0 else 0
    return p, r, f

def update_answer_patch():
    # 代码 Bug 导致该分片记录的答案都是下一个样本的（详见对应目录的 README）；更新一下
    src_data = load_json('data/paper/webqsp_0107.test.1000.json')
    qid_to_answer = {
        item['qid']: [ans["answer_argument"] for ans in item["answer"]]
        for item in src_data
    }
    target_dir = "output/webqsp_0107.test.250_349.json_2024-06-01_11:18:52/_tmp"
    updated_combined_result = list()
    for tmp_idx in [0, 1]:
        tmp_res = load_json(os.path.join(target_dir, f'{tmp_idx}.json'))
        for _item in tmp_res:
            qid = _item["qid"]
            _item["gold_answer"] = qid_to_answer[qid]
        updated_combined_result.extend(tmp_res)
    dump_json(updated_combined_result, "output/webqsp_0107.test.250_349.json_2024-06-01_11:18:52/results.json")

def combine_detection_result(parent_dir):
    assert os.path.isdir(parent_dir)
    combined_results = list()
    for sub_dir in os.listdir(parent_dir):
        sub_result = load_json(os.path.join(parent_dir, sub_dir, 'results.json'))
        combined_results.extend(sub_result)
    unique_keys = set([
        item["qid"] for item in combined_results
    ])
    print(f"unique_keys: {len(unique_keys)}")
    dump_json(combined_results, os.path.join(parent_dir, 'results.json'))

def combine_tmp_result(parent_dir):
    assert os.path.isdir(parent_dir)
    combined_results = list()
    for tmp_file in os.listdir(os.path.join(parent_dir, "_tmp", "")):
        sub_result = load_json(os.path.join(parent_dir, "_tmp", tmp_file))
        combined_results.extend(sub_result)
    unique_keys = set([
        item["qid"] for item in combined_results
    ])
    print(f"unique_keys: {len(unique_keys)}")
    dump_json(combined_results, os.path.join(parent_dir, 'results.json'))


if __name__=='__main__':
    # get_webqsp_subset()
    # get_grailqa_subset()
    # get_cwq_subset()
    # split_cwq()
    # split_webqsp()
    # split_grailqa()
    calculate_result('output/paper/webqsp_test_0_1000/results.json')
    # update_answer_patch()
    # combine_detection_result("output/paper/webqsp_test_0_1000")
    # combine_tmp_result("output/paper/webqsp_test_0_1000/webqsp_0107.test.200_250.json_2024-05-31_15:13:19")