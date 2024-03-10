# 环境
- SPARQL url: README 中要求使用 DKILAB 的端口
- conda 环境: 
    - 新建一个 Python 3.10 的环境
    - 参考 https://github.com/castorini/pyserini/blob/master/docs/installation.md, 安装 pyserini 相关
```
pip install torch==2.0.1 faiss-cpu
pip install pyserini==0.22.0
```
- 安装 requirements.txt 中的其他依赖
    - openai 应该安装 0.27.8 版本
activate 该环境
得通过这个方法来 activate
```
conda activate /home4/xwu/.conda/envs/kb-binder
```

# 代码修改
- select_shot_prompt_train(): 应该从 s_expression 不为 null 的样本中，选择 prompt
- name_to_id_dict 的处理太慢了；debug 阶段能否跳过?
- api_key 参数修改成一个文件，轮换使用 key
- openai 访问: 适配 gpt-3.5-turbo
- 日志补充
- 实验结果记录
    - 对于一个问题，6个 draft 多数投票所选择的那个答案, 作为最终答案
    - 同一个问题下，记录答案到所有（执行结果为这个答案的）lf 的映射
- 参考黄祥学长这边的改动
    - 增加了输出的内容
- process_one_example()
    - 把每个数据集样本的 QA 过程放在这个函数里
    - 目的是可以检查是否超时；作为一个函数，超时的时候直接 return 即可（Python 没有 goto）
# 关键点（坑）记录
- 生成的 S-expression 是旧版格式的, 我们计算等价时做好适配
- 不确定 openai 版本是啥，可能会报错
- 默认的代码设定，好像就实现了 Majority Voting, 应该是 KB-Binder (6)

## 理一下我们所需的数据
对于每个样本会（通过大模型）生成 6 个 gene_exp
对于每个 gene_exp 会获得 n 个可执行 Sexp (n 不固定)

answer: 对于一个样本的所有可执行 Sexp 的执行结果做 Majority Voting 得到的
answer_to_grounded_dict: 在这个样本中，answer 到 可执行 Sexp 的映射

我们要做的: 对于每个样本，收集 answer 和 answer_to_grounded_dict
answer_to_grounded_dict[answer] 我们均视为 KB-BINDER 的输出
- 首先检查执行结果是否都是 answer
- 其次在评价的时候，对于所有可执行 Sexp, 我们取评价最好的那个？

# 代码执行
## KB-BINDER (6) WebQSP 1000
论文中说是 100 shot
```
python3 few_shot_kbqa.py --shot_num 100 --temperature 0.3 \
 --api_key_list_file api_keys/exclusive/openai_keys_2.json --engine gpt-3.5-turbo \
 --train_data_path data/webqsp_0107.train.json --eva_data_path data/webqsp_0107.test.1000.json \
 --fb_roles_path data/fb_roles --surface_map_path data/surface_map_file_freebase_complete_all_mention --timeout_limit 600
```
## KB-BINDER (6) GrailQA 1000
论文中说是 40 shot
```
python3 few_shot_kbqa.py --shot_num 40 --temperature 0.3 \
 --api_key_list_file api_keys/exclusive/openai_keys_1.json --engine gpt-3.5-turbo \
 --train_data_path data/grailqa_v1.0_train.json --eva_data_path data/grailqa_v1.0_dev.1000.json \
 --fb_roles_path data/fb_roles --surface_map_path data/surface_map_file_freebase_complete_all_mention --timeout_limit 600
```

## KB-BINDER (6) CWQ 200
Follow GrailQA 的设置，40 shot
CWQ 有几点需要注意
- topic_entity 和 topic_entity_name

```
python3 few_shot_kbqa.py --shot_num 40 --temperature 0.3 \
 --api_key_list_file api_keys/exclusive/openai_keys_3.json --engine gpt-3.5-turbo \
 --train_data_path data/cwq.train.json  --eva_data_path data/cwq.test.200.json \
 --fb_roles_path data/fb_roles --surface_map_path data/surface_map_file_freebase_complete_all_mention
```