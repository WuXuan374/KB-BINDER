# 环境
- SPARQL url: README 中要求使用 DKILAB 的端口 
    - 由于 SPARQL 端口的负载，投稿论文中，对于 CWQ 和 WebQSP 使用 official 端口，GrailQA 使用 DKILAB 端口
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
- sparql_exe.py
    - SPARQL 查询增加失败重试机制
- 0530, few_shot_kbqa.py
    - issue https://github.com/ltl3A87/KB-BINDER/issues/7 说，运行 WebQSP 时要替换 from_fn_to_id_set 和 convz_fn_to_mids 函数
    - 相应加了个 try-catch, 不然会报错
    - 跳过 WebQSP WebQTest-575, 不知道为啥，会卡死在这
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
- 执行之前检查一下 SPARQL 端口!
- WebQSP 和 CWQ 有两个函数需要替换
## KB-BINDER (6) WebQSP 1000
论文中说是 100 shot; 和论文一致，使用 DKILAB 端口
```
python3 few_shot_kbqa.py --shot_num 100 --temperature 0.3 \
 --api_key_list_file api_keys/exclusive/openai_keys_0.json --engine gpt-3.5-turbo \
 --train_data_path data/webqsp_0107.train.json --eva_data_path data/webqsp_0107.test.1000.json \
 --fb_roles_path data/fb_roles --surface_map_path data/surface_map_file_freebase_complete_all_mention --timeout_limit 600 --checkpoint_size 50
```
## KB-BINDER (6) GrailQA 1000
论文中说是 40 shot
```
python3 few_shot_kbqa.py --shot_num 40 --temperature 0.3 \
 --api_key_list_file api_keys/exclusive/openai_keys_1.json --engine gpt-3.5-turbo \
 --train_data_path data/grailqa_v1.0_train.json --eva_data_path data/grailqa_v1.0_dev.1000.json \
 --fb_roles_path data/fb_roles --surface_map_path data/surface_map_file_freebase_complete_all_mention --timeout_limit 600 --checkpoint_size 50
```

## KB-BINDER (6) CWQ 1000
Follow GrailQA 的设置，40 shot
CWQ 有几点需要注意
- topic_entity 和 topic_entity_name

```
python3 few_shot_kbqa.py --shot_num 40 --temperature 0.3 \
 --api_key_list_file api_keys/exclusive/openai_keys_0.json --engine gpt-3.5-turbo \
 --train_data_path data/cwq.train.json  --eva_data_path data/cwq.test.1000.json \
 --fb_roles_path data/fb_roles --surface_map_path data/surface_map_file_freebase_complete_all_mention --timeout_limit 600 --checkpoint_size 50
```

# 投稿论文代码执行
Follow GrailQA 的设置，40 shot

## KB-BINDER (6) CWQ 1000
使用的 key sk-ROoV4l0NELy39vGl45Fe5bDc7a954224A717BeFe4eCf10Ba
SPARQL 端口 http://210.28.134.34:8890/sparql/

```
python3 few_shot_kbqa.py --shot_num 40 --temperature 0.3 \
 --engine gpt-3.5-turbo-0613 \
 --train_data_path data/cwq.train.json  --eva_data_path data/paper/cwq.test.1000.json \
 --fb_roles_path data/fb_roles --surface_map_path data/surface_map_file_freebase_complete_all_mention --timeout_limit 600 --checkpoint_size 50
```
由于发现缺少 log, 上面的程序运行完了前 100 个就先终止了，接下来运行 100-1000 --> 实际上只运行了 100-150

```
python3 few_shot_kbqa.py --shot_num 40 --temperature 0.3 \
 --engine gpt-3.5-turbo-0613 \
 --train_data_path data/cwq.train.json  --eva_data_path data/paper/cwq.test.100_1000.json \
 --fb_roles_path data/fb_roles --surface_map_path data/surface_map_file_freebase_complete_all_mention --timeout_limit 600 --checkpoint_size 50
```

... 接下来运行 150-1000
```
python3 few_shot_kbqa.py --shot_num 40 --temperature 0.3 \
 --engine gpt-3.5-turbo-0613 \
 --train_data_path data/cwq.train.json  --eva_data_path data/paper/cwq.test.150_1000.json \
 --fb_roles_path data/fb_roles --surface_map_path data/surface_map_file_freebase_complete_all_mention --timeout_limit 600 --checkpoint_size 50
```

## KB-BINDER (6) WebQSP 1000
使用的 key sk-ROoV4l0NELy39vGl45Fe5bDc7a954224A717BeFe4eCf10Ba
论文中说是 100 shot; 投稿版本使用 OFFICIAL 端口

0-300 --> 实际上只运行了 0-200
```
python3 few_shot_kbqa.py --shot_num 100 --temperature 0.3 \
 --engine gpt-3.5-turbo-0613 \
 --train_data_path data/webqsp_0107.train.json --eva_data_path data/paper/webqsp_0107.test.0_300.json \
 --fb_roles_path data/fb_roles --surface_map_path data/surface_map_file_freebase_complete_all_mention --timeout_limit 600 --checkpoint_size 50
```

0-50, 但是替换了 WebQSP 的两个函数，看看效果是否发生变化 --> 替换这两个函数非常有必要
```
python3 few_shot_kbqa.py --shot_num 100 --temperature 0.3 \
 --engine gpt-3.5-turbo-0613 \
 --train_data_path data/webqsp_0107.train.json --eva_data_path data/paper/webqsp_0107.test.0_50.json \
 --fb_roles_path data/fb_roles --surface_map_path data/surface_map_file_freebase_complete_all_mention --timeout_limit 600 --checkpoint_size 50
```

50-200，使用的 key 是 sk-MULDnyMuh4HH4WcJ7fC6F2923b8246A9AbF205D1D9Bd50C6
```
python3 few_shot_kbqa.py --shot_num 100 --temperature 0.3 \
 --engine gpt-3.5-turbo-0613 \
 --train_data_path data/webqsp_0107.train.json --eva_data_path data/paper/webqsp_0107.test.50_200.json \
 --fb_roles_path data/fb_roles --surface_map_path data/surface_map_file_freebase_complete_all_mention --timeout_limit 600 --checkpoint_size 50
```

200-250，使用的 key 是 sk-MULDnyMuh4HH4WcJ7fC6F2923b8246A9AbF205D1D9Bd50C6
```
python3 few_shot_kbqa.py --shot_num 100 --temperature 0.3 \
 --engine gpt-3.5-turbo-0613 \
 --train_data_path data/webqsp_0107.train.json --eva_data_path data/paper/webqsp_0107.test.200_400.json \
 --fb_roles_path data/fb_roles --surface_map_path data/surface_map_file_freebase_complete_all_mention --timeout_limit 600 --checkpoint_size 50
```

250-349，使用的 key 是 sk-MULDnyMuh4HH4WcJ7fC6F2923b8246A9AbF205D1D9Bd50C6
```
python3 few_shot_kbqa.py --shot_num 100 --temperature 0.3 \
 --engine gpt-3.5-turbo-0613 \
 --train_data_path data/webqsp_0107.train.json --eva_data_path data/paper/webqsp_0107.test.250_400.json \
 --fb_roles_path data/fb_roles --surface_map_path data/surface_map_file_freebase_complete_all_mention --timeout_limit 600 --checkpoint_size 50
```

349-500 使用的 key 是 sk-1c8bIaeuur9kAvsi04355047B3C443DaAd9642C80a48Cb51
```
python3 few_shot_kbqa.py --shot_num 100 --temperature 0.3 \
 --engine gpt-3.5-turbo-0613 \
 --train_data_path data/webqsp_0107.train.json --eva_data_path data/paper/webqsp_0107.test.349_500.json \
 --fb_roles_path data/fb_roles --surface_map_path data/surface_map_file_freebase_complete_all_mention --timeout_limit 600 --checkpoint_size 50
```

500-800 使用的 key 是 sk-1c8bIaeuur9kAvsi04355047B3C443DaAd9642C80a48Cb51
```
python3 few_shot_kbqa.py --shot_num 100 --temperature 0.3 \
 --engine gpt-3.5-turbo-0613 \
 --train_data_path data/webqsp_0107.train.json --eva_data_path data/paper/webqsp_0107.test.500_800.json \
 --fb_roles_path data/fb_roles --surface_map_path data/surface_map_file_freebase_complete_all_mention --timeout_limit 600 --checkpoint_size 50
```

## KB-BINDER (6) CWQ 1000 重跑
使用的 key sk-MULDnyMuh4HH4WcJ7fC6F2923b8246A9AbF205D1D9Bd50C6
按照 issue 中的说法，替换了函数，看看效果是否有变化
SPARQL 端口 http://210.28.134.34:8890/sparql/

0_50
```
python3 few_shot_kbqa.py --shot_num 40 --temperature 0.3 \
 --engine gpt-3.5-turbo-0613 \
 --train_data_path data/cwq.train.json  --eva_data_path data/paper/cwq.test.0_50.json \
 --fb_roles_path data/fb_roles --surface_map_path data/surface_map_file_freebase_complete_all_mention --timeout_limit 600 --checkpoint_size 50
```

## debug

```
python3 few_shot_kbqa.py --shot_num 40 --temperature 0.3 \
 --engine gpt-3.5-turbo-0613 \
 --train_data_path data/cwq.train.json  --eva_data_path data/paper/cwq.test.1000.json \
 --fb_roles_path data/fb_roles --surface_map_path data/surface_map_file_freebase_complete_all_mention --timeout_limit 600 --checkpoint_size 50
 --do_debug true (这个参数好像没生效？)
```