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
# 关键点（坑）记录
- 生成的 S-expression 是旧版格式的, 我们计算等价时做好适配
- 不确定 openai 版本是啥，可能会报错
- 默认的代码设定，好像就实现了 Majority Voting, 应该是 KB-Binder (6)

# 代码修改

# 代码执行
## KB-BINDER (6) WebQSP
```
python3 few_shot_kbqa.py --shot_num 40 --temperature 0.3 \
 --api_key_list_file api_keys/shared/openai_keys.json --engine gpt-3.5-turbo \
 --train_data_path data/webqsp_0107.train.json --eva_data_path data/webqsp_0107.test.json \
 --fb_roles_path data/fb_roles --surface_map_path data/surface_map_file_freebase_complete_all_mention
```