# 环境
- 使用 agent-bench 环境即可
- SPARQL url: README 中要求使用 DKILAB 的端口
# 代码简单阅读
- 
# 关键点（坑）记录
- 生成的 S-expression 是旧版格式的
- 不确定 openai 版本是啥，可能会报错
- 默认的代码设定，好像就实现了 Majority Voting, 应该是 KB-Binder (6)

# 代码修改
- api_key 参数修改成一个文件，轮换使用 key
- 执行结果输出

# 代码执行
## KB-BINDER (6)
```
python3 few_shot_kbqa.py --shot_num 40 --temperature 0.3 \
 --api_key_list_file api_keys/shared/openai_keys.json --engine gpt-3.5-turbo \
 --train_data_path webqsp_0107.train.json --eva_data_path webqsp_0107.test.json \
 --fb_roles_path data/fb_roles --surface_map_path data/surface_map_file_freebase_complete_all_mention
```