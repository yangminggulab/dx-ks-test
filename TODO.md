# 待办事项

## 立即（下次开机后）

- [ ] 开 Windows GPU 服务器，运行 `KUAIREC_SERVER="thisislbk@192.168.1.18" bash sync_and_run.sh` 同步最新代码并启动全量训练
- [ ] 训练结束后查看 CL4SRec 日志：找最后一个 ★ 出现在第几 epoch，判断 early stopping 是否太早触发

## 代码质量（中优先级）

- [ ] `eval_recommenders.py`：返回的 `n_users` 字段包含空推荐用户，与实际参与统计的用户数不一致，建议改为 `len(hit_rates)`
- [ ] `svd_recommender.py:264`：`compute_rmse` 里第一行 `pred_vals` 赋值是死代码，被第二次赋值立刻覆盖，删掉即可

## 低优先级

- [ ] 各模型的 `_get_device()`、`_build_user_sequences()`、checkpoint 保存/恢复逻辑高度重复，可抽到共享工具模块
- [ ] `bert4rec.py`：结果字典字段名 `"bpr_loss"` 实际是 CrossEntropy loss，名字误导，改为 `"ce_loss"` 更准确
