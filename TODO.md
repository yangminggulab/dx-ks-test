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

---

## 本项目完成后：A/B Test 作品集

> 题目方向：短视频推荐策略 A/B Test 分析——从随机偏差到业务显著性判断

- [ ] 第一层：模拟实验——用 numpy 模拟 p_A=0.10 vs p_B 的二项分布，改变样本量和效应大小，验证"样本越大越能检出小差异"
- [ ] 第二层：Kaggle Marketing A/B Testing 数据集——广告组 vs PSA 组转化率，Z 检验 + 置信区间 + 业务结论
- [ ] 第三层：Udacity A/B Testing 项目——完整实验设计报告（sanity check、不变指标、统计+业务双重显著性、上线建议）
