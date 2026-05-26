# 待办事项

## 立即（明天 09:00 机器自动唤醒后）

> 连接方式：`ssh win-local`（局域网）或 `ssh win`（外网 Tailscale）

### ① 一次性修复：禁用 idle 自动睡眠（根本问题）

机器当前会在唤醒后 ~19 分钟因无人操作触发 idle sleep 重新入睡，导致 SSH 断连、训练无法启动。
原因：`powercfg` 的空闲睡眠计时器未关闭，与盖子/休眠设置无关。

SSH 进去后在管理员 PowerShell 里跑一次（**永久生效，重启不丢失**）：

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0
```

之后机器只会在 `DailySleep_2130` 任务（21:30）主动触发时才睡眠，不再自动掉线。

### ② 同步代码 + 启动训练

```bash
cd /Users/liubike/Desktop/快手test/kuairec_abtest/scripts
KUAIREC_SERVER="thisi@192.168.1.18" bash sync_and_run.sh
```

### ③ 训练结束后

- [ ] 查看 CL4SRec 日志：找最后一个 ★ 出现在第几 epoch，判断 early stopping 是否太早触发

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
