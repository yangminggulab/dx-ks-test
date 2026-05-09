-- 文件用途：给出实验分流的示例 SQL，用于将用户随机分配到实验组与对照组。

INSERT INTO experiment_assignment (
    user_id,
    experiment_name,
    group_name,
    assigned_at
)
SELECT
    u.user_id,
    'homepage_ranker_v1' AS experiment_name,
    CASE
        WHEN MOD(u.user_id, 2) = 0 THEN 'control'
        ELSE 'treatment'
    END AS group_name,
    NOW() AS assigned_at
FROM user_profile AS u
WHERE u.active_days_30d >= 3;

-- 查看分流结果是否均衡
SELECT
    experiment_name,
    group_name,
    COUNT(*) AS user_cnt
FROM experiment_assignment
WHERE experiment_name = 'homepage_ranker_v1'
GROUP BY experiment_name, group_name
ORDER BY group_name;
