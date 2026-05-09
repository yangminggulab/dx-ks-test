-- 文件用途：基于 KuaiRec small_matrix 构建第一版 AB Test 的用户级指标表，供 Python 侧做 t-test 与分层分析。

WITH user_metrics AS (
    SELECT
        CASE
            WHEN MOD(CRC32(CAST(s.user_id AS CHAR)), 2) = 0 THEN 'control'
            ELSE 'treatment'
        END AS group_name,
        s.user_id,
        SUM(CASE WHEN s.play_duration > 0 THEN 1 ELSE 0 END) AS play_cnt,
        SUM(
            CASE
                WHEN s.play_duration >= s.video_duration AND s.video_duration > 0 THEN 1
                ELSE 0
            END
        ) AS complete_cnt,
        AVG(LEAST(GREATEST(s.watch_ratio, 0), 1)) AS avg_watch_ratio,
        AVG(s.play_duration) AS avg_play_duration,
        MAX(u.user_active_degree) AS user_active_degree
    FROM kuairec_small_matrix AS s
    LEFT JOIN kuairec_user_features AS u
        ON s.user_id = u.user_id
    GROUP BY group_name, s.user_id
)
SELECT
    group_name,
    user_id,
    user_active_degree,
    play_cnt,
    complete_cnt,
    complete_cnt / NULLIF(play_cnt, 0) AS completion_rate,
    avg_watch_ratio,
    avg_play_duration
FROM user_metrics
WHERE play_cnt > 0;
