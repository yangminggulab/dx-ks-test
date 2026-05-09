-- 文件用途：在正式 AB Test 前执行 A/A Test，验证实验分流系统是否稳定、组间指标是否无系统性差异。

WITH aa_assignment AS (
    SELECT
        user_id,
        CASE
            WHEN MOD(user_id, 2) = 0 THEN 'aa_group_a'
            ELSE 'aa_group_b'
        END AS aa_group
    FROM experiment_assignment
    WHERE experiment_name = 'homepage_ranker_v1'
      AND group_name = 'control'
),
aa_metrics AS (
    SELECT
        a.aa_group,
        l.user_id,
        l.is_play,
        l.is_complete,
        l.watch_time_seconds
    FROM aa_assignment AS a
    INNER JOIN video_exposure_log AS l
        ON a.user_id = l.user_id
    WHERE l.experiment_name = 'homepage_ranker_v1'
)
SELECT
    aa_group,
    COUNT(*) AS exposure_pv,
    SUM(is_play) AS play_pv,
    SUM(is_complete) AS complete_pv,
    ROUND(SUM(is_complete) / NULLIF(SUM(is_play), 0), 4) AS completion_rate,
    ROUND(AVG(watch_time_seconds), 2) AS avg_watch_time
FROM aa_metrics
GROUP BY aa_group
ORDER BY aa_group;
