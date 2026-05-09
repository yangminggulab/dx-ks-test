-- 文件用途：统计实验组与对照组的核心指标，例如曝光人数、播放人数、完播率与平均观看时长。

WITH base_data AS (
    SELECT
        experiment_name,
        group_name,
        user_id,
        video_id,
        is_play,
        is_complete,
        watch_time_seconds
    FROM video_exposure_log
    WHERE experiment_name = 'homepage_ranker_v1'
)
SELECT
    experiment_name,
    group_name,
    COUNT(*) AS exposure_pv,
    COUNT(DISTINCT user_id) AS exposure_uv,
    SUM(is_play) AS play_pv,
    SUM(is_complete) AS complete_pv,
    ROUND(SUM(is_play) / NULLIF(COUNT(*), 0), 4) AS play_rate,
    ROUND(SUM(is_complete) / NULLIF(SUM(is_play), 0), 4) AS completion_rate,
    ROUND(AVG(watch_time_seconds), 2) AS avg_watch_time
FROM base_data
GROUP BY experiment_name, group_name
ORDER BY group_name;
