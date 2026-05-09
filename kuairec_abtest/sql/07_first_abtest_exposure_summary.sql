-- 文件用途：统计第一版 AB Test 的曝光级完播列联表，供 Python 侧进行卡方检验。

SELECT
    CASE
        WHEN MOD(CRC32(CAST(user_id AS CHAR)), 2) = 0 THEN 'control'
        ELSE 'treatment'
    END AS group_name,
    SUM(
        CASE
            WHEN play_duration > 0
             AND play_duration >= video_duration
             AND video_duration > 0 THEN 1
            ELSE 0
        END
    ) AS complete_play_cnt,
    SUM(
        CASE
            WHEN play_duration > 0
             AND NOT (play_duration >= video_duration AND video_duration > 0) THEN 1
            ELSE 0
        END
    ) AS incomplete_play_cnt
FROM kuairec_small_matrix
GROUP BY group_name
ORDER BY group_name;
