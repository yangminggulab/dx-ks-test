-- 文件用途：创建 AB Test 分析所需的示例业务表结构，便于后续进行实验分流、指标统计和效果评估。

CREATE TABLE IF NOT EXISTS user_profile (
    user_id BIGINT PRIMARY KEY,
    gender VARCHAR(16),
    age INT,
    city_level VARCHAR(32),
    register_date DATE,
    active_days_30d INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS experiment_assignment (
    user_id BIGINT PRIMARY KEY,
    experiment_name VARCHAR(64) NOT NULL,
    group_name VARCHAR(32) NOT NULL,
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS video_exposure_log (
    event_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id BIGINT NOT NULL,
    video_id BIGINT NOT NULL,
    experiment_name VARCHAR(64) NOT NULL,
    group_name VARCHAR(32) NOT NULL,
    exposure_time DATETIME NOT NULL,
    is_play TINYINT DEFAULT 0,
    is_complete TINYINT DEFAULT 0,
    watch_time_seconds DECIMAL(10, 2) DEFAULT 0,
    like_cnt TINYINT DEFAULT 0,
    comment_cnt TINYINT DEFAULT 0,
    share_cnt TINYINT DEFAULT 0,
    INDEX idx_exp_group_time (experiment_name, group_name, exposure_time),
    INDEX idx_user_id (user_id),
    INDEX idx_video_id (video_id)
);
