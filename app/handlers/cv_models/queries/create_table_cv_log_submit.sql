create database if not exists CV_MODELS;
create table if not exists CV_MODELS.cv_log_submit
(
    datetime       Datetime,
    submit_id      UUID,
    seconds        Int64,
    number_frame   Int64,
    type_violation Int64,
    name_file      String
)
    engine = MergeTree ORDER BY (submit_id, seconds)
        SETTINGS index_granularity = 8192;

