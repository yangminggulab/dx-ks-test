"""
文件用途：提供 MySQL 数据库连接配置模板，并支持基础连通性测试。
"""

from __future__ import annotations

import re

from sqlalchemy.engine import URL
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "kuairec"
DB_USER = "root"
DB_PASSWORD = "@Lbk13121755130"
DB_CHARSET = "utf8mb4"


def build_connection_url(
    user: str = DB_USER,
    password: str = DB_PASSWORD,
    host: str = DB_HOST,
    port: int = DB_PORT,
    database: str | None = DB_NAME,
    charset: str = DB_CHARSET,
    hide_password: bool = False,
) -> str:
    """构建数据库连接字符串。"""
    connection_url = URL.create(
        drivername="mysql+pymysql",
        username=user or None,
        password=password or None,
        host=host,
        port=port,
        database=database or None,
        query={"charset": charset},
    )
    return connection_url.render_as_string(hide_password=hide_password)


def get_engine(
    user: str = DB_USER,
    password: str = DB_PASSWORD,
    host: str = DB_HOST,
    port: int = DB_PORT,
    database: str | None = DB_NAME,
):
    """创建 SQLAlchemy Engine。"""
    missing_fields: list[str] = []
    if not user:
        missing_fields.append("DB_USER")
    if not password:
        missing_fields.append("DB_PASSWORD")
    if missing_fields:
        raise ValueError(
            f"请先在 db_config.py 中填写 {', '.join(missing_fields)}。"
        )

    connection_url = build_connection_url(
        user=user,
        password=password,
        host=host,
        port=port,
        database=database,
    )
    return create_engine(connection_url, pool_pre_ping=True, future=True)


def ensure_database_exists(
    database: str = DB_NAME,
    charset: str = DB_CHARSET,
) -> bool:
    """确保目标数据库存在。"""
    try:
        if not re.fullmatch(r"[A-Za-z0-9_]+", database):
            raise ValueError("数据库名称仅允许字母、数字和下划线。")

        engine = get_engine(database=None)
        with engine.begin() as connection:
            connection.execute(
                text(
                    f"CREATE DATABASE IF NOT EXISTS `{database}` "
                    f"CHARACTER SET {charset}"
                )
            )
        print(f"数据库已就绪：{database}")
        return True
    except SQLAlchemyError as exc:
        print(f"数据库初始化失败，原因：{exc}")
        return False
    except Exception as exc:
        print(f"数据库初始化失败，原因：{exc}")
        return False


def test_connection() -> bool:
    """测试数据库连接并打印结果。"""
    try:
        engine = get_engine()
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        print("连接成功")
        return True
    except SQLAlchemyError as exc:
        print(f"连接失败，原因：{exc}")
        return False
    except Exception as exc:
        print(f"连接失败，原因：{exc}")
        return False


if __name__ == "__main__":
    print("数据库连接字符串模板：")
    print(build_connection_url(hide_password=True))
    test_connection()
