"""MinIO 客户端（懒加载单例）。

改动原因：
  原实现在模块级就建连、探测 bucket、设置策略，等于「import 本模块」= 「访问一次 MinIO」。
  MinIO 不可达时，minio SDK 会重试 5 次（每次连接超时按分钟计），导入服务在启动阶段
  就被卡住好几分钟——报错还看不到，因为异常被吞进 print 里了。
  改成首次真正用到 MinIO 时才初始化，服务启动与单测都不再依赖对象存储可达。
"""

import json
import logging
from typing import Optional

from minio import Minio

from config.minio_config import minio_config

logger = logging.getLogger(__name__)

# 全局单例；None 表示尚未初始化或初始化失败
_minio_client: Optional[Minio] = None
# 初始化失败后不再反复重试：MinIO 挂掉时每次调用都去连一遍会拖慢整条导入链路
_init_failed: bool = False


def _ensure_bucket(client: Minio) -> None:
    """确保桶存在，并把桶策略设为 Public Read（前端可直接用 URL 访问图片）。"""
    if not client.bucket_exists(minio_config.bucket_name):
        client.make_bucket(minio_config.bucket_name)

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{minio_config.bucket_name}/*"],
            }
        ],
    }
    client.set_bucket_policy(minio_config.bucket_name, json.dumps(policy))


def _init_client() -> Optional[Minio]:
    """建连并初始化桶；失败返回 None（沿用原来的降级约定，只是不再阻塞 import）。"""
    client = Minio(
        endpoint=minio_config.endpoint,
        access_key=minio_config.access_key,
        secret_key=minio_config.secret_key,
        # secure=False 是否启用HTTPS加密连接；False=用HTTP，True=用HTTPS；本地/内网部署一律写False
        secure=False,
    )
    _ensure_bucket(client)
    return client


def get_minio_client() -> Optional[Minio]:
    """获取 MinIO 客户端；不可用时返回 None，调用方需自行判空。

    初始化失败只尝试一次：MinIO 挂掉时反复建连会拖慢整条导入链路。
    """
    global _minio_client, _init_failed

    if _minio_client is not None:
        return _minio_client
    if _init_failed:
        return None

    try:
        _minio_client = _init_client()
    except Exception as e:
        # 注意：这里必须用 logger 而不是 print，原来的 f-string 还漏了 f 前缀，
        # 报错时只会打印字面的 "{e}"，等于把真实原因藏起来了。
        logger.error("MinIO 初始化失败，图片上传相关功能将不可用：%s", e)
        _init_failed = True
        _minio_client = None

    return _minio_client
