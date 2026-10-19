"""守住「import 不该产生网络连接」这条底线。

背景：项目里有两个工具模块曾经在模块级直接建连——
  - utils/mongo_history_utils.py 在 import 时就创建 MongoClient 并建索引；
  - utils/minio_utils.py 在 import 时就探测 bucket、设置桶策略。
结果是「import 一下」等于「连一次中间件」：中间件不可达时，服务启动直接卡住几十秒到几分钟，
单元测试也没法在没有数据库的机器上跑。

这两条用例只断言"导入后单例仍为空"，看着很轻，但一旦有人把它们改回模块级初始化就会立刻失败。
"""

import sys


def test_mongo_tool_is_not_created_at_import_time():
    from utils import mongo_history_utils

    assert mongo_history_utils._history_mongo_tool is None, (
        "mongo_history_utils 又变成 import 即建连了，会导致服务启动被数据库阻塞"
    )


def test_minio_client_is_not_created_at_import_time():
    from utils import minio_utils

    assert minio_utils._minio_client is None, (
        "minio_utils 又变成 import 即建连了，会导致服务启动被对象存储阻塞"
    )


def test_these_modules_are_importable_without_any_middleware():
    """补充说明：上面两句 import 本身就是测试——只要中间件不可达时 import 会抛异常或卡住，
    这条用例就跑不完。能跑完就说明模块是干净的。
    """
    for name in ("utils.mongo_history_utils", "utils.minio_utils"):
        assert name in sys.modules
