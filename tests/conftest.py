"""测试公共配置。

这里只做一件事：把项目根目录塞进 sys.path。
项目源码用的是 `from processor... / from utils...` 这种「以项目根为基准」的绝对导入，
pytest 从别的目录启动时模块就找不到，所以必须显式补一下。

刻意不 import 任何业务模块——测试应当能在没有 Milvus / MongoDB / MinIO / 模型权重
的机器上直接跑通，全靠各自的测试文件按需导入。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
