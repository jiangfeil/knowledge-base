import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

print(os.getenv("OPENAI_API_KEY"))

# 示例：假设系统有环境变量 MY_KEY=system_val，.env里 MY_KEY=dotenv_val
print(os.getenv("MY_KEY"))
# load_dotenv() → 输出 system_val（系统优先级高）
# load_dotenv(override=True) → 输出 dotenv_val（.env覆盖系统）