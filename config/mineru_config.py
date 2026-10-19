import os
from dataclasses import dataclass

from dotenv import load_dotenv
# 1.加载配置文件
load_dotenv(override=True)

@dataclass
class MineruConfig:
    base_url_test: str
    api_token_test: str
    base_url_prod: str
    api_token_prod: str
mineru_config = MineruConfig(
    base_url_test="test",
    api_token_test="test",
    base_url_prod=os.getenv("MINERU_BASE_URL",""),
    api_token_prod=os.getenv("MINERU_API_TOKEN","")

)
