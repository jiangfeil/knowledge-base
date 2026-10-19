"""把同一份文档用「修复后的切分逻辑」重新入库到独立集合，用于 A/B 对比。

为什么单独建集合而不覆盖 kb_chunks：
  覆盖是不可逆的，而这次要验证的只是「切分修好后有没有变好」。
  建个平行集合，好就切过去，不好就删掉，原数据一点不动。

用法：
    python evaluation/build_v2_collection.py                 # 默认写入 kb_chunks_v2
    python evaluation/build_v2_collection.py --collection kb_chunks_v3
    python -c "from pymilvus import MilvusClient; MilvusClient(uri='...').drop_collection('kb_chunks_v2')"   # 想撤就删

只跑三件事：切分 → 向量化 → 入库。跳过 PDF 解析、图片上传、主体名识别（这些对 A/B 没影响，
主体名统一沿用库里已有的 'HAK180烫金机'）。
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env", override=True)

from config.milvus_config import milvus_config
from processor.import_processor.nodes.node_bge_embedding import NodeBGEEmbedding
from processor.import_processor.nodes.node_document_split import NodeDocumentSplit
from processor.import_processor.nodes.node_import_milvus import NodeImportMilvus

SOURCE_MD = PROJECT_ROOT / "output" / "pdf_to_md" / "hak180产品安全手册" / "hak180产品安全手册.md"
ITEM_NAME = "HAK180烫金机"


def main():
    parser = argparse.ArgumentParser(description="用修复后的切分逻辑重建集合")
    parser.add_argument("--collection", default="kb_chunks_v2")
    parser.add_argument("--source", default=str(SOURCE_MD))
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"找不到源文件：{source}")

    text = source.read_text(encoding="utf-8")
    file_title = source.stem

    # 1. 切分（落盘备份需要一个可写目录，这里用临时目录，不留垃圾）
    with tempfile.TemporaryDirectory() as tmp:
        md_path = Path(tmp) / f"{file_title}.md"
        md_path.write_text(text, encoding="utf-8")
        state = {
            "task_id": "build_v2",
            "file_title": file_title,
            "md_path": str(md_path),
            "md_content": text,
        }
        chunks = NodeDocumentSplit().process(state)["chunks"]
    print(f"[1/3] 切分完成：{len(chunks)} 个 chunk")

    # 2. 补主体名 + 向量化
    for c in chunks:
        c["item_name"] = ITEM_NAME
    t0 = time.time()
    state["chunks"] = chunks
    chunks = NodeBGEEmbedding().process(state)["chunks"]
    print(f"[2/3] 向量化完成：{len(chunks)} 条，耗时 {time.time()-t0:.1f}s")

    # 3. 写入新集合（切换 collection 再调入库节点）
    original = milvus_config.chunks_collection
    milvus_config.chunks_collection = args.collection
    try:
        print(f"[3/3] 写入集合 {args.collection} ...")
        NodeImportMilvus().process(state)
    finally:
        milvus_config.chunks_collection = original

    print(f"\n完成。原集合 {original} 未被改动。")
    print(f"用新集合跑评估：python evaluation/run_eval.py --collection {args.collection}")


if __name__ == "__main__":
    main()
