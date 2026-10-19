"""文档切分（node_document_split）的单测。

被测对象：processor/import_processor/nodes/node_document_split.py

这个节点的输入是整篇 Markdown，输出是要进向量库的 chunk 列表，是整条导入链路的
「语义边界划定者」——切坏了，后面 rerank 再准也救不回来。所以这里按它的六个步骤
逐段覆盖：输入校验 → 标题粗切 → 无标题兜底 → 长切短合 → 父标题兜底 → 落盘备份。

测试用很小的 max_content_length / min_content_length（120 / 40），
既能让切分逻辑在几行文本内触发，也不依赖任何真实文档和外部服务。
"""

import json

import pytest

from processor.import_processor.exceptions import ImportProcessError, StateFieldError
from processor.import_processor.import_config import ImportConfig
from processor.import_processor.nodes.node_document_split import NodeDocumentSplit

MAX_LEN = 120
MIN_LEN = 40


@pytest.fixture()
def node():
    """用缩小阈值的配置构造节点，避免在测试里塞几千字的文档。"""
    return NodeDocumentSplit(config=ImportConfig(max_content_length=MAX_LEN, min_content_length=MIN_LEN))


def _section(title, body, parent=None, file_title="手册"):
    sec = {"title": title, "content": f"{title}\n\n{body}" if title else body, "file_title": file_title}
    if parent is not None:
        sec["parent_title"] = parent
    return sec


class TestStep1Inputs:
    """步骤 1：输入校验与标准化"""

    def test_returns_normalized_content(self, node):
        content, title = node._step_1_get_inputs({
            "file_title": "手册",
            "md_path": "x.md",
            "md_content": "第一行\r\n第二行\r第三行",
        })

        # \r\n 与裸 \r 都要归一成 \n，否则后面按行切分会多出空行
        assert content == "第一行\n第二行\n第三行"
        assert title == "手册"

    @pytest.mark.parametrize("missing", ["file_title", "md_path", "md_content"])
    def test_missing_field_raises(self, node, missing):
        state = {"file_title": "手册", "md_path": "x.md", "md_content": "内容"}
        del state[missing]

        with pytest.raises(StateFieldError):
            node._step_1_get_inputs(state)

    def test_empty_value_also_raises(self, node):
        """字段存在但为空字符串，走的是 `if not xxx` 分支，同样按缺失处理。"""
        with pytest.raises(StateFieldError):
            node._step_1_get_inputs({"file_title": "", "md_path": "x.md", "md_content": "内容"})


class TestStep2SplitByTitles:
    """步骤 2：按 Markdown 标题粗切"""

    def test_splits_one_section_per_heading(self, node):
        md = "# 一级标题\n正文A\n## 二级标题\n正文B\n### 三级标题\n正文C"

        sections, title_count, lines = node._step_2_split_by_titles(md, "手册")

        assert title_count == 3
        assert lines == 6
        assert [s["title"] for s in sections] == ["# 一级标题", "## 二级标题", "### 三级标题"]
        # 标题行本身也保留在 content 里，作为向量的上下文
        assert sections[1]["content"] == "## 二级标题\n正文B"
        assert sections[1]["file_title"] == "手册"

    def test_preamble_before_first_heading_is_kept_with_empty_title(self, node):
        """首个标题之前的正文不能丢——它会被归到一个 title 为空的章节里。"""
        md = "这是一段前言\n# 正文标题\n内容"

        sections, title_count, _ = node._step_2_split_by_titles(md, "手册")

        assert title_count == 1
        assert sections[0]["title"] == ""
        assert sections[0]["content"] == "这是一段前言"

    def test_heading_inside_code_block_is_not_split(self, node):
        """代码块里的 # 注释不能当标题，否则一段代码会被切碎。"""
        md = "# 真标题\n```bash\n# 这是注释不是标题\nls -l\n```\n结束"

        sections, title_count, _ = node._step_2_split_by_titles(md, "手册")

        assert title_count == 1
        assert len(sections) == 1
        assert "# 这是注释不是标题" in sections[0]["content"]

    def test_indented_heading_is_recognized(self, node):
        """允许标题前有缩进（正则里是 \\s* 而不是行首严格匹配）。"""
        md = "  ## 缩进标题\n正文"

        sections, title_count, _ = node._step_2_split_by_titles(md, "手册")

        assert title_count == 1
        assert sections[0]["title"] == "## 缩进标题"

    def test_hash_without_space_is_not_a_heading(self, node):
        """# 后必须有空格，`#标签` 这种不算标题。"""
        md = "#标签不是标题\n正文"

        _, title_count, _ = node._step_2_split_by_titles(md, "手册")

        assert title_count == 0


class TestStep3NoTitleFallback:
    """步骤 3：无标题兜底"""

    def test_whole_document_becomes_single_section(self, node):
        sections = node._step_3_handle_no_title("纯文本内容", [], 0, "手册")

        assert len(sections) == 1
        assert sections[0]["title"] == "无标题"
        assert sections[0]["content"] == "纯文本内容"
        assert sections[0]["file_title"] == "手册"

    def test_keeps_sections_when_titles_found(self, node):
        original = [{"title": "# T", "content": "# T\n内容", "file_title": "手册"}]

        result = node._step_3_handle_no_title("随便", original, 1, "手册")

        assert result is original


class TestSplitLongSection:
    """步骤 4-1：超长章节二次切分"""

    def test_short_section_is_not_split_but_parent_backfilled(self, node):
        """短章节不切分，但要补上 parent_title，供后续短合阶段判断"是否同一章节"。"""
        sec = _section("# 短标题", "一句话")

        parts = node._split_long_section(sec)

        assert len(parts) == 1
        assert parts[0]["content"] == sec["content"]
        assert parts[0]["parent_title"] == "# 短标题"

    def test_long_section_is_split_into_numbered_parts(self, node):
        """超过阈值时必须切成多块，且每块带序号、父标题，能回指原章节。"""
        body = "\n\n".join(f"第{i}段正文内容，用来把这一段撑长" for i in range(1, 12))
        sec = _section("# 长章节", body)
        assert len(sec["content"]) > MAX_LEN  # 前提确认

        parts = node._split_long_section(sec)

        assert len(parts) > 1
        assert [p["part"] for p in parts] == list(range(1, len(parts) + 1))
        assert all(p["parent_title"] == "# 长章节" for p in parts)
        assert parts[0]["title"] == "# 长章节-1"
        # 每个子块都带上标题前缀，保证脱离原文后仍有上下文
        assert all(p["content"].startswith("# 长章节") for p in parts)

    def test_each_part_respects_max_length(self, node):
        """切分的目的是适配上下文窗口，所以长度上限必须守住。"""
        body = "\n\n".join(f"第{i}段正文内容，用来把这一段撑长" for i in range(1, 12))

        parts = node._split_long_section(_section("# 长章节", body))

        assert all(len(p["content"]) <= MAX_LEN for p in parts)

    def test_does_not_duplicate_title_in_body(self, node):
        """正文开头重复出现的标题会被剥掉，避免每个子块都重复一遍标题文字。"""
        body = "\n\n".join(f"第{i}段正文内容，用来把这一段撑长" for i in range(1, 12))

        parts = node._split_long_section(_section("# 长章节", body))

        # 第一块 = 标题前缀 + 去重后的正文，标题只应出现一次
        assert parts[0]["content"].count("# 长章节") == 1

    def test_untitled_section_uses_chunk_prefix(self, node):
        """无标题章节切分后，子块标题退化成 chunk-N。"""
        body = "很长的一段正文。" * 40
        sec = {"title": "", "content": body, "file_title": "手册"}

        parts = node._split_long_section(sec)

        assert len(parts) > 1
        assert parts[0]["title"] == "chunk-1"

    def test_regression_no_attribute_error_on_long_section(self, node):
        """回归测试：曾经这里写成 `self.self.logger.debug(...)`，
        只要文档里出现一个超过 max_content_length 的章节，切分就会抛 AttributeError，
        整篇文档导入失败。这条用例专门盯住这个坑。
        """
        body = "\n\n".join(f"第{i}段正文内容，用来把这一段撑长" for i in range(1, 12))

        parts = node._split_long_section(_section("# 长章节", body))  # 不应抛异常

        assert len(parts) > 1


class TestMergeShortSections:
    """步骤 4-2：过短章节合并"""

    def test_empty_list(self, node):
        assert node._merge_short_sections([]) == []

    def test_merges_short_chunks_of_same_parent(self, node):
        sections = [
            {"title": "# P-1", "content": "# P\n短一", "parent_title": "# P", "part": 1},
            {"title": "# P-2", "content": "# P\n短二", "parent_title": "# P", "part": 2},
        ]

        merged = node._merge_short_sections(sections)

        assert len(merged) == 1
        # 合并时剥掉后一块重复的父标题
        assert merged[0]["content"] == "# P\n短一\n\n短二"
        # 序号保留最新的，便于溯源到原文章节末尾
        assert merged[0]["part"] == 2

    def test_does_not_merge_across_different_parents(self, node):
        """跨章节合并会造成语义串味，必须禁止。"""
        sections = [
            {"title": "# A-1", "content": "# A\n短", "parent_title": "# A"},
            {"title": "# B-1", "content": "# B\n短", "parent_title": "# B"},
        ]

        merged = node._merge_short_sections(sections)

        assert len(merged) == 2

    def test_long_chunk_is_not_absorbed(self, node):
        """当前块已经够长，就不该再吞并下一块。"""
        long_text = "# P\n" + "正文" * 30
        assert len(long_text) > MIN_LEN

        merged = node._merge_short_sections([
            {"title": "# P-1", "content": long_text, "parent_title": "# P"},
            {"title": "# P-2", "content": "# P\n短", "parent_title": "# P"},
        ])

        assert len(merged) == 2


class TestStep4RefineChunks:
    """步骤 4 整体：长切 + 短合 + 父标题兜底"""

    def test_parent_title_fallback(self, node):
        """Milvus schema 里 parent_title 是必填字段，没有就用自身 title 兜底。"""
        sections = [{"title": "# 只有标题", "content": "# 只有标题\n正文", "file_title": "手册"}]

        refined = node._step_4_refine_chunks(sections)

        assert refined[0]["parent_title"] == "# 只有标题"

    def test_short_distinct_headings_are_not_merged(self, node):
        """回归测试：两个都很短、但标题不同的章节，不能被合并成一个 chunk。

        曾经 _split_long_section 对短章节原样返回、不补 parent_title，
        导致 _merge_short_sections 看到 None == None，误判成"同一父章节"而跨标题合并，
        "# 产品概述" 和 "# 安全须知" 就串味了。
        """
        sections = [
            {"title": "# 产品概述", "content": "# 产品概述\n这台设备用于压印。", "file_title": "手册"},
            {"title": "# 安全须知", "content": "# 安全须知\n使用前请断电。", "file_title": "手册"},
        ]

        refined = node._step_4_refine_chunks(sections)

        assert len(refined) == 2
        assert refined[0]["parent_title"] == "# 产品概述"
        assert refined[1]["parent_title"] == "# 安全须知"

    def test_empty_title_falls_back_to_empty_string(self, node):
        """title 也为空时兜底成空字符串，而不是 None（否则 Milvus 插不进去）。"""
        refined = node._step_4_refine_chunks([{"title": "", "content": "段落", "file_title": "手册"}])

        assert refined[0]["parent_title"] == ""


class TestProcessEndToEnd:
    """节点完整跑一遍：从 Markdown 到 chunks.json"""

    def test_produces_chunks_and_backup_file(self, node, tmp_path):
        md = "# 产品概述\n这台设备用于压印。\n\n# 安全须知\n使用前请断电。"
        md_path = tmp_path / "manual.md"

        state = {
            "task_id": "t1",
            "file_title": "手册",
            "md_path": str(md_path),
            "md_content": md,
        }
        result = node(state)

        assert len(result["chunks"]) == 2
        assert all("parent_title" in c for c in result["chunks"])
        # 备份文件落在 md 同目录的 chunks.json
        backup = json.loads((tmp_path / "chunks.json").read_text(encoding="utf-8"))
        assert backup == result["chunks"]

    def test_missing_task_id_surfaces_wrapped_error(self, node):
        """基类 __call__ 依赖 state["task_id"]，缺失时被包成 ImportProcessError 抛出。"""
        with pytest.raises(ImportProcessError):
            node({"file_title": "手册", "md_path": "x.md", "md_content": "# T\n内容"})
