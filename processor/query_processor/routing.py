"""查询流程的条件路由判定。

为什么单独放一个模块：
  1. 路由规则本质是「纯函数」——输入 state、输出下一个节点名，不依赖任何外部服务；
  2. main_graph 会连带导入 Milvus / OpenAI / MCP 等重依赖，import 一次要十几秒，
     把判定逻辑放在里面就没法做秒级回归测试；
  3. 路由是最容易写错、也最该被测试覆盖的一环（错一次就是整条链路跑偏）。
"""

# 直接输出答案（跳过检索）
ROUTE_ANSWER = "node_answer_output"
# 进入多路检索
ROUTE_SEARCH = "node_multi_search"


def decide_after_item_name_confirm(state) -> str:
    """主体名称确认后的条件路由：要不要继续检索。

    判定依据只有一个——state 里有没有现成的 answer。
    有 answer 说明 node_item_name_confirm 已经给出了最终回复，直接输出即可；
    没有则说明主体已确认，可以放心往下走检索。

    state 里出现 answer 的两种情况：

    1. 多选一（反问用户）
       - 场景：用户问得太模糊（比如"华为P60"），库里有"华为P60 128G"和"华为P60 Art"两个型号，
         不足以直接确认。
       - 处理：节点生成一条反问句作为 answer，例如
         "您是想问以下哪个产品：华为P60 128G、华为P60 Art？请明确一下型号。"
       - 结果：不需要检索文档，直接把这句话发给用户让他选。

    2. 查无此人（拒绝回答）
       - 场景：用户问了一个系统里压根没有的商品（比如"小米15"，库里只有华为的数据）。
       - 处理：节点生成一条拒绝句作为 answer，例如
         "抱歉，未找到相关产品，请提供准确型号以便我为您查询。"
       - 结果：同样不需要后续检索，直接结束流程。

    :param state: 查询流程状态（QueryGraphState，本质是 dict）
    :return: 下一个节点的名称
    """
    if state.get("answer"):
        return ROUTE_ANSWER
    return ROUTE_SEARCH
