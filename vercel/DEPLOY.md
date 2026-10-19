# 部署到自己的域名（Vercel）

目标：把 demo 从 `kb-rag-demo.app.workbuddy.host` 换成一个自己的地址，
比如 `kb.jiangfeil.top`。托管在 Vercel 免费额度内，一年只花域名钱（约 10 元）。

## 为什么这样改

| | 原版 `demo/server.py` | 本目录 |
| --- | --- | --- |
| 运行方式 | 长驻 `http.server` | Vercel Serverless Function |
| 语料向量 | 启动时调 9 次 embedding | **预计算存进 `api/_kb.json`** |
| 冷启动 | 要等预热完成才能回答 | 无预热，首个请求直接可用 |
| 单次问答开销 | 3 次 API | 3 次 API（不变） |
| 入口 | `server.py` 自己托管 HTML | 根目录 `index.html` 静态托管 |

向量预计算是必须的：Serverless 每个冷启动都是新进程，如果每次都 embed 一遍语料，
既拖慢首问（要多等 2 秒）又白烧 9 次调用额度。

## 文件

```
vercel/
├── index.html          前端（与 demo 版同一份）
├── vercel.json         函数超时配置
├── build_vectors.py    换语料后重算向量，一次性脚本
├── _smoke.py           本地冒烟测试，改完代码跑一下
└── api/
    ├── _core.py        检索 + 生成逻辑（下划线开头 = 不是路由）
    ├── _kb.json        9 条切片 + 预计算向量（98 KB）
    ├── ask.py          POST /api/ask
    └── health.py       GET  /api/health
```

## 部署步骤

### 1. 注册 Vercel

打开 https://vercel.com/signup ，用 **GitHub 账号**登录（不需要信用卡）。

### 2. 导入仓库

- 点 **Add New → Project**
- 选 `jiangfeil/knowledge-base`
- **Root Directory 填 `vercel`**（这一步最容易漏，填错会报 "No Output Directory"）
- Framework Preset 保持 **Other**
- 先不要点 Deploy，展开 **Environment Variables**

### 3. 配环境变量

| Name | Value |
| --- | --- |
| `DASHSCOPE_API_KEY` | 你的 DashScope key |

### 4. 部署

点 Deploy，等 1 分钟左右。

先访问 `https://<项目名>.vercel.app/api/health`，应该返回：

```json
{"ready": true, "chunks": 9, "dim": 1024, "error": ""}
```

`ready: false` 说明 key 没读到，回去检查环境变量名有没有拼错。

再访问首页问一句「HAK180 的工作温度是多少」，能答出来就说明**境外出节能调通 DashScope**。
如果这里报 502，说明当前 key 的来源做了地域限制，需要换成自己的阿里云百炼 key（见文末）。

### 5. 绑自己的域名

**买域名**（10 元左右）

- 阿里云 / 腾讯云搜 `.top`，首年通常 9–12 元
- 名字用拼音全拼，比如 `jiangfeil.top`
- 买完**不需要备案**——域名解析到境外服务器（Vercel）不涉及备案

**加 DNS 记录**

在域名控制台 → 解析设置，加一条：

| 记录类型 | 主机记录 | 记录值 |
| --- | --- | --- |
| CNAME | `kb` | `cname.vercel-dns.com` |

**在 Vercel 绑定**

项目 → Settings → Domains → 输入 `kb.jiangfeil.top` → Add。
等几十秒，证书自动签好，变成绿色 Valid 就通了。

最终地址：**https://kb.jiangfeil.top**

### 6. 把 Function Region 改成香港

项目 → Settings → Functions → Function Region 选 **Hong Kong (hkg1)**。
默认是美国华盛顿，调到香港后到 DashScope（杭州）的延迟从 ~250ms 降到 ~40ms，
首字时间能快 0.5 秒左右。

## 换语料

改完 `demo/data/chunks.json` 后：

```bash
cd vercel
DASHSCOPE_API_KEY=sk-xxx python build_vectors.py
```

再 `git push`，Vercel 会自动重新部署。

## 本地调试

```bash
cd vercel
DASHSCOPE_API_KEY=sk-xxx python _smoke.py
```

会依次验证 health / 空问题 / 非法 JSON / 正常问答 / 限流五项。

## 关于 key 的来源

如果部署后 `/api/health` 是 `ready: true` 但问答报 **502**，
说明 key 被限制了来源地域（Serverless 在境外）。解决：

1. 去 https://bailian.console.aliyun.com 注册（阿里云账号，实名）
2. 开通「百炼大模型服务」，新用户有免费额度
3. 创建 API-KEY，格式是 `sk-` 开头的 32 位
4. 在 Vercel 环境变量里替换掉

`text-embedding-v4` 和 `qwen-flash` 都在百炼的免费额度覆盖范围内，
个人 demo 的调用量基本用不完。
