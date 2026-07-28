# CogDoc 设计升级说明

本项目没有直接复制 CogDoc 的 Rust 内核或完整服务层，而是借鉴其设计思想，按当前
LangGraph、Streamlit、FastAPI 和本地单机部署方式实现了四项能力。核心状态存储使用
Python 标准库 SQLite，不增加新的第三方依赖。

## 1. Citation Validator

RAG 路径在 `generate` 后增加 `citation_validation` 节点：

```text
generate
  -> citation_validation
  -> 校验失败且未超过次数：citation_repair -> citation_validation
  -> source_verification
```

校验内容包括：

- `[n]` 是否能够映射到本轮 rerank 后的证据；
- 回答中列出的 source 文档名和页码是否与检索元数据一致；
- 事实性结论是否带引用，以及引用覆盖率是否达到阈值；
- 校验失败时调用当前回答模型做一次受约束修复，模型不可增加新事实；
- 无模型 API 时不伪造修复，保留校验失败状态并在界面明确提示。

只有 `Source Verification` 和 `Citation Validator` 同时通过的 RAG 回答，才允许写入
Agent 记忆。

## 2. Trace Debug

每次调用生成唯一 `trace_id`，并记录以下结构化事件：

- Router 类型、路径、置信度与理由；
- Query Rewrite 轮次和查询变体；
- 混合检索候选数、向量命中数与 BM25 命中数；
- Rerank 模型、保留数和最高分；
- Generation 后端、答案长度和证据数；
- Citation Validation 结果与覆盖率；
- Citation Repair 次数；
- Source Verification 结果和检索轮次；
- 每个节点的执行耗时。

Trace 会展示在 Streamlit 右侧面板，并保存到 `data/traces/<trace_id>.json`。FastAPI
提供 `GET /traces` 和 `GET /traces/{trace_id}`。

## 3. 分层记忆

记忆存储在 `data/agent_state.db`：

| 层级 | 范围 | 内容 | 遗忘策略 |
| --- | --- | --- | --- |
| 短期记忆 | 当前 session | 最近通过校验的用户/助手消息 | 消息数和字符数双重限制 |
| 摘要记忆 | 当前 session | 从短期窗口淘汰的历史对话抽取摘要 | 限制最大摘要字符数 |
| 长期记忆 | memory scope 跨 session | 用户显式保存的稳定事实、偏好和约束 | 去重、重要度排序、容量限制 |

长期记忆采用中英文词法重合度、重要度和更新时间进行 query-aware 召回。短期消息用于
追问补全；摘要和长期记忆只帮助 Router 和生成节点理解上下文，不能被当作文献证据。

## 4. 反馈闭环

每条反馈与 `trace_id`、问题、答案、引用校验结果和检索证据绑定。支持：

- 有帮助/需要改进；
- 问题类型；
- 正确答案或纠正内容；
- 待审核、通过、拒绝、已解决状态；
- Streamlit 审核队列；
- FastAPI 提交、查询和审核接口。

负反馈或带纠错的反馈会写入 `evaluation/feedback_bad_cases.jsonl`，作为待人工审核的
离线评测坏例。反馈不会直接修改知识库或线上排序，避免错误反馈污染生产索引。

## 主要接口

```text
POST   /chat
GET    /memory/{session_id}
DELETE /memory/{session_id}
POST   /memory/long-term
DELETE /memory/long-term/{memory_id}
GET    /traces
GET    /traces/{trace_id}
POST   /feedback
GET    /feedback
POST   /feedback/{feedback_id}/review
```
