# SceneSmith 改造总结（服务配置解耦 + 全流程 Tracking）

## 1) 本次目标

本次按照你的要求，完成了三件事：

1. 把外部服务配置独立管理（VLM、文生图、各检索/生成服务端点）
2. 在代码中加入全过程 tracking，记录调用程序/服务与耗时
3. 每次生成场景输出两类报告：
   - 流程顺序耗时报告
   - 按程序/服务分类统计报告（调用次数、每次耗时、总耗时）

另外，仓库已先恢复到原始版本后再开始改造。

## 2) 服务配置独立管理

### 2.1 新增统一配置文件

新增：`configurations/services/default.yaml`

集中管理：

- LLM 默认模型与视觉细节
- API timeout
- 图像生成后端（OpenAI/Gemini）
- 资产来源策略（默认改为 `hssd`，更适合无 GPU 场景）
- 各服务端口与主机
- VLM/图像服务的 `api_key` 与 `base_url` 的环境变量映射

### 2.2 主配置接入

`configurations/config.yaml` 增加默认项：

- `- services: default`

### 2.3 Agent/实验配置改为引用统一 services

以下配置中，外部服务相关字段改为 `${services...}` 引用：

- `configurations/floor_plan_agent/base_floor_plan_agent.yaml`
- `configurations/furniture_agent/base_furniture_agent.yaml`
- `configurations/wall_agent/base_wall_agent.yaml`
- `configurations/ceiling_agent/base_ceiling_agent.yaml`
- `configurations/manipuland_agent/base_manipuland_agent.yaml`
- `configurations/experiment/base_experiment.yaml`

额外调整：

- 各 agent 的 `session_memory.summarization_model` 改为 `"agent"`，跟随统一模型，便于你一键换模型。

## 3) VLM / 文生图端点与 Key 可独立替换

### 3.1 新增统一服务解析工具

新增：`scenesmith/utils/service_config.py`

提供：

- OpenAI endpoint/api key 解析
- Gemini api key 解析
- OpenAI/AsyncOpenAI 客户端统一构造

### 3.2 VLM 调用接入统一配置

- `scenesmith/agent_utils/vlm_service.py`
- `scenesmith/*_agents/base_*_agent.py`（家具/墙面/天花/manipuland）
- `scenesmith/experiments/base_experiment.py`（把 `services` 透传给 agent cfg）

现在 VLM 可通过 `services/default.yaml + 环境变量` 独立换：

- 模型：`services.llm.model`
- 端点：`SCENESMITH_VLM_BASE_URL`
- key：`SCENESMITH_VLM_API_KEY`

### 3.3 文生图调用接入统一配置

- `scenesmith/agent_utils/image_generation.py`
- `scenesmith/agent_utils/asset_manager.py`

现在图像生成可独立换：

- 后端：`services.image_generation.backend`（`openai` / `gemini`）
- OpenAI 图像端点：`SCENESMITH_IMAGE_BASE_URL`
- OpenAI 图像 key：`SCENESMITH_IMAGE_API_KEY`
- Gemini key：`SCENESMITH_GEMINI_API_KEY`

并保留 `OPENAI_API_KEY` / `GOOGLE_API_KEY` 作为回退。

## 4) 全流程 Tracking 与报告

### 4.1 新增 tracking 模块

新增：`scenesmith/utils/runtime_tracking.py`

功能：

- 记录事件到 `jsonl`（支持多进程并行写入）
- 提供 `track_runtime(...)` 上下文埋点
- 生成两份 JSON + Markdown 报告

### 4.2 埋点覆盖范围

已接入关键路径：

- 场景总流程（scene-level）
- 房间各阶段程序调用：家具/墙面/天花/manipuland/后处理/导出
- floor plan 子进程
- 组装输出阶段
- 关键服务调用：
  - VLM OpenAI API（responses/chat）
  - 图像生成 API（OpenAI/Gemini）
  - HSSD/Objaverse/Articulated/Materials/Geometry 的 HTTP 客户端调用
  - session summarization 调用

### 4.3 每个场景输出文件

在 `scene_xxx/` 下自动生成：

- `runtime_timeline_report.json`
- `runtime_timeline_report.md`
- `runtime_program_report.json`
- `runtime_program_report.md`
- （原始事件）`runtime_events.jsonl`

其中：

- `runtime_timeline_report.*`：按流程顺序
- `runtime_program_report.*`：按程序/服务分类，包含调用次数、每次耗时、总耗时、平均耗时

## 5) 当前默认策略（考虑无 GPU）

在 `services/default.yaml` 里，默认将：

- `asset_manager.general_asset_source: "hssd"`

这样默认不走本地 3D 生成后端（更接近无 GPU 可运行路径）。

## 6) 你后续最常改的位置

优先改这一处：`configurations/services/default.yaml`

常用修改项：

- 换模型：`llm.model`
- 换 VLM 端点：设置环境变量 `SCENESMITH_VLM_BASE_URL`
- 换 VLM key：设置环境变量 `SCENESMITH_VLM_API_KEY`
- 换图像后端：`image_generation.backend`
- 换图像端点/key：`SCENESMITH_IMAGE_BASE_URL` / `SCENESMITH_IMAGE_API_KEY`

## 7) 说明

由于当前机器无 GPU，本次只做了静态与语法层验证（`py_compile` + 配置加载）；未执行完整生成流程压测。后续可在你指定的服务 key 与端点就绪后，做一轮小样本运行验证（例如单场景、单房间）。
