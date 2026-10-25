# MY_AGENT 环境搭建 / 一键重建指南（D1 整理）

> 目标：任何人拿到这份代码，都能在本机**一键重建**出可运行的环境。
> 依赖清单：`requirements.txt`（直接依赖，日常维护用）；`requirements.lock`（全量锁定，逐字节复现用）。

## 一、前置条件
- Python 3.10+（本机已用 3.12 验证）
- 一个 `.env` 文件（复制 `.env.example` 并填入 API Key，见下）

## 二、一键重建步骤（Windows）
```bash
# 1. 进入项目根目录
cd D:/my_agent

# 2. 创建虚拟环境（只需第一次）
python -m venv venv

# 3. 激活环境
venv\Scripts\activate

# 4. 安装依赖（直接依赖，版本见 requirements.txt）
pip install -r requirements.txt

# 5.（可选）逐字节完全一致地复现
# pip install -r requirements.lock

# 6. 准备环境变量
copy .env.example .env
# 然后编辑 .env，填入 ZHIPU_API_KEY 等

# 7. 启动（任选其一）
streamlit run app.py          # Streamlit 三模式界面
# 或 python main.py            # 双 Agent 报告流水线
```

## 三、macOS / Linux 差异
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
（其余步骤相同）

## 四、为什么分两个文件？
| 文件 | 内容 | 何时用 |
|---|---|---|
| `requirements.txt` | 7 个**直接依赖** + 精确版本 | 日常开发、升级、维护 |
| `requirements.lock` | `pip freeze` 全量（含所有传递依赖） | CI、部署、别人复现、排查依赖冲突 |

只锁定直接依赖，传递依赖交给 pip 解析，升级时更灵活；
锁文件保证"今天能跑，三个月后别人拉下来也能跑"。

## 五、验证"一键重建"是否成功
```bash
# 激活后，确认关键包都在
python -c "import chromadb, zhipuai, yaml, rank_bm25, streamlit, langgraph; print('OK')"
```
输出 `OK` 即说明依赖完整、环境可重建。

## 六、注意事项
- `chorma_db/` 是本地向量库数据（目录名是历史拼写，保留原样），`.gitignore` 已忽略，重建环境不影响它。
- `.env` 含密钥，**绝不**提交到 git（已在 `.gitignore`）。
- 若重装后 day47~day50（langgraph 相关）报 `ModuleNotFoundError: langgraph`，说明用了旧 requirements.txt——本版已补 `langgraph==1.2.12`。
