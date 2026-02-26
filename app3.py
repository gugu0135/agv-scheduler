import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import datetime

# ⚠️ 修复了导入路径，改为引入正确的 nsga2_core01
from nsga2_core01 import NSGA2Scheduler, parse_schedule_to_gantt

st.set_page_config(page_title="AGV调度系统", layout="wide")

# --- 初始化 Session State ---
if 'data_imported' not in st.session_state:
    st.session_state.data_imported = False

st.title("🚀 AGV智能车间调度优化平台")

# --- 界面 1：基础数据设置 ---
st.header("车间基础参数配置")
col_base1, col_base2, col_base3, col_base4 = st.columns(4)
with col_base1:
    m_count = st.number_input("机器数量 (M)", min_value=1, value=3)
with col_base2:
    j_count = st.number_input("工件数量 (J)", min_value=1, value=2)
with col_base3:
    agv_count = st.number_input("AGV数量", min_value=1, value=2)
with col_base4:
    st.info(f"当前配置：{j_count}工件 × {m_count}机器")

# --- 数据录入与缺失检查 ---
st.subheader("工序明细录入")
# 默认展示数据
init_df = pd.DataFrame([
    {"工件": 1, "工序": 1, "机器列表": "1,2", "加工时间": "10,12"},
    {"工件": 1, "工序": 2, "机器列表": "2,3", "加工时间": "15,10"},
    {"工件": 2, "工序": 1, "机器列表": "1,3", "加工时间": "8,11"}
])

edited_df = st.data_editor(init_df, num_rows="dynamic")


def check_missing_data(df):
    """验证数据完整性"""
    if df.isnull().values.any():
        return False, "发现空白单元格，请填写完整！"
    # 检查工件的工序是否填写
    for j in range(1, j_count + 1):
        job_ops = df[df["工件"] == j]
        if len(job_ops) == 0:
            return False, f"错误：缺失工件 {j} 的所有工序数据！"
    return True, ""


# --- 界面 2：导入与 MK01 展示 ---
if st.button("确认导入并校验数据"):
    is_ok, err_msg = check_missing_data(edited_df)
    if not is_ok:
        st.error(err_msg)
        st.session_state.data_imported = False
    else:
        st.success("数据校验成功！已转换为标准格式：")
        st.session_state.data_imported = True

        # 构建展示表格
        mk01_rows = []
        for _, r in edited_df.iterrows():
            m_ids = str(r['机器列表']).split(',')
            t_vals = str(r['加工时间']).split(',')
            detail = " | ".join([f"(M{mi.strip()}, T{ti.strip()})" for mi, ti in zip(m_ids, t_vals)])
            mk01_rows.append({"工件-工序": f"J{r['工件']}-O{r['工序']}", "可选机器数": len(m_ids), "详细配置": detail})

        st.table(pd.DataFrame(mk01_rows))

# --- 界面 3：算法参数设置 ---
if st.session_state.data_imported:
    st.divider()
    st.header("2. NSGA-II 算法参数设置")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pop = st.number_input("种群大小", 10, 500, 100)
    with c2:
        gen = st.number_input("迭代次数", 10, 1000, 50)
    with c3:
        pc = st.slider("交叉概率", 0.0, 1.0, 0.8)
    with c4:
        pm = st.slider("变异概率", 0.0, 1.0, 0.1)

    btn_run = st.button("开始调度优化", type="primary")

    # --- 界面 4：结果展示 ---
    if btn_run:
        st.subheader("3. 调度优化结果")
        # 实例化真正的调度器
        solver = NSGA2Scheduler(pop, gen, pc, pm, edited_df, agv_count)

        progress_bar = st.progress(0)
        history = solver.run(lambda p: progress_bar.progress(p))

        # 结果图表
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.write("**帕累托前沿分布 (Pareto Front)**")
            fig, ax = plt.subplots()
            h_np = np.array(history)
            ax.scatter(h_np[:, 0], h_np[:, 1], c='red', edgecolors='k')
            ax.set_xlabel("Makespan (min)")
            ax.set_ylabel("AGV Energy Consumption")
            st.pyplot(fig)

        with res_col2:
            st.write("**收敛曲线**")
            st.line_chart(h_np[:, 0])

        st.divider()
        st.subheader("🗓️ 机器与AGV协同调度甘特图")
        st.info("图中展示的是使得 Makespan (完工时间) 最小的代表性排产方案。")

        # ⚠️ 关键修复：从 solver 中提取真实的排产数据，并转化为甘特图格式
        raw_schedule_data = solver.get_best_schedule_data()

        if raw_schedule_data:
            gantt_list = parse_schedule_to_gantt(raw_schedule_data, start_hour=8, time_unit="minutes")
            df_gantt = pd.DataFrame(gantt_list)

            if not df_gantt.empty:
                # 绘制 Plotly 真实时间线图
                fig_gantt = px.timeline(
                    df_gantt,
                    x_start="Start",
                    x_end="Finish",
                    y="Task",
                    color="Resource",
                    hover_name="Resource"
                )

                # 翻转 Y 轴，让机器按升序排列
                fig_gantt.update_yaxes(autorange="reversed")
                fig_gantt.update_layout(showlegend=True, height=500)

                st.plotly_chart(fig_gantt, use_container_width=True)
            else:
                st.warning("甘特图数据为空，未能成功解析排产方案。")
        else:
            st.error("未能找到最优排产计划，请检查数据输入和算法参数！")