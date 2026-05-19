import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay

# 设置页面配置
st.set_page_config(page_title="通用页岩气产能智能预测系统", layout="wide")

# 初始化 Session State (用于跨页面传递动态数据)
if 'model' not in st.session_state: st.session_state.model = None
if 'scaler' not in st.session_state: st.session_state.scaler = None
if 'features' not in st.session_state: st.session_state.features = []
if 'target' not in st.session_state: st.session_state.target = ""
if 'feature_stats' not in st.session_state: st.session_state.feature_stats = {}

st.title("📊 通用型页岩气产能主控因素与预测平台")
st.caption("自适应特征引擎：支持任意维度地质/工程参数导入")

tab1, tab2, tab3 = st.tabs(["1. 训练与主控分析", "2. 单井产能评估", "3. 使用指南"])

# --- TAB 1: 训练与主控分析 ---
with tab1:
    st.subheader("模型训练：动态特征选择")
    train_file = st.file_uploader("上传历史井数据 (CSV格式，包含地质/工程及产能数据)", type=['csv'])
    
    if train_file:
        df = pd.read_csv(train_file)
        # 数据清洗：仅保留数值型特征
        df_numeric = df.select_dtypes(include=[np.number])
        all_cols = df_numeric.columns.tolist()
        
        if len(all_cols) < 2:
            st.error("数据集中数值列不足，无法训练。请检查数据格式。")
        else:
            # 动态选择特征和目标
            col_sel1, col_sel2 = st.columns([1, 3])
            with col_sel1:
                # 默认将最后一列或名为'eur'的列设为目标Y
                default_target_idx = all_cols.index('eur') if 'eur' in all_cols else len(all_cols) - 1
                target_col = st.selectbox("选择预测目标 (Y)", options=all_cols, index=default_target_idx)
            
            with col_sel2:
                available_features = [c for c in all_cols if c != target_col]
                selected_features = st.multiselect("选择输入特征 (X)", options=available_features, default=available_features)
            
            if st.button("🚀 运行训练 & 科研分析"):
                if not selected_features:
                    st.error("请至少选择一个输入特征！")
                else:
                    # 1. 保存当前选择的特征状态
                    st.session_state.features = selected_features
                    st.session_state.target = target_col
                    # 提取训练集的统计信息(平均值)，用于Tab2的默认值填充
                    st.session_state.feature_stats = df_numeric[selected_features].mean().to_dict()
                    
                    # 2. 准备数据
                    X = df_numeric[selected_features]
                    y = df_numeric[target_col]
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # 3. 训练模型
                    model = RandomForestRegressor(n_estimators=200, random_state=42)
                    model.fit(X_scaled, y)
                    
                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    st.success(f"模型训练成功！共使用 {len(selected_features)} 个特征。")
                    
                    # 4. 可视化分析
                    c1, c2 = st.columns(2)
                    with c1:
                        # 特征重要性
                        importances = pd.DataFrame({'特征参数': selected_features, '权重': model.feature_importances_})
                        importances = importances.sort_values(by='权重', ascending=True)
                        fig1 = px.bar(importances, x='权重', y='特征参数', orientation='h', title="主控因素权重排序")
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with c2:
                        # 相关性热力图 (包含目标Y)
                        corr_cols = selected_features + [target_col]
                        fig2 = px.imshow(df_numeric[corr_cols].corr(), text_auto=".2f", title="参数耦合关系热力图")
                        st.plotly_chart(fig2, use_container_width=True)

                    # PDP 部分依赖图 (自动选取最重要的前两个特征进行绘制，防止报错)
                    top_features = importances.sort_values(by='权重', ascending=False)['特征参数'].tolist()[:2]
                    top_idx = [selected_features.index(f) for f in top_features]
                    
                    st.subheader(f"关键因素非线性响应 (PDP) - {top_features[0]} & {top_features[1]}")
                    fig_pdp, ax = plt.subplots(figsize=(10, 4))
                    PartialDependenceDisplay.from_estimator(model, X_scaled, features=top_idx, 
                                                           feature_names=selected_features, ax=ax)
                    st.pyplot(fig_pdp)

# --- TAB 2: 单井评估 (静动耦合预测) ---
with tab2:
    st.subheader("新井产能动态剖面预测 (ML + Duong模型耦合)")
    if st.session_state.model is None or not st.session_state.features:
        st.warning("请先在 Tab 1 完成动态特征模型训练。")
    else:
        # 1. 动态生成输入区 (基于 Tab 1 选择的特征)
        st.markdown("##### 📍 输入新井地质与工程参数")
        input_dict = {}
        # 每行放 4 个输入框
        cols = st.columns(4)
        for i, feature in enumerate(st.session_state.features):
            default_val = float(st.session_state.feature_stats[feature])
            with cols[i % 4]:
                # 动态生成数字输入框
                val = st.number_input(f"{feature}", value=default_val, format="%.4f")
                input_dict[feature] = val
        
        st.markdown("---")
        # 2. 物理模型参数设置
        with st.expander("⚙️ 高级选项: Duong 物理模型衰减参数设置"):
            d_col1, d_col2 = st.columns(2)
            m_val = d_col1.slider("递减指数 m (控制早期衰减速度, >1)", 1.01, 1.50, 1.15)
            a_val = d_col2.slider("常数 a (控制晚期尾部产量)", 0.5, 2.0, 1.05)
            q_ab = st.number_input("经济极限/废弃产量", value=0.5, step=0.1)
            
        if st.button("🚀 生成产能动态剖面"):
            # 将字典按特征顺序转为数组
            input_data = np.array([[input_dict[f] for f in st.session_state.features]])
            scaled_input = st.session_state.scaler.transform(input_data)
            
            # 机器学习预测体量
            pred_target = st.session_state.model.predict(scaled_input)[0]
            
            # 物理规律约束：反算初期产量
            t_days = np.arange(1, 3651) 
            shape_func = (t_days**-m_val) * np.exp((a_val / (1 - m_val)) * ((t_days**(1 - m_val)) - 1))
            shape_integral = np.sum(shape_func) / 10000.0 
            qi_calc = pred_target / shape_integral 
            
            # 生成日产量曲线
            q_time = qi_calc * shape_func
            valid_idx = np.where(q_time >= q_ab)[0]
            life_days = valid_idx[-1] if len(valid_idx) > 0 else 0
            life_years = life_days / 365.0
            
            # 结果展示
            st.markdown("### 📊 综合预测结果")
            res_c1, res_c2, res_c3 = st.columns(3)
            res_c1.metric(f"预测 {st.session_state.target} (机器学习)", f"{pred_target:.2f}")
            res_c2.metric("推算初期日产 ($q_i$)", f"{qi_calc:.2f}")
            res_c3.metric("经济开采寿命", f"{life_years:.1f} 年")
            
            # 绘制递减曲线
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(x=t_days, y=q_time, mode='lines', name='日产量', line=dict(color='#1f77b4', width=3)))
            fig_curve.add_hline(y=q_ab, line_dash="dash", line_color="red", annotation_text=f"废弃线 ({q_ab})")
            
            fig_curve.update_layout(
                title=f"单井生命周期产能递减预测 ({st.session_state.target})", 
                xaxis_title="生产时间 (天)", 
                yaxis_title="产量",
                hovermode="x unified"
            )
            st.plotly_chart(fig_curve, use_container_width=True)
            
            # 双对数诊断图
            fig_log = go.Figure()
            fig_log.add_trace(go.Scatter(x=t_days, y=q_time, mode='lines', name='日产量'))
            fig_log.update_layout(
                title="流动阶段诊断图 (双对数坐标)", 
                xaxis_title="生产时间 (天)", 
                yaxis_title="产量",
                xaxis_type="log", yaxis_type="log"
            )
            with st.expander("查看高级双对数诊断图"):
                st.plotly_chart(fig_log, use_container_width=True)

# --- TAB 3: 使用指南 ---
with tab3:
    st.markdown("""
    ### 系统升级说明 (自适应特征版)：
    1. **动态特征读取**：现在您可以上传任何包含地质、工程参数的表单。系统会自动剔除文字列。
    2. **自由指派 X 与 Y**：在 Tab 1，您可以自由选择哪些列作为输入，哪一列作为预测目标。
    3. **自适应 UI 生成**：Tab 2 的输入界面不再是固定的几个框，而是根据您在 Tab 1 选择的特征**自动生成**，并自动填入训练集的平均值作为基准参考。
    """)
