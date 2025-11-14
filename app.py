import streamlit as st
import pandas as pd
from vrag import VRAG
from PIL import Image
from time import sleep

# ============ 页面配置 ============
st.set_page_config(
    page_title="Multimodal RAG Demo",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ 侧边栏配置 ============
st.sidebar.markdown("## ⚙️ 多模态RAG配置")
st.sidebar.markdown("---")

# 多模态模式选择
st.sidebar.markdown("### 启用的检索模式")
enable_text_search = st.sidebar.checkbox(
    "📄 文字检索",
    value=True,
    help="启用文本内容的搜索与检索"
)
enable_visual_search = st.sidebar.checkbox(
    "🖼️ 图像检索",
    value=True,
    help="启用图像信息的搜索与检索"
)
enable_table_search = st.sidebar.checkbox(
    "📊 表格检索",
    value=False,
    help="启用表格结构数据的搜索"
)

st.sidebar.markdown("---")

# 调试参数
MAX_ROUNDS = st.sidebar.slider(
    "🔄 最大检索步数",
    min_value=5,
    max_value=20,
    value=10,
    help="限制模型的最大搜索次数"
)

multimodal_weight = st.sidebar.slider(
    "🔗 多模态对齐权重",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    help="调整文字-图像对齐的重要程度（用于Reward函数）"
)

st.sidebar.markdown("---")

# ============ 示例问题 ============
examples = {
    "🔍 综合多模态问题": "用文字和图表解释全球气候变化的最新趋势",
    "📄 文字为主问题": "什么是VRAG方法？它的创新点是什么？",
    "🖼️ 图像为主问题": "请找到并展示伦敦大本钟的建筑特征",
    "🎓 学术问题": "对比深度学习中Transformer和RNN的优劣"
}

selected_example = st.sidebar.selectbox(
    "选择问题模板",
    options=list(examples.keys()),
    format_func=lambda x: x
)

# ============ 主界面标题 ============
st.markdown("# 🔗 多模态RAG系统 Demo")
st.markdown("""
这是一个支持**文字、图像、表格**多模态检索的问答系统。
系统将：
1. 📄 检索相关文字内容
2. 🖼️ 检索相关图像信息
3. 📊 提取相关表格数据（如启用）
4. 🔗 融合多模态信息生成答案
""")

st.markdown("---")

# ============ 问题输入 ============
st.markdown("### 📝 输入您的问题")
default_question = examples.get(selected_example, "")

question = st.text_input(
    "问题：",
    value=default_question,
    placeholder="请输入您的多模态问题...",
    label_visibility="collapsed"
)

col_submit, col_info = st.columns([1, 4])
with col_submit:
    submit_button = st.button("🚀 开始分析", key="submit_button")
with col_info:
    st.info("💡 系统将自动选择合适的检索工具来回答您的问题")

# ============ 初始化VRAG Agent ============
@st.cache_resource
def load_agent():
    """加载VRAG Agent"""
    agent = VRAG(
        base_url='http://localhost:8000/v1',
        generator=True,
        api_key='EMPTY'
    )
    return agent

try:
    agent = load_agent()
except Exception as e:
    st.error(f"❌ 无法加载VRAG Agent: {str(e)}")
    st.stop()

# ============ 结果展示区域 ============
if submit_button and question:
    st.markdown("---")
    st.markdown("### 📊 分析过程与结果")
    
    # 创建多列布局展示不同模态
    col_text, col_image, col_table = st.columns([1.2, 1.2, 1])
    
    # 初始化容器
    with col_text:
        st.markdown("#### 📄 文字检索结果")
        text_container = st.container()
    
    with col_image:
        st.markdown("#### 🖼️ 图像检索结果")
        image_container = st.container()
    
    with col_table:
        st.markdown("#### 📊 表格提取结果")
        table_container = st.container()
    
    # 思考过程与对齐展示
    process_container = st.container()
    alignment_container = st.container()
    answer_container = st.container()
    
    # ============ 生成器处理逻辑 ============
    agent.max_steps = MAX_ROUNDS
    generator = agent.run(question)
    
    # 存储多模态结果
    multimodal_data = {
        'text_results': [],
        'image_results': [],
        'table_results': [],
        'thinking_steps': [],
        'alignment_scores': []
    }
    
    try:
        step_count = 0
        
        for action, content, raw_content in generator:
            step_count += 1
            
            # ============ 思考步骤 ============
            if action == 'think':
                thinking_step = f"**步骤 {step_count}** - 🤔 思考中..."
                multimodal_data['thinking_steps'].append(thinking_step)
                
                with process_container:
                    with st.expander(f"🤔 思考步骤 {step_count}"):
                        st.write(content[:200] + "..." if len(content) > 200 else content)
                
                sleep(0.2)
            
            # ============ 文字搜索 ============
            elif action == 'search_text':
                if enable_text_search:
                    with text_container:
                        st.success("✓ 检索到文字内容")
                        text_preview = str(content)[:200] if isinstance(content, str) else str(content)[:200]
                        st.write(text_preview)
                        multimodal_data['text_results'].append(content)
            
            # ============ 图像搜索 - 新标准 ============
            elif action == 'search_image':
                if enable_visual_search:
                    try:
                        if isinstance(content, Image.Image):
                            with image_container:
                                st.success("✓ 检索到图像")
                                st.image(content, use_column_width=True)
                                multimodal_data['image_results'].append(content)
                    except Exception as e:
                        with image_container:
                            st.warning(f"图像加载失败: {str(e)}")
            
            # ============ 兼容原始搜索动作（仅图像）============
            elif action == 'search':
                if enable_visual_search:
                    try:
                        if isinstance(content, Image.Image):
                            with image_container:
                                st.success("✓ 检索到图像")
                                st.image(content, use_column_width=True)
                                multimodal_data['image_results'].append(content)
                        else:
                            # 作为文字处理
                            if enable_text_search:
                                with text_container:
                                    st.success("✓ 检索到内容")
                                    st.write(str(content)[:200])
                                    multimodal_data['text_results'].append(content)
                    except Exception as e:
                        st.warning(f"检索处理失败: {str(e)}")
            
            # ============ 表格搜索 ============
            elif action == 'search_table':
                if enable_table_search:
                    with table_container:
                        st.success("✓ 提取表格数据")
                        try:
                            if isinstance(content, pd.DataFrame):
                                st.dataframe(content, use_container_width=True)
                            else:
                                st.write(content)
                            multimodal_data['table_results'].append(content)
                        except Exception as e:
                            st.warning(f"表格处理失败: {str(e)}")
            
            # ============ 图像裁剪 ============
            elif action == 'crop_image':
                try:
                    with process_container:
                        with st.expander("🔍 已裁剪关键区域"):
                            if isinstance(content, tuple) and len(content) == 2:
                                # content 可能是 (cropped_image, marked_image)
                                st.image(content[0], use_column_width=True)
                            elif isinstance(content, Image.Image):
                                st.image(content, use_column_width=True)
                except Exception as e:
                    st.warning(f"裁剪显示失败: {str(e)}")
            
            # ============ 最终答案 ============
            elif action == 'answer':
                with answer_container:
                    st.markdown("---")
                    st.markdown("### ✅ 最终答案")
                    st.success(content)
                
                # 显示多模态对齐信息
                with alignment_container:
                    st.markdown("---")
                    st.markdown("### 🔗 多模态信息整合统计")
                    
                    col_summary1, col_summary2, col_summary3 = st.columns(3)
                    with col_summary1:
                        st.metric(
                            "📄 文字结果数",
                            len(multimodal_data['text_results']),
                            help="检索到的文字内容数量"
                        )
                    with col_summary2:
                        st.metric(
                            "🖼️ 图像结果数",
                            len(multimodal_data['image_results']),
                            help="检索到的图像数量"
                        )
                    with col_summary3:
                        st.metric(
                            "📊 表格结果数",
                            len(multimodal_data['table_results']),
                            help="提取的表格数量"
                        )
                    
                    # 对齐信息说明
                    st.info("""
                    **系统采用以下多模态对齐策略：**
                    - 🔗 **关键词匹配**：文字内容与图像标签的语义相似度
                    - 🎯 **上下文对齐**：确保不同模态信息指向同一概念
                    - 📚 **信息互补**：图像补充文字细节，文字提供图像背景
                    
                    **对齐权重**：{:.1f}
                    """.format(multimodal_weight))
    
    except StopIteration:
        st.info("✓ 处理完成")
    
    except Exception as e:
        st.error(f"❌ 处理过程中出错：{str(e)}")
        import traceback
        st.error("详细错误：" + traceback.format_exc()[:200])

# ============ 使用说明 ============
with st.expander("📖 使用说明"):
    st.markdown("""
    ### 如何使用这个多模态RAG系统？
    
    1. **选择问题模板** - 从左侧侧边栏选择问题模板或输入自定义问题
    2. **配置检索模式** - 勾选要启用的检索方式（文字/图像/表格）
    3. **调整参数** - 设置最大检索步数和多模态对齐权重
    4. **提交问题** - 点击"🚀 开始分析"按钮
    5. **查看结果** - 在三列布局中查看各模态的检索结果
    6. **查阅统计** - 查看多模态信息整合统计
    
    ### 问题类型建议
    
    - **纯文字问题**: "什么是XXX？"
    - **需要图像的问题**: "请展示XXX的外观"
    - **多模态综合问题**: "用文字和图表对比XXX和YYY"
    - **表格数据问题**: "统计各国GDP排名"
    """)

# ============ 页脚信息 ============
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px;'>
    <p>🔗 多模态RAG演示系统 | 基于VRAG框架 | 支持文字、图像、表格多模态融合</p>
    <p>当前对齐权重: {:.2f} | 最大步数: {} | 启用模式: {}</p>
</div>
""".format(
    multimodal_weight,
    MAX_ROUNDS,
    f"文字{'✓' if enable_text_search else '✗'} 图像{'✓' if enable_visual_search else '✗'} 表格{'✓' if enable_table_search else '✗'}"
), unsafe_allow_html=True)