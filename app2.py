import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from openai import OpenAI
from langchain.prompts import PromptTemplate
import base64, glob, os
from langchain_community.document_loaders import PyPDFLoader

st.set_page_config(page_title="AI Interior Style Analyzer", layout="centered")
st.title("🏠 인테리어 스타일 분석 & 문서 기반 추천")

openai_api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요:", type="password")
client = OpenAI(api_key=openai_api_key) if openai_api_key else None
uploaded_image = st.file_uploader("인테리어 사진을 업로드하세요", type=['png', 'jpg', 'jpeg'])

# 세션 상태 초기화
if "image_analysis" not in st.session_state:
    st.session_state["image_analysis"] = None
if "rag_result" not in st.session_state:
    st.session_state["rag_result"] = None

if uploaded_image:
    st.image(uploaded_image, caption="업로드한 이미지", width=600)

    # 추가 요구사항 입력란
    user_req = st.text_input("추가 요구사항(예: 자연친화적인, 뉴트럴톤, 소파 등)", "")

    if st.button("스타일 분석 및 문서 기반 추천"):
        if not openai_api_key:
            st.warning("OpenAI API 키를 입력해주세요.")
        else:
            with st.spinner("이미지 분석 및 문서 참조 중..."):
                # 1. 이미지 스타일 분석
                encoded = base64.b64encode(uploaded_image.getvalue()).decode("utf-8")
                styles = ["Modern", "Scandinavian", "Industrial", "Zen", "Minimalism", "Bohemian", "Brutalism", "Victorian"]
                prompt_text = (
                    "이 이미지를 보고 가장 적합한 스타일을 고르고 이유를 설명하세요.\n"
                    f"스타일 리스트: {', '.join(styles)}\n"
                    "색상, 가구 배치, 스타일에 대한 간단한 평가도 추가해주세요."
                )
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an interior design expert."},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
                        ]}
                    ],
                    max_tokens=1000,
                    temperature=0.2,
                )
                image_analysis = response.choices[0].message.content

                # 2. PDF 문서 RAG
                pdf_files = glob.glob("data/*.pdf")
                docs = []
                for pdf in pdf_files:
                    loader = PyPDFLoader(pdf)
                    pages = loader.load()
                    for p in pages:
                        p.metadata["source"] = pdf
                        p.metadata["page"] = p.metadata.get("page", 0)
                    docs.extend(pages)
                if not docs:
                    st.error("PDF 문서를 찾을 수 없습니다.")
                    st.stop()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                texts = splitter.split_documents(docs)
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="vectordb2")
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                prompt = PromptTemplate(
                    template="""당신은 인테리어 디자인 전문가입니다.\n\n문서:\n{context}\n\n질문: {question}\n\n참조 문서를 인용하며 답변하세요.\n답변:""",
                    input_variables=["context", "question"]
                )
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=openai_api_key),
                    retriever=retriever,
                    memory=memory,
                    combine_docs_chain_kwargs={"prompt": prompt}
                )
                rag_prompt = f"""
                다음은 인테리어 이미지 분석 결과입니다:\n{image_analysis}\n
                위 분석 결과를 바탕으로, 참조 문서에서 찾을 수 있는 관련된 인테리어 디자인 원칙과 개선 방안을 제시해주세요.
                """
                rag_result = qa_chain({"question": rag_prompt})
                st.session_state["image_analysis"] = image_analysis
                st.session_state["rag_result"] = rag_result['answer']

    # 결과가 세션에 저장됐을 때만 버튼 활성화
    if st.session_state["rag_result"]:
        st.subheader("🔍 이미지 스타일 분석 결과")
        st.write(st.session_state["image_analysis"])

        st.subheader("📚 문서 기반 전문가 추천")
        st.write(st.session_state["rag_result"])

        if st.button("리모델링 이미지 생성 (DALL·E)"):
            with st.spinner("DALL·E로 이미지 생성 중..."):
                dalle_prompt = (
                    f"Photorealistic, clean, minimal, and realistic interior design for a living room. "
                    f"Do not overcrowd with furniture. Use only essential, modern, and stylish furniture. "
                    f"Natural lighting, balanced composition, and a sense of spaciousness. "
                    f"Reflect the following remodeling advice: {st.session_state['rag_result']}. "
                    f"User requirements: {user_req}. "
                    f"Make it look like a real, beautiful, and livable space."
                )
                img_response = client.images.generate(
                    model="dall-e-3",
                    prompt=dalle_prompt,
                    size="1024x1024",
                    n=1
                )
                img_url = img_response.data[0].url
                st.subheader("🖼️ 리모델링된 인테리어 이미지")
                st.image(img_url, caption="리모델링 결과", use_column_width=True)
