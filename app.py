import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from openai import OpenAI
import glob
import base64

# 페이지 설정
st.set_page_config(page_title="AI Interior Advisor", layout="centered")
st.title("🏠 인테리어 디자인 평가 및 리모델링 방안 추천 AI")

# OpenAI API 키 입력
openai_api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요:", type="password")

# GPT-4o API Client 초기화
client = OpenAI(api_key=openai_api_key) if openai_api_key else None

# 이미지 업로드
uploaded_image = st.file_uploader("인테리어 사진을 업로드하세요", type=['png', 'jpg', 'jpeg'])

if uploaded_image:
    st.image(uploaded_image, caption="업로드한 이미지", use_column_width=True)

    if st.button("스타일 분석 및 RAG 기반 리모델링 추천"):
        if not openai_api_key:
            st.warning("OpenAI API 키를 입력해주세요!")
        else:
            with st.spinner("이미지 분석 및 문서 참조 중입니다..."):
                
                # 이미지 Base64 인코딩
                encoded_image = base64.b64encode(uploaded_image.getvalue()).decode('utf-8')

                # GPT-4o 이미지 분석 요청
                styles = [
                    "Modern", "Contemporary", "Rustic", "Scandinavian", "Industrial",
                    "Hygge", "Sustainable", "Retro", "Zen", "Brutalism", "Mediterranean",
                    "Oriental", "Shabby", "Provence", "Cottage core", "Victorian",
                    "Tudor", "French", "Neo-classic", "Bohemian", "Art Nouveau",
                    "Maximalist", "Kinfolk", "Eclectic", "Junk"
                ]

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "당신은 인테리어 디자인 전문가입니다."},
                        {"role": "user", "content": [
                            {"type": "text", "text": f"""
                                이 이미지를 보고 가장 적합한 스타일을 고르고 이유를 상세히 설명하세요.
                                스타일 리스트: {', '.join(styles)}
                                또한 색상, 가구 배치, 스타일에 대한 간단한 평가도 추가해주세요. 가구에 색상이 있다면 정확하게 표현해주세요.
                            """},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }}
                        ]}
                    ],
                    max_tokens=1000,
                    temperature=0.2,
                )

                image_analysis = response.choices[0].message.content

                st.subheader("🔍 이미지 스타일 분석 결과:")
                st.write(image_analysis)

                # RAG 문서 로딩
                pdf_files = glob.glob("data/*.pdf")
                all_docs = []
                for pdf_file in pdf_files:
                    loader = PyPDFLoader(pdf_file)
                    docs = loader.load()
                    all_docs.extend(docs)

                # 텍스트 분할 및 벡터DB 생성
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                texts = text_splitter.split_documents(all_docs)
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="vectordb")

                # RAG Chain 생성
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=openai_api_key),
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                    memory=memory
                )

                # 이미지 분석 결과를 기반으로 RAG 평가 및 추천
                rag_prompt = f"""
                다음은 한 인테리어 이미지에 대한 GPT-4o의 분석 결과입니다:
                {image_analysis}

                위 분석 내용을 바탕으로 인테리어 전문가의 관점에서 더욱 상세하게 평가하고, 개선이 필요한 부분을 구체적으로 제안해 주세요. 
                특히 인테리어 문서에서 참조한 전문가적 의견이나 스타일, 가구 배치, 색상에 대한 추가적인 조언을 포함해주세요.
                """

                rag_result = qa_chain({"question": rag_prompt})
                st.subheader("📚 전문가 문서 참조 평가 및 추천:")
                st.write(rag_result['answer'])

                # 상세한 리모델링 방안 추가 요청
                remodel_prompt = f"""
                위의 전문가 평가를 기반으로 이 공간의 리모델링 방안을 더욱 구체적으로 설명해주세요.
                추천하는 인테리어 소품 추천이나 가구 배치 방법, 색상 조합 등을 구체적으로 포함해주세요.
                """

                remodel_result = qa_chain({"question": remodel_prompt})
                st.subheader("✨ 구체적인 리모델링 방안 추천:")
                st.write(remodel_result['answer'])
