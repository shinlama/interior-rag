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
from langchain.prompts import PromptTemplate

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
    st.image(uploaded_image, caption="업로드한 이미지", use_container_width=True)

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

                # RAG 문서 로딩 및 메타데이터 추가
                pdf_files = glob.glob("data/*.pdf")
                all_docs = []

                st.write(f"찾은 PDF 파일들: {pdf_files}")  # 디버깅용

                for pdf_file in pdf_files:
                    try:
                        loader = PyPDFLoader(pdf_file)
                        pages = loader.load()
                        st.write(f"{pdf_file}에서 {len(pages)}페이지 로드됨")  # 디버깅용
                        # 각 문서의 메타데이터 명확히 추가
                        for doc in pages:
                            doc.metadata['source'] = pdf_file
                            doc.metadata['page'] = doc.metadata.get('page', 0)
                        all_docs.extend(pages)
                    except Exception as e:
                        st.error(f"PDF 로딩 중 오류 발생: {str(e)}")

                if not all_docs:
                    st.error("PDF 문서를 찾을 수 없거나 로드할 수 없습니다.")
                    st.stop()

                # 텍스트 분할 (최적 split 전략 유지)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
                texts = text_splitter.split_documents(all_docs)
                st.write(f"총 {len(texts)}개의 텍스트 청크 생성됨")  # 디버깅용

                # OpenAI Embeddings로 벡터DB 생성 (Chroma 유지)
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="vectordb")

                # 신뢰성 높은 출처 정보 포함하여 결과 생성
                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 4}
                )

                # 메모리 및 RAG 체인 유지
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=openai_api_key),
                    retriever=retriever,
                    memory=memory,
                    combine_docs_chain_kwargs={"prompt": PromptTemplate(
                        template="""당신은 인테리어 디자인 전문가입니다. 다음 정보를 바탕으로 답변해주세요:

                        참조 문서:
                        {context}

                        질문: {question}

                        답변할 때는 반드시 참조 문서의 내용을 인용하고, 해당 내용이 어떤 문서에서 왔는지 명시해주세요.
                        답변:""",
                        input_variables=["context", "question"]
                    )}
                )

                # 이미지 분석 결과를 바탕으로 RAG 평가 및 추천
                rag_prompt = f"""
                다음은 인테리어 이미지에 대한 분석 결과입니다:
                {image_analysis}

                위 분석 결과를 바탕으로, 참조 문서에서 찾을 수 있는 관련된 인테리어 디자인 원칙과 개선 방안을 제시해주세요.
                특히 다음 항목들을 포함해주세요:
                1. 현재 공간의 스타일과 관련된 전문가의 조언
                2. 색상, 가구 배치, 조명 등에 대한 구체적인 개선 방안
                3. 참조 문서에서 인용한 내용과 해당 문서의 출처
                """

                rag_result = qa_chain({"question": rag_prompt})
                
                st.subheader("📚 전문가 문서 참조 평가 및 추천:")
                st.write(rag_result['answer'])

                # 소스 문서 출처 명확히 표시
                if 'source_documents' in rag_result:
                    st.subheader("📑 참조한 문서 출처:")
                    for doc in rag_result['source_documents']:
                        source = doc.metadata.get('source', 'Unknown')
                        page = doc.metadata.get('page', 'Unknown')
                        st.write(f"📄 **파일명**: {source}, 📃 **페이지**: {page}")
