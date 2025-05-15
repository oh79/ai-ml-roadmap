## 0~12주: 기초→Hugging Face 파인튜닝 단계

| 주차 | 학습 테마 | 핵심 활동 & 추천 리소스 | 주간 산출물 | A100 GPU 활용 예시 |
| --- | --- | --- | --- | --- |
| **준비주(Week 0)** | 로드맵 세팅 & 환경 구축 | • 노션/깃허브 백로그 템플릿 정리• 코랩(Pro) → KT Cloud A100 2대 세팅• CUDA 11.x & cuDNN 설치• Conda 환경 생성 (PyTorch, HF, FAISS 등) | ✔ 학습 저널 초안 | • 두 대 GPU가 정상 인식되는지 `nvidia-smi` 확인 |
| **1주차** | 기초 수학 복습 (선형대수·확률통계) | • Coursera ML Specialization (Week 1–2) 시청• Khan Academy 선형대수 15개 문제 풀이 | ✔ 수학 핸드노트, 확률분포 요약 | • N/A (CPU 중심) |
| **2주차** | Python 데이터 핸들링 | • NumPy 100제 풀기• Kaggle “Python” 미니 코스 완료• Pandas EDA 패턴 정리 | ✔ Titanic EDA 노트북 | • N/A |
| **3주차** | PyTorch 기초 & 로지스틱 회귀 | • PyTorch 60-Minute Blitz 실습• Kaggle Titanic Logistic Regression(PyTorch) 코드 재현 | ✔ PyTorch Titanic 노트북, 학습곡선 그래프 | • 모델·데이터 `.to('cuda')`로 옮겨 GPU에서 훈련• batch_size 2× 증가 후 속도 비교 |
| **4주차** | 자동미분·GPU 사용·MSE Loss | • `nn.Module` 커스텀 구현• CUDA Tensor 전송 실습• Huber vs MSE Loss 성능 비교 | ✔ MLP + TensorBoard 로그 | • `torch.nn.DataParallel`으로 multi-GPU 실험• 추론속도/메모리 사용 차트 생성 |
| **5주차** | Transformer 이론 | • “Attention Is All You Need” 논문 정독• Stanford CS25 ‘LLM Overview’ 강의 시청 | ✔ 논문 요약 5문단, Self-Attention 계산 엑셀 시트 | • N/A |
| **6주차** | Tiny-Transformer 구현 | • `harvard-nlp/annotated-transformer` 코드 리뷰• Encoder 1 layer로 toy 번역 실습• BLEU 측정 & 파라미터 실험 | ✔ Tiny Transformer Colab, 실험 리포트 | • 토큰화·행렬곱 연산을 A100 GPU로 가속• `.to('cuda')`로 모델 옮겨 대용량 배치 실험 |
| **7주차** | Hugging Face Transformers 기본 | • HF “Transformers Course” Chapter 1–3 (모델 사용, 디코더-인코더 이해) https://huggingface.co/course/chapter1• 사전학습 BERT 문장 분류 예제 실습 | ✔ Transformers 기본 노트북, 분류 예제 노트북 | • GPU로 토큰화·인퍼런스 배치 처리 (batch_size 16↑) |
| **8주차** | Datasets & Tokenizers | • HF “Datasets Course” Chapter 1–2 (데이터 로딩·전처리) https://huggingface.co/course/chapter2• 한국어 KoBERT 토크나이저 커스터마이징 & 분포 시각화 | ✔ 전처리 노트북, 토크 분포 히스토그램 | • 대용량 CSV→Dataset 변환 시 `num_proc=4` 멀티프로세스 + GPU I/O 가속 |
| **9주차** | KoBERT 파인튜닝 & 실험 | • `monologg/kobert` 감정 분석 파인튜닝 (HF Trainer) • Learning Rate·Batch Size 실험• 정확도·F1 평가 | ✔ KoBERT 파인튜닝 노트북, 하이퍼파라미터 리포트 | • `accelerate launch --num_processes 2 --mixed_precision fp16` 으로 DDP 파인튜닝• 2대 GPU 활용해 batch_size 8→16 확장 실험 |
| **10주차** | Llama2 LoRA & 양자화 | • HF “LLM Course” Chapter 4 (LoRA, 8bit 양자화) https://huggingface.co/course/chapter4• Llama2 7B LoRA 튜닝 후 뉴스 요약 실습• `bitsandbytes` 양자화 메모리·속도 비교 | ✔ LoRA 튜닝 스크립트, 파라미터 용량·추론속도 표 | • Multi-GPU DDP + LoRA 결합 훈련: `--num_processes 2`• 8bit 양자화 모델 FP16 추론 속도 측정 |
| **11주차** | 사용자 정의 Inference Pipeline | • HF `pipeline` API로 입력→모델→후처리 파이프라인 작성• Rouge·BLEU 자동 평가 스크립트 작성 | ✔ Pipeline 예제 `.py`, 평가 스크립트 노트북 | • GPU로 대량 텍스트 배치 추론 성능 비교 |
| **12주차** | **미니 프로젝트 1**: 한국어 요약 챗봇 제작 | • KoBERT 요약 또는 Llama2 LoRA 요약 모델 배치 스크립트 작성• FastAPI `/summarize` 엔드포인트 구현• Streamlit UI 연동 | ✔ 요약 챗봇 레포, 데모 영상 링크 | • FastAPI 서비스에 Triton Inference로 A100 2대 연결• FP16 추론 모드로 응답 지연 최소화 |

## 3~4 단계 (12 ~ 24 주) ― “프롬프트→RAG→배포·MLOps” + A100 GPU 활용

| 주차 | 학습 테마 | 핵심 활동 & 추천 리소스 | 주간 산출물 | A100 GPU 활용 예시 |
| --- | --- | --- | --- | --- |
| **13 주차** | 프롬프트 엔지니어링 기초 | • OpenAI Prompting Guide  https://platform.openai.com/docs/guides/prompting• DeepLearning.ai Generative AI with LLMs Module 2  https://www.deeplearning.ai/courses/generative-ai-with-llms | ✔ 프롬프트 카탈로그 (패턴별 예시 20개) | • N/A (GPU 불필요) |
| **14 주차** | LangChain 기본 체인 | • LangChain 튜토리얼  https://python.langchain.com/en/latest/getting_started.html• Chain·Tool 개념 이해 및 체인 간단 구현 | ✔ `SimpleChain.ipynb` | • N/A |
| **15 주차** | LangChain 에이전트 & 도구 통합 | • LangChain Agents 예제 따라하기 (ToolAgent)• GitHub `hwchase17/langchain` 예제 코드 리뷰• OpenAI API 키 연결 후 Agent에 Tool 등록 실습 | ✔ `AgentDemo.ipynb` | • 프롬프트→모델 호출 시 GPU 디바이스(`device='cuda'`) 지정• 배치 토큰 수 늘려 인퍼런스 스루풋 측정 |
| **16 주차** | 자체 RAG 도구 준비 (FAISS & 벡터 DB) | • Python FAISS 튜토리얼  https://github.com/facebookresearch/faiss/wiki/Python-Integration• Pinecone RAG 개요  https://www.pinecone.io/learn/embeddings-rag-langchain/ | ✔ FAISS 인덱스 생성 스크립트 | • `model.encode(..., device='cuda')` 로 대량 문서 임베딩• 두 GPU 각각에 파티셔닝하여 동시 임베딩 |
| **17 주차** | RAG 툴 LangChain 연동 | • LangChain RetrievalQA 예제  https://python.langchain.com/en/latest/modules/chains/combine_docs_examples/retrieval_qa.html• Pinecone 무료 계정 테스트 | ✔ RetrievalQA 코드 + 테스트 결과 | • 임베딩→벡터 검색→LLM 생성 연속 파이프라인에서 GPU×2로 병렬화• `faiss.index_cpu_to_all_gpus()` 로 GPU 인덱스 생성 |
| **18 주차** | **미니 프로젝트 2**: PDF QA 챗봇 | • PDF 업로드→텍스트 추출→FAISS/Pinecone 인덱싱→LangChain Agent 질문답변• Streamlit/Gradio UI 배포 | ✔ PDF QA 챗봇 레포 + 데모 영상 링크 | • 문서 임베딩(batch) 및 LLM 응답(batch) 단계에서 멀티-GPU 처리• FP16 추론 활성화로 응답 지연 감소 |
| **19 주차** | 고급 RAG 최적화 | • Pinecone RAG 가이드  https://docs.pinecone.io/docs/rag• DL.ai Advanced RAG course (3h)  https://courses.deeplearning.ai/advanced-rag | ✔ 검색 지연 측정 & 최적화 리포트 | • GPU 인덱스 빌드(`IndexIVFPQ` GPU 버전)로 수십 분→수 분 단축• 대규모 쿼리(100개 배치) 벡터 검색 GPU 가속 |
| **20 주차** | End-to-End RAG 챗봇 | • Pinecone+LangChain 공식 예제  https://www.pinecone.io/learn/embeddings-rag-langchain• Cron/Airflow 자동 갱신 스케줄러 설계 | ✔ 완성형 RAG 챗봇 레포 + 데모 | • 멀티-GPU 인덱스 & LLM 추론 서버 구성 → 동시 요청 처리량 2배↑ |
| **21 주차** | FastAPI 배포 & Docker화 | • FastAPI Deployment (Docker) 가이드  https://fastapi.tiangolo.com/deployment/docker/• Docker Compose로 모델 서비스+벡터 DB 컨테이너 오케스트레이션 | ✔ `docker-compose.yml` + 배포 스크립트 | • NVIDIA Container Toolkit 사용해 A100 GPU 컨테이너 마운트• 서비스 컨테이너×2로 로드밸런싱 구성 |
| **22 주차** | CI/CD & IaC | • GitHub Actions 튜토리얼  https://docs.github.com/en/actions• Terraform/AWS CDK 간단 IaC 튜토리얼 | ✔ `.github/workflows/ci.yml` | • GPU 인스턴스 프로비저닝 Terraform 모듈 테스트 (예: AWS EC2 P4d) |
| **23 주차** | MLOps 모니터링 & 실험 관리 | • MLOps Zoomcamp 튜토리얼  https://github.com/DataTalksClub/mlops-zoomcamp• Prometheus Python Client  https://github.com/prometheus/client_python | ✔ Prometheus 대시보드 스크린샷 | • `nvidia_smi_exporter` 설치 후 GPU 메모리·사용률 모니터링• Grafana로 A100 클러스터 대시보드 구성 |
| **24 주차** | **최종 캡스톤**: 특허 RAG SaaS MVP 배포 | • 프로젝트 1 MVP RAG 기능 + 웹 UI 통합• AWS ECS/EKS 배포, HTTPS 로드밸런싱 테스트• 사용자 초대 및 알파테스터 피드백 수집 | ✔ SaaS MVP URL + 사용자 가이드✔ 베타테스트 보고서 | • KT Cloud A100 GPU 2대 클러스터로 실서비스 운영• 부하 테스트 시 GPU×2 병렬 추론으로 동시 100TPS 이상 처리 |