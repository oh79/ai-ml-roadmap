# 00\_env\_setup/env\_setup.md

## 🖥️ AI/ML 실습 환경 세팅 가이드

본 문서는 AI/ML 엔지니어 로드맵 실습을 위한 환경 세팅 절차를 정리합니다.
(Windows 기준, Mac/Linux는 커맨드만 약간 다름)

---

### 1. 아나콘다(Anaconda) 설치

* [Anaconda 공식 다운로드](https://www.anaconda.com/products/distribution)
* 설치 완료 후, 명령 프롬프트(cmd/터미널)에서 아래 명령 실행

---

### 2. 가상환경 생성 및 활성화

```bash
conda create -n mlroadmap python=3.9
conda activate mlroadmap
```

---

### 3. 필수 패키지 설치

#### (1) PyTorch + CUDA (권장: conda)

> CUDA 12.9 드라이버 환경에서 PyTorch는 12.1/12.8 등 하위버전도 동작합니다.

```bash
# CUDA 12.1 (안정적)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### (2) 기타 ML/데이터 분석 패키지

```bash
conda install numpy pandas scikit-learn matplotlib jupyter
```

필요시:

```bash
pip install transformers datasets seaborn openpyxl
```

---

### 4. 설치 확인

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import numpy; print(numpy.__version__)"
```

* `torch.cuda.is_available()`가 `True`면 GPU 연동 OK

---

### 5. Jupyter Notebook 실행

```bash
jupyter notebook
```

브라우저에서 새 노트북 생성, 실습 시작!

---

### 6. (옵션) Colab Pro 환경 병행 사용

* [Colab 바로가기](https://colab.research.google.com/)
* Google 계정으로 로그인 후 GPU/TPU 런타임 설정

---

### 7. 환경 재현/공유 팁

* 패키지 내역 저장:

  ```bash
  conda list --explicit > requirements.txt
  ```
* 환경 내보내기:

  ```bash
  conda env export > environment.yml
  ```
* 팀원/다른 환경에서 복구:

  ```bash
  conda env create -f environment.yml
  ```

---

![환경 세팅 결과](assets\results_photo\스크린샷 2025-05-16 010513.png)


## Troubleshooting

* 설치 중 에러 발생 시, 터미널 메시지와 함께 질문!
* 가상환경 삭제/재생성으로 빠르게 문제 해결 가능:

  ```bash
  conda deactivate
  conda remove -n mlroadmap --all
  ```

---

## 참고 자료

* [PyTorch 설치 가이드](https://pytorch.org/get-started/locally/)
* [Anaconda 공식문서](https://docs.anaconda.com/)
* [Kaggle](https://www.kaggle.com/)
* [Coursera](https://www.coursera.org/)

---

> 이 파일은 주기적으로 업데이트하며, 실습 환경에 맞는 변경 사항을 기록합니다.

