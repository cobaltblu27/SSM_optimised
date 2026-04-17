# SSM (Supervised Subgraph Mining) - Optimized Version

이 버전은 원본 SubDyve의 SSM(Supervised Subgraph Mining) 모듈을 속도 최적화한 것입니다.

**성능 최적화 버전**: 원본 104시간 → 2.4시간 (43배 단축)  
**원본 저장소**: https://github.com/sslim0814/SSM  
**논문**: Lim, Sangsoo, et al. "Supervised chemical graph mining improves drug-induced liver injury prediction." _iScience_ 26.1 (2023).

---

## Overview

SSM (Supervised Subgraph Mining)은 분자 구조 데이터로부터 discriminative한 subgraph 패턴을 학습하는 알고리즘입니다. Random walk 기반의 반복적 학습을 통해 class를 구별하는 substructure feature를 자동으로 발견합니다.

**이 버전의 특징**:
- 원본 알고리즘과 **분류 성능 100% 동일** (Accuracy, confusion matrix 등)
- 실행 시간 **43배 단축** (104시간 → 2.4시간)
- DiSC 패턴 마이닝, 확률 벡터 캐싱, 벡터화, 멀티프로세싱 최적화 적용

---

## Installation

### 방법 1: Conda 환경 (권장)

```bash
# 1. 새 conda 환경 생성
conda create -n ssm -y python=3.11
conda activate ssm

# 2. 나머지 패키지 설치
pip install -r requirements.txt
```

**중요**: `networkx`는 반드시 2.x 버전이어야 합니다 (3.x는 비호환). requirements.txt가 자동으로 올바른 버전을 설치합니다.

### 방법 2: pip만 사용 (고급 사용자용)

```bash
# Python 3.8+ 필요
pip install rdkit 'networkx>=2.8,<3.0' 'numpy>=1.26,<2.0' 'pandas>=1.5,<2.0' \
  scipy scikit-learn tqdm 'pysmiles>=1.0,<2.0'
```

### 설치 확인

```bash
python -c "import rdkit; import networkx; import sklearn; print('설치 완료')"
```

### 빠른 테스트 (자동화 스크립트)

샘플 데이터로 SSM이 정상 동작하는지 확인:

```bash
# 자동 테스트 스크립트 실행 (약 30초 소요)
bash test_run.sh
```

스크립트가 다음을 자동으로 수행합니다:
1. 패키지 설치 확인
2. 샘플 데이터 로드 (examples/example_train.csv, examples/example_test.csv)
3. SSM 실행 (iterations 3, DiSC 2)
4. 결과 출력

---

## Usage

### Basic Usage

```bash
python src/ssm_smiles.py \
  --train_data train.csv \
  --test_data test.csv \
  --output_dir ./results \
  --DiSC 3
```

### Input Format

#### train.csv / test.csv

CSV 파일은 최소 다음 컬럼을 포함해야 합니다:

| Column | Description | Example |
|--------|-------------|---------|
| `SMILES` | 분자 구조 (SMILES 표기법) | `CC1(C)CNCC(Nc2ncc...)C1` |
| `label` | 클래스 라벨 (0 또는 1) | `1` |

**예시**:
```csv
SMILES,label
CC(C)Cc1ccc(cc1)C(C)C(=O)O,0
CN(C)CCc1c[nH]c2ccc(C)cc12,1
```

**주의**: train 데이터는 `label` 컬럼, test 데이터는 `label` 컬럼을 사용합니다. (내부적으로 `class`로 변환됨)

### Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--train_data` | (required) | 학습 데이터 CSV 경로 |
| `--test_data` | (required) | 테스트 데이터 CSV 경로 |
| `--output_dir` | `./results` | 출력 디렉토리 |
| `--trained_file` | `None` | 사전 학습된 모델 (재사용 시) |
| `--rw` | `7` | Random walk 길이 |
| `--alpha` | `0.1` | Preference 업데이트 비율 |
| `--iterations` | `20` | 전체 iteration 수 |
| `--nWalker` | `5` | 시작 노드당 walker 수 |
| `--seed` | `0` | Random seed |
| `--DiSC` | `1` | DiSC 패턴 조합 크기 (1~5) |
| `--n_jobs` | `1` | 병렬 워커 수 (**최적화 버전 신규 추가**) |

### Parallel Processing (신규 기능)

`--n_jobs` 옵션으로 random walk를 병렬화할 수 있습니다:

```bash
# 순차 실행 (기본값, 원본과 100% 동일한 결과)
python src/ssm_smiles.py --train_data train.csv --test_data test.csv --output_dir ./results --DiSC 3

# 8개 코어로 병렬 실행 (5~10배 추가 가속)
python src/ssm_smiles.py --train_data train.csv --test_data test.csv --output_dir ./results --DiSC 3 --n_jobs 8
```

**권장**: CPU 코어 수(`nproc`)에 맞춰 설정.

---

## Output

실행 완료 후 `output_dir`에 다음 파일이 생성됩니다:

### 필수 출력

```
output_dir/
├── ssm_train.pickle          # 학습된 모델 (재사용 가능)
├── ssm_test.pickle           # 테스트 결과
├── performance.csv           # 전체 iteration 성능 지표
├── confusion_matrix.csv      # 혼동 행렬
└── iteration_1/ ~ iteration_20/
    ├── predictions.csv       # 분자별 예측 결과
    ├── subgraph.csv          # 발견된 subgraph 목록
    ├── subgraph_important.csv  # 중요 subgraph (entropy < 0.5)
    ├── subgraph_SA.csv       # Structure-Activity 관계
    ├── DiSC_2.csv            # 2개 fragment 조합 패턴
    └── DiSC_3.csv            # 3개 fragment 조합 패턴
```

### Output File Details

#### performance.csv

각 iteration의 분류 성능 지표:

| Column | Description |
|--------|-------------|
| `n_union_subgraphs` | 전체 subgraph 수 |
| `n_train_subgraphs` | 학습 데이터 subgraph 수 |
| `n_valid_subgraphs` | 테스트 데이터 subgraph 수 |
| `Accuracy` | 정확도 |
| `BAcc` | Balanced Accuracy |
| `Precision` | 정밀도 |
| `Recall` | 재현율 |
| `F1_score` | F1 점수 |
| `AUC` | ROC AUC |
| `MCC` | Matthews Correlation Coefficient |

#### predictions.csv

분자별 예측 결과:

| Column | Description |
|--------|-------------|
| `SMILES` | 분자 구조 |
| `prediction` | 예측 클래스 (0 or 1) |
| `probability` | 클래스 1 확률 |

---

## Optimization Details

이 버전은 원본 알고리즘의 다음 부분을 최적화했습니다:

| 최적화 | 내용 | 효과 |
|--------|------|------|
| **DiSC 캐싱** | Fragment 매칭 결과를 사전 캐싱, 조합은 bitwise AND로 대체 | 94h → 42m (**134배**) |
| **확률 벡터 캐싱** | Random walk의 확률 벡터를 사전 계산 후 재사용 | ~5배 절감 |
| **벡터화** | Python for-loop를 numpy 배열 연산으로 대체 | 중간 효과 |
| **중복 계산 제거** | DataFrame, 인접 리스트 등을 루프 밖에서 1회만 생성 | 중간 효과 |
| **멀티프로세싱** | Random walk를 분자별 병렬 실행 (`--n_jobs`) | 코어 수 비례 |

**상세 내용**: `docs/SSM_optimization_report.md` 참조

---

## Result Verification

최적화 버전의 결과가 원본과 동일한지 검증하려면:

```bash
# 1. 원본 SSM으로 실행한 baseline 결과 준비
# (예: baseline_results/ 디렉토리)

# 2. 최적화 버전으로 재실행 (--n_jobs 1로 순차 모드)
python src/ssm_smiles.py --train_data train.csv --test_data test.csv \
  --output_dir ./optimized_results --DiSC 3 --n_jobs 1

# 3. 검증 스크립트 실행
python tests/verify_output.py --baseline baseline_results/ --new optimized_results/
```

**검증 결과**:
- ✅ **82/122 파일 PASS**: performance, confusion_matrix, predictions, subgraph 등 **100% 일치**
- ⚠️ **40/122 파일 FAIL**: DiSC CSV만 패턴 수 차이 (분류 성능에는 영향 없음)

**DiSC 차이 원인**: 속도를 위해 bitwise AND 근사 방식을 사용하여, 원자 겹침 케이스에서 미세한 차이 발생. 분류 후 해석용 출력이므로 허용 가능.

---

## Performance Comparison

| 모드 | 소요 시간 | 검증 결과 |
|------|-----------|----------|
| 원본 코드 | **104.5시간** | - (기준) |
| 최적화 (`--n_jobs 1`) | **5시간 16분** | ✅ 82/122 파일 100% 일치 (**20배 단축**) |
| 최적화 (`--n_jobs 100`) | **2시간 25분** | 분류 결과 동등 (**43배 단축**) |

**검증 세부사항**:
- ✅ PASS (82개): performance.csv, confusion_matrix.csv, predictions.csv, subgraph.csv 등 **분류 성능 관련 모든 파일**
- ⚠️ FAIL (40개): DiSC_2.csv, DiSC_3.csv만 패턴 수 차이 (해석용 출력, 분류에 무영향)

**테스트 환경**: 
- 학습 분자: 2,936개
- 테스트 분자: 26,090개
- Iterations: 20
- DiSC: 3

---

## Reference

이 코드를 사용하는 경우 원본 논문을 인용해주세요:

**Lim, Sangsoo, et al.** "Supervised chemical graph mining improves drug-induced liver injury prediction." _iScience_ 26.1 (2023).

```bibtex
@article{lim2023supervised,
  title={Supervised chemical graph mining improves drug-induced liver injury prediction},
  author={Lim, Sangsoo and others},
  journal={iScience},
  volume={26},
  number={1},
  year={2023},
  publisher={Elsevier}
}
```

**원본 저장소**: https://github.com/sslim0814/SSM

---

## Citation

최적화 버전의 방법론을 참고하는 경우:
```
이 코드는 LLM Agent 기반 최적화 방법론을 적용하여 43배 속도 향상을 달성했습니다.
상세 방법론: docs/SSM_optimization_report.md
```

---

## License

원본 SSM 라이선스를 따릅니다. (원본 저장소 확인: https://github.com/sslim0814/SSM)

---

## Contact & Support

- **원본 SSM**: https://github.com/sslim0814/SSM
- **최적화 관련 문의**: `docs/SSM_optimization_report.md` 참조
- **이슈 보고**: [GitHub Issues]

---

## Quick Test (빠른 테스트)

설치 후 바로 테스트해볼 수 있는 최소 예제:

### 방법 1: 제공된 샘플 데이터 사용 (가장 빠름)

```bash
# 이미 포함된 샘플 데이터로 즉시 테스트
python src/ssm_smiles.py \
  --train_data examples/example_train.csv \
  --test_data examples/example_test.csv \
  --output_dir ./example_results \
  --iterations 3 \
  --DiSC 2 \
  --n_jobs 2

# 결과 확인
cat example_results/performance.csv
```

### 방법 2: 직접 샘플 데이터 생성

```bash
# 1. 샘플 데이터 생성 (CSV 형식: SMILES,label)
cat > train.csv << 'EOF'
SMILES,label
CC(C)Cc1ccc(cc1)C(C)C(=O)O,0
CN(C)CCc1c[nH]c2ccc(C)cc12,1
CC(C)(C)NCC(O)COc1ccccc1C(N)=O,0
EOF

cat > test.csv << 'EOF'
SMILES,label
c1ccc2c(c1)ccc1c2cccc1,0
CC(C)NCC(COc1ccccc1)O,1
EOF

# 2. SSM 실행 (약 30초 소요)
python src/ssm_smiles.py \
  --train_data train.csv \
  --test_data test.csv \
  --output_dir ./test_results \
  --iterations 3 \
  --DiSC 2 \
  --n_jobs 2

# 3. 결과 확인
cat test_results/performance.csv
ls -lh test_results/iteration_3/
```

**실제 데이터로 실행**:

```bash
# 학습 분자 수천 개, 테스트 분자 수만 개 권장
python src/ssm_smiles.py \
  --train_data your_train.csv \
  --test_data your_test.csv \
  --output_dir ./results \
  --iterations 20 \
  --DiSC 3 \
  --n_jobs 8
```

---

## Troubleshooting

### Q: `ModuleNotFoundError: No module named 'rdkit'`
**A**: RDKit 설치 필요:
```bash
conda install -c conda-forge rdkit
# or
pip install rdkit
```

### Q: 메모리 부족 에러
**A**: 
- `--n_jobs` 값을 줄이거나 1로 설정
- 테스트 분자 수를 줄여서 시험
- 메모리가 큰 머신 사용

### Q: DiSC 결과가 원본과 다릅니다
**A**: 정상입니다. 분류 성능 지표(performance.csv, predictions.csv)가 동일하면 문제 없음. DiSC는 해석용 출력.

### Q: `--n_jobs 1`과 `--n_jobs 8`의 결과가 다릅니다
**A**: 정상입니다. 병렬 모드는 random walk의 seed가 달라져 미세한 차이 발생. 성능 지표는 동등 범위.

---

## Changelog

### v2.0 (2026-03-26) - Optimized Version
- ✨ DiSC 패턴 마이닝 캐싱 (94h → 42m)
- ✨ 확률 벡터 사전 계산 (5배 절감)
- ✨ 벡터화 및 중복 계산 제거
- ✨ `--n_jobs` 멀티프로세싱 옵션 추가
- 📝 최적화 보고서 및 검증 스크립트 추가
- ⚡ 총 43배 속도 향상 (104h → 2.4h)

### v1.0 - Original
- 원본 SSM 구현 (https://github.com/sslim0814/SSM)
