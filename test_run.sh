#!/bin/bash
# SSM 빠른 테스트 스크립트
# 샘플 데이터로 SSM이 정상 동작하는지 확인

set -e

echo "=========================================="
echo "SSM 최적화 버전 테스트 시작"
echo "=========================================="

# 1. 환경 확인
echo ""
echo "[1/4] 패키지 설치 확인..."
python -c "
import rdkit
import networkx as nx
import sklearn
import pysmiles

# networkx 버전 체크
if int(nx.__version__.split('.')[0]) >= 3:
    print(f'✗ networkx {nx.__version__} 검출 - 2.x 필요 (README 참고)')
    exit(1)

print(f'✓ rdkit {rdkit.__version__}')
print(f'✓ networkx {nx.__version__}')
print(f'✓ sklearn {sklearn.__version__}')
print(f'✓ 모든 패키지 설치됨')
" || {
    echo "✗ 패키지 설치 실패. README의 Installation 참고"
    exit 1
}

# 2. 샘플 데이터 확인
echo ""
echo "[2/4] 샘플 데이터 확인..."
if [ ! -f "examples/example_train.csv" ] || [ ! -f "examples/example_test.csv" ]; then
    echo "✗ 샘플 데이터 파일 없음"
    exit 1
fi
echo "✓ examples/example_train.csv ($(wc -l < examples/example_train.csv) lines)"
echo "✓ examples/example_test.csv ($(wc -l < examples/example_test.csv) lines)"

# 3. SSM 실행 (빠른 테스트: iterations 3, DiSC 2)
echo ""
echo "[3/4] SSM 실행 중 (약 30초 소요)..."
python src/ssm_smiles.py \
  --train_data examples/example_train.csv \
  --test_data examples/example_test.csv \
  --output_dir ./example_results \
  --iterations 3 \
  --DiSC 2 \
  --n_jobs 2

# 4. 결과 확인
echo ""
echo "[4/4] 결과 확인..."
if [ -f "example_results/performance.csv" ]; then
    echo "✓ 실행 완료!"
    echo ""
    echo "=========================================="
    echo "성능 지표 (performance.csv):"
    echo "=========================================="
    cat example_results/performance.csv
    echo ""
    echo "=========================================="
    echo "생성된 파일:"
    echo "=========================================="
    ls -lh example_results/
    echo ""
    echo "✓ 테스트 성공! 실제 데이터로 실행하려면 README 참고"
else
    echo "✗ 결과 파일 생성 실패"
    exit 1
fi
