"""
SSM 최적화 검증 스크립트

사용법:
    python tests/verify_output.py \
        --baseline test_run/3_intermediate/ssm_result \
        --new ./ssm_result

검증 항목:
    1. performance.csv — 전체 iteration 성능 지표 (Accuracy, AUC 등)
    2. confusion_matrix.csv — 혼동 행렬
    3. iteration_N/predictions.csv — 분자별 예측 결과
    4. iteration_N/subgraph.csv — subgraph entropy/importance
    5. iteration_N/DiSC_k.csv — DiSC 패턴

--n_jobs 1 (기본값)으로 실행한 결과는 기존과 100% 동일해야 함.
--n_jobs > 1은 random walk의 random seed가 달라지므로 결과가 다를 수 있음.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np


def compare_csv(baseline_path, new_path, name, rtol=0, atol=0):
    """두 CSV 파일을 비교. 숫자 컬럼은 rtol/atol 허용."""
    if not os.path.exists(baseline_path):
        print(f"  [SKIP] {name}: baseline 없음 ({baseline_path})")
        return True
    if not os.path.exists(new_path):
        print(f"  [FAIL] {name}: new 파일 없음 ({new_path})")
        return False

    df_base = pd.read_csv(baseline_path, index_col=0)
    df_new = pd.read_csv(new_path, index_col=0)

    # shape 비교
    if df_base.shape != df_new.shape:
        print(f"  [FAIL] {name}: shape 불일치 — baseline {df_base.shape} vs new {df_new.shape}")
        return False

    # 컬럼 비교
    if list(df_base.columns) != list(df_new.columns):
        print(f"  [FAIL] {name}: 컬럼 불일치")
        print(f"         baseline: {list(df_base.columns)[:5]}...")
        print(f"         new:      {list(df_new.columns)[:5]}...")
        return False

    # 인덱스 비교
    if list(df_base.index) != list(df_new.index):
        print(f"  [FAIL] {name}: 인덱스 불일치")
        return False

    # 값 비교
    if rtol == 0 and atol == 0:
        # 완전 일치 비교 (문자열 포함, NaN == NaN으로 처리)
        diff_mask = df_base.ne(df_new)
        # NaN == NaN인 경우 일치로 처리 (IEEE 754에서는 NaN != NaN이지만 데이터 비교 시 동일 취급)
        both_nan = df_base.isna() & df_new.isna()
        diff_mask = diff_mask & ~both_nan
        n_diff = diff_mask.sum().sum()
        if n_diff > 0:
            print(f"  [FAIL] {name}: {n_diff}개 셀 불일치")
            # 첫 몇 개 차이 표시
            for col in diff_mask.columns:
                diff_rows = diff_mask[col]
                if diff_rows.any():
                    idx = diff_rows.idxmax()
                    print(f"         [{idx}, {col}]: baseline={df_base.loc[idx, col]} vs new={df_new.loc[idx, col]}")
                    break
            return False
    else:
        # 숫자 컬럼만 근사 비교
        for col in df_base.columns:
            try:
                base_vals = pd.to_numeric(df_base[col])
                new_vals = pd.to_numeric(df_new[col])
                if not np.allclose(base_vals, new_vals, rtol=rtol, atol=atol, equal_nan=True):
                    max_diff = np.nanmax(np.abs(base_vals - new_vals))
                    print(f"  [FAIL] {name}: 컬럼 '{col}' 수치 불일치 (max diff={max_diff:.2e})")
                    return False
            except (ValueError, TypeError):
                # 문자열 컬럼 — 완전 일치
                if not df_base[col].equals(df_new[col]):
                    print(f"  [FAIL] {name}: 컬럼 '{col}' 문자열 불일치")
                    return False

    print(f"  [PASS] {name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="SSM 결과 검증")
    parser.add_argument("--baseline", required=True, help="기존 결과 디렉토리")
    parser.add_argument("--new", required=True, help="새 결과 디렉토리")
    parser.add_argument("--rtol", type=float, default=0, help="상대 허용 오차 (0=완전일치)")
    parser.add_argument("--atol", type=float, default=0, help="절대 허용 오차 (0=완전일치)")
    args = parser.parse_args()

    baseline = args.baseline
    new = args.new

    if not os.path.isdir(baseline):
        print(f"ERROR: baseline 디렉토리 없음: {baseline}")
        sys.exit(1)
    if not os.path.isdir(new):
        print(f"ERROR: new 디렉토리 없음: {new}")
        sys.exit(1)

    total, passed, failed, skipped = 0, 0, 0, 0

    # 1. 최상위 CSV
    print("\n=== 최상위 파일 ===")
    for fname in ["performance.csv", "confusion_matrix.csv"]:
        total += 1
        result = compare_csv(
            os.path.join(baseline, fname),
            os.path.join(new, fname),
            fname,
            rtol=args.rtol, atol=args.atol
        )
        if result:
            passed += 1
        else:
            failed += 1

    # 2. iteration별 파일
    iterations = sorted([
        d for d in os.listdir(baseline)
        if d.startswith("iteration_") and os.path.isdir(os.path.join(baseline, d))
    ], key=lambda x: int(x.split("_")[1]))

    for it_dir in iterations:
        print(f"\n=== {it_dir} ===")
        base_it = os.path.join(baseline, it_dir)
        new_it = os.path.join(new, it_dir)

        if not os.path.isdir(new_it):
            print(f"  [FAIL] 디렉토리 없음: {new_it}")
            # 해당 iteration의 모든 파일을 fail로 카운트
            n_files = len([f for f in os.listdir(base_it) if f.endswith(".csv")])
            total += n_files
            failed += n_files
            continue

        for fname in sorted(os.listdir(base_it)):
            if not fname.endswith(".csv"):
                continue
            total += 1
            result = compare_csv(
                os.path.join(base_it, fname),
                os.path.join(new_it, fname),
                f"{it_dir}/{fname}",
                rtol=args.rtol, atol=args.atol
            )
            if result:
                passed += 1
            else:
                failed += 1

    # 결과 요약
    print(f"\n{'='*50}")
    print(f"검증 결과: {passed}/{total} PASS, {failed} FAIL")
    if failed == 0:
        print("결론: 기존 결과와 100% 동일합니다.")
    else:
        print("결론: 차이가 발견되었습니다. 위 [FAIL] 항목을 확인하세요.")
    print(f"{'='*50}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
