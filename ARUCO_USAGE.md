# D435 ArUco/AprilTag Detection Guide

## 개요
RealSense D435 카메라로 ArUco 마커와 AprilTag를 감지하는 스크립트입니다.

## 마커 타입 선택

### AprilTag 사용 (기본값)
AprilTag 36h11 (ID 10 등):
```bash
python3 ds435_ar_tag_detect.py
# 또는
python3 ds435_ar_tag_detect.py --dict DICT_APRILTAG_36h11
```

다른 AprilTag 딕셔너리:
```bash
# AprilTag 16h5
python3 ds435_ar_tag_detect.py --dict DICT_APRILTAG_16h5

# AprilTag 25h9
python3 ds435_ar_tag_detect.py --dict DICT_APRILTAG_25h9

# AprilTag 36h10
python3 ds435_ar_tag_detect.py --dict DICT_APRILTAG_36h10
```

### ArUco 마커 사용
6x6 ArUco 마커:
```bash
python3 ds435_ar_tag_detect.py --dict DICT_6X6_250
```

4x4 ArUco 마커:
```bash
python3 ds435_ar_tag_detect.py --dict DICT_4X4_50
```

다른 ArUco 딕셔너리:
```bash
# 5x5 마커
python3 ds435_ar_tag_detect.py --dict DICT_5X5_100

# 7x7 마커
python3 ds435_ar_tag_detect.py --dict DICT_7X7_50

# Original ArUco
python3 ds435_ar_tag_detect.py --dict DICT_ARUCO_ORIGINAL
```

## 마커 크기 설정

마커 크기 변경 (미터 단위):
```bash
# 20mm 마커 (기본값)
python3 ds435_ar_tag_detect.py --size 0.02

# 30mm 마커
python3 ds435_ar_tag_detect.py --size 0.03

# 50mm 마커
python3 ds435_ar_tag_detect.py --size 0.05
```

## 디버그 모드

디버그 정보 출력:
```bash
python3 ds435_ar_tag_detect.py --debug
```

## 전체 옵션 예제

AprilTag 36h11, 20mm, 디버그 모드:
```bash
python3 ds435_ar_tag_detect.py --dict DICT_APRILTAG_36h11 --size 0.02 --debug
```

ArUco 6x6, 30mm:
```bash
python3 ds435_ar_tag_detect.py --dict DICT_6X6_250 --size 0.03
```

## 실행 중 컨트롤

- `d` 키: 디버그 모드 토글 (감지 통계 표시/숨김)
- `q` 또는 `ESC` 키: 프로그램 종료

## 지원되는 딕셔너리

### AprilTag
- `DICT_APRILTAG_16h5` - AprilTag 16h5 family
- `DICT_APRILTAG_25h9` - AprilTag 25h9 family
- `DICT_APRILTAG_36h10` - AprilTag 36h10 family
- `DICT_APRILTAG_36h11` - AprilTag 36h11 family ⭐ (기본값)

### ArUco
- `DICT_4X4_50` - 4x4 마커, 50개 ID
- `DICT_5X5_100` - 5x5 마커, 100개 ID
- `DICT_6X6_250` - 6x6 마커, 250개 ID
- `DICT_7X7_50` - 7x7 마커, 50개 ID
- `DICT_ARUCO_ORIGINAL` - Original ArUco 마커

## 현재 설정 확인

### 캘리브레이션 정보
- 카메라: Intel RealSense D435
- Serial Number: 207522071359
- 해상도: 640x480
- Reprojection Error: 0.208

### 현재 사용 중인 마커
- 타입: **AprilTag 36h11**
- ID: **10**
- 크기: **20mm**

## 트러블슈팅

### 마커가 감지되지 않을 때
1. 올바른 딕셔너리를 사용하고 있는지 확인
   - AprilTag → `DICT_APRILTAG_*` 사용
   - ArUco → `DICT_*X*_*` 사용

2. 마커 크기가 정확한지 확인
   - 실제 마커 크기를 측정하여 `--size` 옵션에 정확히 입력

3. 조명 조건 개선
   - 균일한 조명
   - 반사 최소화

4. 디버그 모드로 확인
   ```bash
   python3 ds435_ar_tag_detect.py --debug
   ```
   - Rejected candidates가 많다면 잘못된 딕셔너리 사용 중

## 마커 생성

새로운 마커가 필요하면:
```bash
python3 generate_aruco_markers.py
```

## 도움말

전체 옵션 보기:
```bash
python3 ds435_ar_tag_detect.py --help
```
