import cv2
import numpy as np
import os
import math


# 입출력 설정
RESULT_DIR  = "Result"
INPUT_FILES = [
    "EV1.jpg", "EV2.jpg", "EV3.jpg",
    "EV1_Q20.jpg", "EV2_R50_Q70.jpg", "EV3_R65_Q50.jpg",
]

# 파라미터
FFT_KY_BAND     = 3        # FFT 줄무늬 마스크 띠 두께
FFT_DC_KEEP     = 6        # DC 성분 보존 폭
BORDER_PAD      = 30       # 보더 패딩 (BlackHat 경계 응답 약화 방지)
BH_H_KERNEL     = (51, 15) # 가로로 긴 BlackHat 커널
BH_V_KERNEL     = (15, 51) # 세로로 긴 BlackHat 커널
BH_THRESHOLD    = 12       # BlackHat MIN 이진화 임계값
POST_OPEN_K     = 5        # opening 커널 크기
BASE_MIN_AREA   = 60       # 최소 면적 (영상 폭 2592 기준)
BASE_MIN_WH     = 6        # 최소 변 길이 (영상 폭 2592 기준)
MAX_BLOB_AREA   = 4200     # 최대 면적
MAX_BLOB_WH     = 110      # 최대 변 길이
MAX_AABB_ASPECT = 1.8      # 축정렬 종횡비 상한
MIN_CIRCULARITY = 0.40     # 둥근정도 하한 (4πA/p²)
MAX_ROT_ASPECT  = 2.0      # 회전 직사각형 종횡비 상한
MAX_ELONGATION  = 2.5      # 모멘트 elongation 상한
BBOX_PAD        = 4        # overlay bbox 패딩


# 영상 폭에 비례한 최소 면적/변 (리사이즈 변형 대응)
def adaptive_minimums(image_width):
    s = max(0.5, image_width / 2592.0)
    return max(15, int(BASE_MIN_AREA * s * s)), max(3, int(BASE_MIN_WH * s))


# FFT로 세로 줄무늬 주파수 제거
def remove_vertical_stripes_fft(gray):
    g = gray.astype(np.float32)
    Fs = np.fft.fftshift(np.fft.fft2(g))
    H, W = g.shape
    cy, cx = H // 2, W // 2
    mask = np.ones((H, W), np.float32)
    mask[cy - FFT_KY_BAND : cy + FFT_KY_BAND + 1, : cx - FFT_DC_KEEP] = 0
    mask[cy - FFT_KY_BAND : cy + FFT_KY_BAND + 1,   cx + FFT_DC_KEEP :] = 0
    img = np.fft.ifft2(np.fft.ifftshift(Fs * mask))
    return np.clip(np.real(img), 0, 255).astype(np.uint8)


# 가로/세로 비대칭 BlackHat의 element-wise MIN
def blackhat_oriented_min(gray):
    bilat = cv2.bilateralFilter(gray, 7, 35, 35)
    bh_h = cv2.morphologyEx(bilat, cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_RECT, BH_H_KERNEL))
    bh_v = cv2.morphologyEx(bilat, cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_RECT, BH_V_KERNEL))
    bh_min = cv2.min(bh_h, bh_v)
    return bh_min, bilat, bh_h, bh_v


# 시각화용: 응답값을 0~255로 스트레칭
def normalize_for_view(gray):
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# 단계별 출력 저장 (파일명 (n) 번호 prefix로 순서 정렬)
def save_step(out_dir, n, name, image):
    cv2.imwrite(os.path.join(out_dir, f"({n:02d})_{name}.jpg"), image)


# 영역 특징 계산 (강의자료 6주차)
def shape_features(contour):
    area = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    if area < 1e-6 or perim < 1e-6:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    aabb_aspect = max(w, h) / max(1.0, float(min(w, h)))
    circularity = 4.0 * math.pi * area / (perim * perim)
    if len(contour) >= 5:
        rw, rh = cv2.minAreaRect(contour)[1]
        rw, rh = max(float(rw), 1.0), max(float(rh), 1.0)
        rot_aspect = max(rw, rh) / min(rw, rh)
    else:
        rot_aspect = aabb_aspect
    m = cv2.moments(contour)
    if m['m00'] < 1e-6:
        return None
    mu20 = m['mu20'] / m['m00']
    mu02 = m['mu02'] / m['m00']
    mu11 = m['mu11'] / m['m00']
    half_tr = (mu20 + mu02) / 2.0
    disc = math.sqrt(max(0.0, ((mu20 - mu02) / 2.0) ** 2 + mu11 * mu11))
    elongation = math.sqrt((half_tr + disc) / max(1e-6, half_tr - disc))
    return {"area": area, "w": w, "h": h,
            "aabb_aspect": aabb_aspect, "circularity": circularity,
            "rot_aspect": rot_aspect, "elongation": elongation}


# 모양 특징으로 정사각형에 가까운 contour만 채택
def filter_by_shape(binary, min_area, min_wh):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    kept = []
    for c in contours:
        f = shape_features(c)
        if f is None: continue
        if not (min_area <= f["area"] <= MAX_BLOB_AREA): continue
        if not (min_wh  <= f["w"]    <= MAX_BLOB_WH):    continue
        if not (min_wh  <= f["h"]    <= MAX_BLOB_WH):    continue
        if f["aabb_aspect"] > MAX_AABB_ASPECT:           continue
        if f["circularity"] < MIN_CIRCULARITY:           continue
        if f["rot_aspect"]  > MAX_ROT_ASPECT:            continue
        if f["elongation"]  > MAX_ELONGATION:            continue
        kept.append(c)
    return kept


# 검출 contour의 외접 사각형 채움 마스크 (단계 시각화용)
def render_bbox_mask(shape_hw, contours):
    mask = np.zeros(shape_hw, np.uint8)
    H, W = shape_hw
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x0, y0 = max(0, x - BBOX_PAD), max(0, y - BBOX_PAD)
        x1, y1 = min(W, x + w + BBOX_PAD), min(H, y + h + BBOX_PAD)
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    return mask


# 검출된 이물질별 정보 (중심좌표, 면적, 평균 밝기) — Rasterized 순
def extract_defect_info(contours, gray_original, image_height):
    items = []
    for c in contours:
        m = cv2.moments(c)
        if m['m00'] < 1e-6: continue
        cx = int(round(m['m10'] / m['m00']))
        cy = int(round(m['m01'] / m['m00']))
        cmask = np.zeros(gray_original.shape, np.uint8)
        cv2.drawContours(cmask, [c], -1, 255, -1)
        area_px = int(cv2.countNonZero(cmask))
        mean_b  = float(cv2.mean(gray_original, mask=cmask)[0])
        items.append({"cx": cx, "cy": cy, "area_px": area_px, "mean_b": mean_b})
    row_h = max(20, image_height // 30)
    items.sort(key=lambda it: (it["cy"] // row_h, it["cx"]))
    return items


# 이물질 정보를 터미널에 표 형태로 출력
def print_defects_table(source_filename, items):
    print(f"\n[{source_filename}] 검출된 이물질 수: {len(items)}")
    print(f"{'번호':>4}  {'중심 좌표 (x, y)':>18}  {'면적 (px²)':>10}  {'평균 밝기':>10}")
    print("-" * 60)
    for i, it in enumerate(items, 1):
        coord = f"({it['cx']}, {it['cy']})"
        print(f"{i:>4}  {coord:>18}  {it['area_px']:>10}  {it['mean_b']:>10.2f}")


# 원본 위 빨간 테두리 overlay (contour별 직접 그림 — 인접 결함 병합 방지)
def make_overlay(bgr, contours):
    out = bgr.copy()
    H, W = out.shape[:2]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x0, y0 = max(0, x - BBOX_PAD), max(0, y - BBOX_PAD)
        x1, y1 = min(W - 1, x + w + BBOX_PAD), min(H - 1, y + h + BBOX_PAD)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2)
    return out


# 메인 처리
def main():
    # 스크립트 폴더 기준 경로 (어디서 실행해도 동작)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(base_dir, RESULT_DIR)
    os.makedirs(result_dir, exist_ok=True)
    for fn in INPUT_FILES:
        src_path = os.path.join(base_dir, fn)
        if not os.path.isfile(src_path):
            print(f"[skip] {fn} not found"); continue
        bgr = cv2.imread(src_path)
        if bgr is None:
            print(f"[skip] {fn} unreadable"); continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        # 영상별 하위 폴더 (단계별 중간 산출물 저장)
        stem = fn.replace(".jpg", "")
        step_dir = os.path.join(result_dir, stem)
        os.makedirs(step_dir, exist_ok=True)

        # 1) 입력 그레이스케일
        save_step(step_dir, 1, "gray", gray)

        # 2) FFT로 세로 줄무늬 제거
        gray_fft = remove_vertical_stripes_fft(gray)
        save_step(step_dir, 2, "fft", gray_fft)

        # 3) 보더 패딩 (BlackHat 경계 응답 약화 방지)
        gray_pad = cv2.copyMakeBorder(gray_fft, BORDER_PAD, BORDER_PAD,
                                      BORDER_PAD, BORDER_PAD, cv2.BORDER_REFLECT)
        save_step(step_dir, 3, "padded", gray_pad)

        # 4) bilateral → 가로/세로 비대칭 BlackHat MIN
        bh_min, bilat, bh_h, bh_v = blackhat_oriented_min(gray_pad)
        save_step(step_dir, 4, "bilateral",     bilat)
        save_step(step_dir, 5, "bh_horizontal", normalize_for_view(bh_h))
        save_step(step_dir, 6, "bh_vertical",   normalize_for_view(bh_v))
        save_step(step_dir, 7, "bh_min",        normalize_for_view(bh_min))

        # 5) 이진화 + opening (작은 노이즈 제거)
        _, binary_thr = cv2.threshold(bh_min, BH_THRESHOLD, 255, cv2.THRESH_BINARY)
        save_step(step_dir, 8, "threshold", binary_thr)
        binary = cv2.morphologyEx(binary_thr, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (POST_OPEN_K, POST_OPEN_K)))
        save_step(step_dir, 9, "opened", binary)

        # 6) 보더 제거 (원본 크기 복원)
        binary = binary[BORDER_PAD:BORDER_PAD + H, BORDER_PAD:BORDER_PAD + W]
        save_step(step_dir, 10, "cropped_proc", binary)

        # 7) 영역 특징 모양 필터 (강의자료 6주차)
        min_area, min_wh = adaptive_minimums(W)
        kept = filter_by_shape(binary, min_area, min_wh)

        # 8) segmentation 마스크 및 overlay
        seg_mask = render_bbox_mask(gray.shape, kept)
        save_step(step_dir, 11, "shape_filtered_mask", seg_mask)
        overlay = make_overlay(bgr, kept)
        save_step(step_dir, 12, "overlay", overlay)

        # 9) 제출용 결과를 Result/ 루트에 저장
        cv2.imwrite(os.path.join(result_dir, stem + "_proc.jpg"), binary)
        cv2.imwrite(os.path.join(result_dir, stem + "_seg.jpg"),  overlay)

        # 10) 이물질 정보를 터미널에 출력 (Rasterized 순)
        defects = extract_defect_info(kept, gray, H)
        print_defects_table(fn, defects)


if __name__ == "__main__":
    main()
