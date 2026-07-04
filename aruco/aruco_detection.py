"""ArUco 검출 ver2.
친구 아이디어 적용:
  (1) 강한 콘트라스트 부여 후 모니터의 네 꼭짓점 검출
  (2) 모니터 영역을 직사각형으로 perspective 보정
  (3) 글자 마스크 생성 → 텍스트 영역만 inpaint로 제거
  (4) 깨끗해진 모니터 영상에 ArucoDetector 적용
  (5) 검출된 corner를 역변환으로 원본 좌표로 복원
모니터 검출이 실패하면(특히 어두운 콘텐츠 케이스) ver1 파이프라인으로 fallback.
"""
import cv2
import os
import numpy as np


# 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "Result_ver2")
INPUT_FILES = [f"Aruco_test{i}.jpg" for i in range(1, 6)]

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
VALID_IDS = {0}
MIN_MARKER_SIDE_PX = 50

CLAHE_C10_T8 = (10.0, (8, 8))
CLAHE_C6_T16 = (6.0, (16, 16))


# Aruco Detector (ver1과 동일)
def make_detector():
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.errorCorrectionRate = 0.4
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 10
    p.adaptiveThreshConstant = 3
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID), p
    )


# --- 단일 전처리 연산들 (ver1 그대로) ---
def clahe(gray, clip, tile):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=tile).apply(gray)


def bilateral(gray, d=9, sc=50, ss=50):
    return cv2.bilateralFilter(gray, d, sc, ss)


def downscale(gray, factor):
    return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)


def dog_filter(gray, s1=4, s2=20):
    g_f = gray.astype(np.float32)
    G1 = cv2.GaussianBlur(g_f, (0, 0), sigmaX=s1, sigmaY=s1)
    G2 = cv2.GaussianBlur(g_f, (0, 0), sigmaX=s2, sigmaY=s2)
    return cv2.normalize(G1 - G2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def percentile_stretch(gray, lo_pct=10, hi_pct=90):
    lo = float(np.percentile(gray, lo_pct))
    hi = float(np.percentile(gray, hi_pct))
    if hi <= lo:
        return gray.copy()
    stretched = np.clip((gray.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255)
    return stretched.astype(np.uint8)


# --- (NEW) 단계 1: 모니터 네 꼭짓점 검출 ---
def find_monitor_corners(bgr):
    """강한 콘트라스트 + Otsu + 가장 큰 4각형 contour → 모니터 4 corner.
    실패하면 None 반환."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    img_area = H * W

    # 강한 콘트라스트로 모니터/배경 명암차 확장
    boosted = clahe(gray, clip=20.0, tile=(8, 8))
    # 콘텐츠 디테일은 큰 가우시안으로 뭉개기 (sigma를 이미지 크기에 약하게 비례)
    sigma = max(51, int(min(H, W) * 0.025)) | 1
    smooth = cv2.GaussianBlur(boosted, (sigma, sigma), 0)

    # 양 polarity + Triangle (모니터가 밝거나 어두운 경우 모두 시도)
    kc = max(101, int(min(H, W) * 0.04)) | 1
    ko = max(51, int(min(H, W) * 0.02)) | 1
    quads = []
    for flag in [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRIANGLE]:
        if flag == cv2.THRESH_TRIANGLE:
            _, bw = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        else:
            _, bw = cv2.threshold(smooth, 0, 255, flag + cv2.THRESH_OTSU)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((kc, kc), np.uint8))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((ko, ko), np.uint8))
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            # 영상의 10%~95% (전체 이미지 foreground는 제외, 그 외엔 모두 후보)
            if not (0.10 * img_area < area < 0.95 * img_area):
                continue
            # 4-corner 다각형 근사 시도, 실패 시 minAreaRect 사용
            peri = cv2.arcLength(c, True)
            approx = None
            for eps in [0.01, 0.02, 0.03, 0.05, 0.08]:
                a = cv2.approxPolyDP(c, eps * peri, True)
                if len(a) == 4 and cv2.isContourConvex(a):
                    approx = a; break
            if approx is None:
                approx = cv2.boxPoints(cv2.minAreaRect(c)).reshape(4, 1, 2)
            q = approx.reshape(4, 2).astype(np.float32)
            # quad의 모든 코너가 이미지 안에 있어야 신뢰 (음수/초과 좌표 다 reject)
            if (q[:, 0].min() < -5 or q[:, 0].max() > W + 5
                    or q[:, 1].min() < -5 or q[:, 1].max() > H + 5):
                continue
            # quad 크기가 image 거의 전체면 신뢰 X (모니터가 화면 다 차도 99% 미만)
            qw = q[:, 0].max() - q[:, 0].min()
            qh = q[:, 1].max() - q[:, 1].min()
            if qw > 0.99 * W and qh > 0.99 * H:
                continue
            quads.append((area, q))
    if not quads:
        return None
    quads.sort(key=lambda x: -x[0])
    return quads[0][1]


def order_quad(pts):
    """TL, TR, BR, BL 순서로 정렬."""
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    return pts[[np.argmin(s), np.argmax(d), np.argmax(s), np.argmin(d)]]


# --- (NEW) 단계 2: 모니터 영역을 직사각형으로 펴기 ---
def rectify_monitor(bgr, corners):
    """검출된 모니터 4 corner → 직사각형으로 perspective 보정.
    반환: (rectified BGR, H_monitor, (w,h)) — H_monitor은 원본→rectified 변환."""
    src = order_quad(corners.astype(np.float32))
    w = int(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3])))
    h = int(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1])))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(bgr, H, (w, h)), H, (w, h)


# --- (NEW) 단계 3: 텍스트 마스크 생성 ---
def make_text_mask(gray, min_size=4, max_size=25, block_size=25, C=10):
    """글자 정도 크기의 dark 구조만 마스크로 잡음.
    - 3주차 강의자료(55쪽) 권장: 조명 불균일/문서 텍스트 → Adaptive Gaussian
    - Stretch + CLAHE 전처리(57쪽)로 결과 향상
    - 글자 크기 정도의 connected component만 보존, 큰 구조(마커)는 제외"""
    # 전처리: percentile stretch + CLAHE (강의 57쪽)
    pre = percentile_stretch(gray, 1, 99)
    pre = clahe(pre, clip=2.0, tile=(8, 8))
    # 노이즈 줄이기 위해 약한 블러
    pre = cv2.GaussianBlur(pre, (3, 3), 0)
    # Adaptive Gaussian: 글자=dark이므로 BINARY_INV로 글자가 흰색
    if block_size % 2 == 0:
        block_size += 1
    bw = cv2.adaptiveThreshold(pre, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV,
                               block_size, C)

    # connected component 크기 필터로 글자 정도만 남기기
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    mask = np.zeros_like(gray, dtype=np.uint8)
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        if w < min_size or h < min_size:
            continue
        if w > max_size or h > max_size:
            continue
        mask[labels == i] = 255
    # 글자들을 살짝 두툼하게 (글자 가장자리도 함께 inpaint되도록)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    return mask


# --- (NEW) 단계 4: 텍스트 제거 (inpaint) ---
def remove_text(bgr, text_mask):
    return cv2.inpaint(bgr, text_mask, 3, cv2.INPAINT_TELEA)


# --- (ver1) Rejected 후보 회복 ---
_DICT = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)


def _make_ref_bits(mid, side=240):
    ref = cv2.aruco.generateImageMarker(_DICT, mid, side)
    cell = side // 6
    m = int(cell * 0.13)
    bits = np.zeros((4, 4), dtype=int)
    for r in range(4):
        for c in range(4):
            y1, x1 = (r + 1) * cell + m, (c + 1) * cell + m
            y2, x2 = (r + 2) * cell - m, (c + 2) * cell - m
            bits[r, c] = 1 if ref[y1:y2, x1:x2].mean() > 128 else 0
    return bits


_REF_BITS_ID0 = _make_ref_bits(0)


def _extract_bits_strong(warped, margin_ratio=0.20):
    cl = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(4, 4)).apply(warped)
    side = cl.shape[0]
    cell = side // 6
    m = int(cell * margin_ratio)
    means = []
    for r in range(4):
        for c in range(4):
            y1, x1 = (r + 1) * cell + m, (c + 1) * cell + m
            y2, x2 = (r + 2) * cell - m, (c + 2) * cell - m
            means.append(cl[y1:y2, x1:x2].mean())
    threshold = float(np.median(means))
    return np.array([1 if v > threshold else 0 for v in means]).reshape(4, 4)


def recover_rejected(rejected_list, source_gray, ref_bits=_REF_BITS_ID0,
                     ham_max=2, min_side=50, max_aspect=2.0, side=240):
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
                   dtype=np.float32)
    recovered = []
    for r in rejected_list:
        pts = r.reshape(4, 2).astype(np.float32)
        sides = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
        if min(sides) < min_side or max(sides) / min(sides) > max_aspect:
            continue
        H = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(source_gray, H, (side, side))
        bits = _extract_bits_strong(warped)
        best_h = 16
        for k in range(4):
            h = (np.rot90(bits, k) != ref_bits).sum()
            if h < best_h:
                best_h = h
        if best_h <= ham_max:
            recovered.append(r)
    return recovered


# --- 후처리 (ver1) ---
def marker_metrics(c):
    pts = c.reshape(4, 2).astype(np.float32)
    sides = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
    return pts.mean(axis=0), float(np.mean(sides)), float(min(sides))


def filter_size_and_id(corners, ids, min_side):
    if ids is None or len(ids) == 0:
        return [], []
    out_c, out_ids = [], []
    for cc, mid in zip(corners, ids.flatten()):
        if VALID_IDS is not None and int(mid) not in VALID_IDS:
            continue
        _, _, ms = marker_metrics(cc)
        if ms >= min_side:
            out_c.append(cc)
            out_ids.append(int(mid))
    return out_c, out_ids


def dedup(corners_list, ids_list):
    groups = []
    for c, mid in zip(corners_list, ids_list):
        center, avg_side, _ = marker_metrics(c)
        matched = False
        for g in groups:
            g_center, g_side, _ = marker_metrics(g["corners"][0])
            if mid == g["id"] and np.linalg.norm(center - g_center) < max(avg_side, g_side) * 0.5:
                g["corners"].append(c)
                matched = True
                break
        if not matched:
            groups.append({"id": mid, "corners": [c]})
    final_c, final_ids = [], []
    for g in groups:
        # 가장 정사각형에 가까운 corner 선택
        def quality(c):
            pts = c.reshape(4, 2)
            sides = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
            return (max(sides) / max(min(sides), 1e-6), -np.mean(sides))
        g["corners"].sort(key=quality)
        final_c.append(g["corners"][0])
        final_ids.append(g["id"])
    return final_c, final_ids


# --- 시각화 ---
def draw_markers(img, corners, ids, color=(0, 0, 255)):
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    h, w = out.shape[:2]
    thick = max(4, w // 400)
    font_scale = max(1.0, w / 1200.0)
    for c, mid in zip(corners, ids):
        pts = c.reshape(4, 2).astype(np.int32)
        cv2.polylines(out, [pts], True, color, thick)
        center = pts.mean(axis=0).astype(int)
        text = f"ID={int(mid)}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        tx = max(0, min(w - tw, int(center[0]) - tw // 2))
        ty = max(th + 4, min(h - 4, int(center[1]) - thick * 4))
        cv2.rectangle(out, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), (0, 0, 0), -1)
        cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 255, 255), thick)
    return out


# --- ver1 multi-variant fallback (모니터 검출 실패 시) ---
def detect_with_multivariant(detector, bgr):
    """ver1 파이프라인 (5개 변형 union)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bilat = bilateral(gray)
    v1 = clahe(bilat, *CLAHE_C6_T16)
    v2 = clahe(percentile_stretch(bilat, 10, 90), *CLAHE_C10_T8)
    variants = [(v1, 1.0), (v2, 1.0)]
    for f in [0.4, 0.5, 0.6]:
        s = downscale(gray, f); sb = bilateral(s)
        variants.append((clahe(dog_filter(sb, 4, 20), *CLAHE_C10_T8), f))

    union_c, union_ids = [], []
    for img, scale in variants:
        c, ids, rej = detector.detectMarkers(img)
        rec = recover_rejected(rej, img)
        cc = list(c) + list(rec)
        ii = (list(ids.flatten().tolist()) if ids is not None else []) + [0] * len(rec)
        if scale != 1.0:
            cc = [x / scale for x in cc]
        ia = np.array(ii, dtype=np.int32).reshape(-1, 1) if ii else None
        c2, ids2 = filter_size_and_id(cc, ia, MIN_MARKER_SIDE_PX)
        union_c.extend(c2); union_ids.extend(ids2)
    return dedup(union_c, union_ids)


# --- ver2 메인 파이프라인 ---
def run_one_image(detector, fn):
    path = os.path.join(BASE_DIR, fn)
    if not os.path.isfile(path):
        return None
    bgr = cv2.imread(path)
    if bgr is None:
        return None

    stem = os.path.splitext(fn)[0]
    debug_dir = os.path.join(RESULT_DIR, stem)
    os.makedirs(debug_dir, exist_ok=True)
    step = [0]
    def dump(name, img):
        step[0] += 1
        cv2.imwrite(os.path.join(debug_dir, f"({step[0]:02d})_{name}.jpg"), img)

    dump("original", bgr)

    # (1) 모니터 4 꼭짓점 검출
    corners = find_monitor_corners(bgr)
    used_fallback = False
    if corners is None:
        used_fallback = True
    else:
        # 검출된 corner 시각화
        vis = bgr.copy()
        for x, y in corners.astype(int):
            cv2.circle(vis, (int(x), int(y)), 30, (0, 0, 255), -1)
        cv2.drawContours(vis, [corners.astype(np.int32)], 0, (0, 255, 255), 6)
        dump("monitor_corners", vis)

    # (A) ver1 multi-variant 검출 (모니터 검출 성공 여부와 무관하게 항상 실행)
    v1_c, v1_ids = detect_with_multivariant(detector, bgr)

    if used_fallback:
        # 모니터 검출 실패 → ver1 결과만 사용
        dump("FALLBACK_to_ver1_only", bgr)
        final_c, final_ids = v1_c, v1_ids
    else:
        # (2) 모니터 영역을 직사각형으로 펴기
        rectified, H_monitor, (w_r, h_r) = rectify_monitor(bgr, corners)
        dump("rectified_monitor", rectified)

        # (3) rectified gray
        rect_gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
        dump("rectified_gray", rect_gray)

        # (4) 텍스트 마스크 생성
        text_mask = make_text_mask(rect_gray, min_size=4, max_size=25)
        dump("text_mask", text_mask)

        # (5) 텍스트 inpaint (제거). 마스크가 영상의 25% 이상이면 마커까지 지워질 위험이
        #     커서 inpaint를 skip하고 rectified 원본을 그대로 사용.
        mask_ratio = float(text_mask.mean() / 255.0)
        if mask_ratio < 0.25:
            cleaned = remove_text(rectified, text_mask)
            dump("text_removed", cleaned)
        else:
            cleaned = rectified.copy()
            dump(f"text_removed_SKIPPED_mask_{int(mask_ratio*100)}pct", cleaned)
        cleaned_gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)

        # (6) 깨끗한 영상에 ArUco 검출
        #     ver1 처럼 멀티 variant를 적용해 안정성 확보
        union_c_rect, union_ids_rect = [], []
        bilat = bilateral(cleaned_gray)
        variants = [
            (clahe(bilat, *CLAHE_C6_T16), 1.0),
            (clahe(percentile_stretch(bilat, 10, 90), *CLAHE_C10_T8), 1.0),
        ]
        for f in [0.4, 0.5, 0.6]:
            s = downscale(cleaned_gray, f); sb = bilateral(s)
            variants.append((clahe(dog_filter(sb, 4, 20), *CLAHE_C10_T8), f))
        for vi, (vimg, scale) in enumerate(variants):
            dump(f"variant_{vi+1}", vimg)
            c, ids, rej = detector.detectMarkers(vimg)
            rec = recover_rejected(rej, vimg)
            cc = list(c) + list(rec)
            ii = (list(ids.flatten().tolist()) if ids is not None else []) + [0] * len(rec)
            if scale != 1.0:
                cc = [x / scale for x in cc]
            ia = np.array(ii, dtype=np.int32).reshape(-1, 1) if ii else None
            c2, ids2 = filter_size_and_id(cc, ia, MIN_MARKER_SIDE_PX)
            union_c_rect.extend(c2); union_ids_rect.extend(ids2)
        # rectified 좌표에서 dedup
        rect_c, rect_ids = dedup(union_c_rect, union_ids_rect)
        dump("detected_in_rectified",
             draw_markers(rectified, rect_c, rect_ids, color=(0, 0, 255)))

        # (7) corner를 원본 좌표로 역변환 (H_monitor의 역행렬 사용)
        if rect_c:
            H_inv = np.linalg.inv(H_monitor)
            ver2_c = [
                cv2.perspectiveTransform(c.reshape(-1, 1, 2).astype(np.float32), H_inv)
                for c in rect_c
            ]
            ver2_ids = rect_ids
        else:
            ver2_c, ver2_ids = [], []

        # (8) ver1 결과 + ver2 결과 union + dedup
        union_c = list(v1_c) + list(ver2_c)
        union_ids = list(v1_ids) + list(ver2_ids)
        final_c, final_ids = dedup(union_c, union_ids)

    # 결과 저장
    overlay = draw_markers(bgr, final_c, final_ids, color=(0, 0, 255))
    cv2.imwrite(os.path.join(RESULT_DIR, f"{stem}_overlay.jpg"), overlay)
    dump("final_overlay", overlay)

    # Perspective 보정 결과 (ver1 RANSAC 방식과 동일)
    if final_c:
        # ver2의 H_monitor가 있으면 그걸 사용, 없으면 ver1 RANSAC
        if not used_fallback:
            H_global = H_monitor
            rectified_out = rectify_monitor(bgr, corners)[0]
            # rectified_out에 마커 표시
            rect_corners_disp = [
                cv2.perspectiveTransform(c.reshape(-1, 1, 2).astype(np.float32), H_global)
                for c in final_c
            ]
            rectified_out = draw_markers(rectified_out, rect_corners_disp, final_ids,
                                          color=(0, 0, 255))
        else:
            rectified_out, H_global = bgr.copy(), None
    else:
        rectified_out, H_global = bgr.copy(), None
    cv2.imwrite(os.path.join(RESULT_DIR, f"{stem}_result.jpg"), rectified_out)
    dump("final_result", rectified_out)

    info = []
    for idx, (c, mid) in enumerate(zip(final_c, final_ids), start=1):
        center, side, _ = marker_metrics(c)
        info.append({
            "index": idx,
            "center": (float(center[0]), float(center[1])),
            "id": int(mid),
            "side_px": float(side),
        })

    tag = " [FALLBACK ver1]" if used_fallback else " [monitor-rectify]"
    print(f"{fn}: {len(final_c)} markers detected{tag}")
    return {"file": fn, "n": len(final_c), "info": info,
            "H_global": H_global.tolist() if H_global is not None else None,
            "fallback": used_fallback}


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    detector = make_detector()
    all_results = []
    for fn in INPUT_FILES:
        r = run_one_image(detector, fn)
        if r:
            all_results.append(r)

    # 요약
    lines = ["=" * 70, "ARUCO DETECTION SUMMARY (ver2)", "=" * 70]
    for r in all_results:
        lines.append("")
        tag = "[FALLBACK ver1]" if r["fallback"] else "[monitor-rectify]"
        lines.append(f"[{r['file']}]  detected = {r['n']}  {tag}")
        lines.append(f"  {'No':>3s} {'center(x,y)':>18s} {'ID':>4s} {'side(px)':>9s}")
        for m in r["info"]:
            cx, cy = m["center"]
            lines.append(f"  {m['index']:>3d} ({cx:>7.1f}, {cy:>7.1f}) {m['id']:>4d} {m['side_px']:>9.1f}")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(RESULT_DIR, "detection_summary.txt"), "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
