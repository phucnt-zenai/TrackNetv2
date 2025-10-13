# evaluate_full.py
import os
import cv2
import csv
import json
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import groupby
from scipy.spatial import distance
from collections import OrderedDict
from typing import Tuple, List
import time

from dataset import Badminton_Dataset
from utils import *

HEIGHT = 288
WIDTH = 512

# ------------------------
# Video & Inference utils
# ------------------------
def read_video(path_video):
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps


def remove_outliers(ball_track, dists, max_dist=100):
    outliers = list(np.where(np.array(dists) > max_dist)[0])
    for i in outliers:
        if i + 1 >= len(dists):
            continue
        if (dists[i+1] > max_dist) | (dists[i+1] == -1):
            ball_track[i] = (None, None)
            outliers.remove(i)
        elif dists[i-1] == -1:
            ball_track[i-1] = (None, None)
    return ball_track


def split_track(ball_track, max_gap=4, max_dist_gap=80, min_track=5):
    list_det = [0 if x[0] else 1 for x in ball_track]
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

    cursor = 0
    min_value = 0
    result = []
    for i, (k, l) in enumerate(groups):
        if (k == 1) & (i > 0) & (i < len(groups) - 1):
            dist = distance.euclidean(ball_track[cursor-1], ball_track[cursor+l])
            if (l >= max_gap) | (dist/l > max_dist_gap):
                if cursor - min_value > min_track:
                    result.append([min_value, cursor])
                    min_value = cursor + l - 1
        cursor += l
    if len(list_det) - min_value > min_track:
        result.append([min_value, len(list_det)])
    return result


def interpolation(coords):
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    x = np.array([x[0] if x[0] is not None else np.nan for x in coords])
    y = np.array([x[1] if x[1] is not None else np.nan for x in coords])

    nons, yy = nan_helper(x)
    x[nons] = np.interp(yy(nons), yy(~nons), x[~nons])
    nans, xx = nan_helper(y)
    y[nans] = np.interp(xx(nans), xx(~nans), y[~nans])

    return [*zip(x, y)]


def write_track(frames, ball_track, path_output_video, fps, trace=7):
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for num in range(len(frames)):
        frame = frames[num].copy()
        for i in range(trace):
            if (num-i > 0):
                if ball_track[num-i][0]:
                    x = int(ball_track[num-i][0])
                    y = int(ball_track[num-i][1])
                    frame = cv2.circle(frame, (x, y), radius=0, color=(0, 0, 255), thickness=10-i)
                else:
                    break
        out.write(frame)
    out.release()


# ------------------------
# Evaluation utils
# ------------------------
def _parse_frame_index_from_name(fname: str) -> int:
    """Cố gắng lấy số frame từ tên file kiểu '0023.jpg' -> 23.
    Nếu không parse được trả ValueError."""
    base = os.path.splitext(os.path.basename(fname))[0]
    # loại bỏ mọi ký tự không phải số ở đầu/cuối (nếu cần)
    # thường base là '0023' nên int(base) OK.
    return int(base)

def _safe_to_int_tuple(x, y):
    """Chuyển x,y có thể float/np.nan/None sang tuple int (rounded) nếu hợp lệ, ngược lại trả None."""
    if x is None or y is None:
        return None
    try:
        if np.isnan(x) or np.isnan(y):
            return None
    except:
        pass
    try:
        return (int(round(float(x))), int(round(float(y))))
    except Exception:
        return None

def validate_from_csv_visual(grouth_truth_csv_path: str,
                            predict_csv_path: str,
                            video_path: str,
                            output_folder: str,
                            min_dist: float = 5.0) -> Tuple[float, float, float, float, List[int], List[int], List[int], List[int], List[int]]:
    """
    Đọc gt và pred csv, vẽ chấm và label lên từng frame lấy từ video_path,
    lưu kết quả vào output_folder và tính precision/recall/f1/accuracy.

    Trả về: precision, recall, f1, accuracy, tp, tn, fp1, fp2, fn
    (tp,tn,fp1,fp2,fn là list theo visibility index như trong code gốc).
    """

    os.makedirs(output_folder, exist_ok=True)

    gt_df = pd.read_csv(grouth_truth_csv_path)
    pred_df = pd.read_csv(predict_csv_path)

    # bỏ 2 frame đầu như trước
    gt_df = gt_df.iloc[2:].reset_index(drop=True)

    merged = pd.merge(gt_df, pred_df, on="file name", how="left")

    tp, fp1, fp2, tn, fn = [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]

    # mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    total_rows = len(merged)
    for idx, row in merged.iterrows():
        fname = row["file name"]
        # parse frame index
        try:
            frame_idx = _parse_frame_index_from_name(fname)
        except ValueError:
            # nếu không parse được, chỉ bỏ qua vẽ nhưng vẫn tính metrics
            frame_idx = None

        x_pred, y_pred = row.get("x"), row.get("y")
        x_gt, y_gt = row.get("x-coordinate"), row.get("y-coordinate")
        vis = int(row.get("visibility", 0)) if not pd.isna(row.get("visibility", 0)) else 0

        has_pred = not (pd.isna(x_pred) or pd.isna(y_pred))
        has_gt   = not (pd.isna(x_gt) or pd.isna(y_gt))

        # classification for this row (one of TP,FP1,FP2,TN,FN)
        classification = None
        if has_pred:
            if vis != 0:
                # both pred and gt exist in sense of visibility !=0
                if has_gt:
                    dst = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                    if dst < min_dist:
                        tp[vis] += 1
                        classification = "TP"
                    else:
                        fp1[vis] += 1
                        classification = "FP1"
                else:
                    # visibility !=0 but no gt coords? treat as FN/FP ambiguous - keep logic: if vis !=0 and has_pred but no valid gt coords
                    # follow original: if has_pred and vis !=0 => compute distance; if no gt coords, treat as FP1
                    fp1[vis] += 1
                    classification = "FP1"
            else:
                # vis == 0 (not visible) but predicted -> FP2
                fp2[vis] += 1
                classification = "FP2"
        else:
            if vis != 0:
                # visible but no prediction -> FN
                fn[vis] += 1
                classification = "FN"
            else:
                # not visible and no pred -> TN
                tn[vis] += 1
                classification = "TN"

        # --- vẽ lên frame nếu có thể ---
        if frame_idx is not None:
            # trong CSV tên kiểu '0023.jpg' mình giả sử frame index bắt đầu từ 0 hoặc 1?
            # Ở đây ta dùng frame_idx as integer trực tiếp (nếu file ghi '0' cho frame 0).
            # Nếu bạn muốn offset (ví dụ file name '0001.jpg' tương ứng frame index 1 trong video),
            # bạn có thể trừ 1 hoặc cộng 0 tùy dataset. Hiện mặc định dùng value trực tiếp.
            # Set vị trí frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                # thử giảm 1 (nhiều dataset đánh số 1-based)
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 1))
                ok2, frame2 = cap.read()
                if ok2 and frame2 is not None:
                    frame = frame2
                else:
                    # không đọc được frame -> bỏ qua lưu ảnh nhưng vẫn tiếp tục
                    print(f"[WARN] Không đọc được frame {frame_idx} cho file {fname}")
                    continue

            # chuyển to integer points
            pt_gt = _safe_to_int_tuple(x_gt, y_gt)
            pt_pred = _safe_to_int_tuple(x_pred, y_pred)

            # vẽ ground-truth: xanh (BGR = (255,0,0) nếu muốn xanh? OpenCV uses BGR => xanh là (255,0,0))
            if pt_gt is not None:
                cv2.circle(frame, pt_gt, radius=5, color=(255,0,0), thickness=-1)  # xanh

            # vẽ prediction: đỏ (BGR (0,0,255))
            if pt_pred is not None:
                cv2.circle(frame, pt_pred, radius=5, color=(0,0,255), thickness=-1)  # đỏ

            # viết classification (TP/FP1/FP2/TN/FN) ở góc trên trái nhỏ
            label = classification if classification is not None else ""
            # chọn màu chữ theo label: TP xanh lá, FP đỏ, FN vàng, TN xám
            color_map = {
                "TP": (0,200,0),
                "FP1": (0,0,255),
                "FP2": (0,0,255),
                "FN": (0,200,200),
                "TN": (150,150,150)
            }
            txt_color = color_map.get(label, (255,255,255))
            cv2.putText(frame, f"{label}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, txt_color, 2, cv2.LINE_AA)

            # optionally cũng in thông tin vis, dst nếu có
            if pt_gt is not None and pt_pred is not None and (not pd.isna(x_gt)) and (not pd.isna(x_pred)):
                try:
                    dst = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                    cv2.putText(frame, f"dist:{dst:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                except:
                    pass

            out_path = os.path.join(output_folder, fname)
            cv2.imwrite(out_path, frame)

    cap.release()

    # metrics
    eps = 1e-15
    fp = sum(fp1) + sum(fp2)
    precision = sum(tp) / (sum(tp) + fp + eps)
    recall = sum(tp) / (sum(tp) + sum(fn) + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    accuracy = (sum(tp) + sum(tn)) / (sum(tp) + sum(tn) + fp + sum(fn) + eps)

    return precision, recall, f1, accuracy, tp, tn, fp1, fp2, fn


def validate_from_csv(grouth_truth_csv_path, predict_csv_path, min_dist=5):
    gt_df = pd.read_csv(grouth_truth_csv_path)
    pred_df = pd.read_csv(predict_csv_path)

    # bỏ 2 frame đầu
    gt_df = gt_df.iloc[2:].reset_index(drop=True)

    merged = pd.merge(gt_df, pred_df, on="file name", how="left")

    tp, fp1, fp2, tn, fn = [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]

    for _, row in merged.iterrows():
        x_pred, y_pred = row["x"], row["y"]
        x_gt, y_gt = row["x-coordinate"], row["y-coordinate"]
        vis = row["visibility"]

        has_pred = not (pd.isna(x_pred) or pd.isna(y_pred))

        if has_pred:
            if vis != 0:
                dst = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                if dst < min_dist:
                    tp[vis] += 1
                else:
                    fp1[vis] += 1
            else:
                fp2[vis] += 1
        else:
            if vis != 0:
                fn[vis] += 1
            else:
                tn[vis] += 1

    eps = 1e-15
    fp = sum(fp1) + sum(fp2)
    precision = sum(tp) / (sum(tp) + fp + eps)
    recall = sum(tp) / (sum(tp) + sum(fn) + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    accuracy = (sum(tp) + sum(tn)) / (sum(tp) + sum(tn) + fp + sum(fn) + eps)

    return precision, recall, f1, accuracy, tp, tn, fp1, fp2, fn

