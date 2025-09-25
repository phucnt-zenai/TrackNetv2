import os
import cv2
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import time

from utils import get_model, get_object_center, get_metric
from utils import frame_first_Gray  # Nếu dùng input_type = '3d'
import math

HEIGHT = 288
WIDTH = 512

def read_ground_truth(csv_file):
    gt_data = pd.read_csv(csv_file)
    ground_truth = {}
    for _, row in gt_data.iterrows():
        frame_num = int(row['file name'])
        visibility = int(row['visibility'])
        x, y = int(row['x-coordinate']), int(row['y-coordinate'])
        ground_truth[frame_num] = (visibility, x, y)
    return ground_truth

def get_frame_unit(frame_queue, num_frame):
    unit = []
    for i in range(0, len(frame_queue), num_frame):
        frames = frame_queue[i:i+num_frame]
        frames_resized = [cv2.resize(f, (WIDTH, HEIGHT)) for f in frames]
        frames_np = np.stack(frames_resized, axis=0)
        frames_np = frames_np.transpose(0, 3, 1, 2)  # (F, C, H, W)
        frames_np = torch.tensor(frames_np) / 255.0
        unit.append(frames_np)
    return torch.stack(unit)  # (B, F, C, H, W)

def get_confusion_matrix(y_pred_map, y_true_map, y_coor_map, tolerance = 4):
    TP, TN, FP1, FP2, FN = 0, 0, 0, 0, 0

    if np.max(y_pred_map) == 0 and np.max(y_true_map) == 0:
        TN += 1
    elif np.max(y_pred_map) > 0 and np.max(y_true_map) == 0:
        FP2 += 1
    elif np.max(y_pred_map) == 0 and np.max(y_true_map) > 0:
        FN += 1
    else:
        cx_pred, cy_pred = get_object_center(y_pred_map)
        cx_true, cy_true = int(y_coor_map[0]), int(y_coor_map[1])
        dist = math.sqrt((cx_pred - cx_true)**2 + (cy_pred - cy_true)**2)
        if dist > tolerance:
            FP1 += 1
        else:
            TP += 1
    return TP, TN, FP1, FP2, FN

def process_video(video_file, gt_csv, model, num_frame, batch_size, save_dir, input_type='3d', tolerance=4):
    video_name = os.path.basename(video_file)[:-4]
    video_format = video_file[-3:]
    out_video_file = f'{save_dir}/{video_name}_pred.{video_format}'
    out_csv_file = f'{save_dir}/{video_name}_ball.csv'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Ground truth
    gt_data = read_ground_truth(gt_csv)

    # Video info
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ratio = h / HEIGHT
    out = cv2.VideoWriter(out_video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    f = open(out_csv_file, 'w')
    f.write('Frame,Visibility,X,Y\n')

    success = True
    frame_count = 0

    total_TP, total_TN, total_FP1, total_FP2, total_FN = 0, 0, 0, 0, 0
    total_frame_time = 0  # Tổng thời gian cho mỗi frame

    while success:
        frame_queue = []
        for _ in range(num_frame * batch_size):
            success, frame = cap.read()
            if not success:
                break
            frame_queue.append(frame)
            frame_count += 1

        if not frame_queue:
            break

        # Đo thời gian bắt đầu xử lý cho frame
        frame_start_time = time.time()

        # Preprocess frames
        x = get_frame_unit(frame_queue, num_frame)

        with torch.no_grad():
            y_pred = model(x.cuda()).cpu().numpy()

        y_pred_bin = (y_pred > 0.5).astype('uint8') * 255
        y_pred_bin = y_pred_bin.reshape(-1, HEIGHT, WIDTH)

        for i in range(len(frame_queue)):
            frame_idx = frame_count - len(frame_queue) + i
            pred_mask = y_pred_bin[i]
            img = frame_queue[i]
            cx, cy = get_object_center(pred_mask)
            cx_out, cy_out = int(cx * ratio), int(cy * ratio)
            vis = 1 if cx_out > 0 and cy_out > 0 else 0

            # Đo thời gian kết thúc sau khi tính toán tọa độ
            frame_end_time = time.time()
            frame_duration = frame_end_time - frame_start_time
            total_frame_time += frame_duration

            f.write(f"{frame_idx},{vis},{cx_out},{cy_out}\n")

            # Evaluate with ground truth
            if frame_idx in gt_data:
                vis_gt, x_gt, y_gt = gt_data[frame_idx]
                if vis_gt == 1:
                    gt_map = np.zeros((HEIGHT, WIDTH), dtype='uint8')
                    gx, gy = int(x_gt / ratio), int(y_gt / ratio)
                    gt_map[gy, gx] = 255
                    tp, tn, fp1, fp2, fn = get_confusion_matrix(pred_mask, gt_map, [gx, gy], tolerance)
                else:
                    gt_map = np.zeros((HEIGHT, WIDTH), dtype='uint8')
                    tp, tn, fp1, fp2, fn = get_confusion_matrix(pred_mask, gt_map, [0, 0], tolerance)

                total_TP += tp
                total_TN += tn
                total_FP1 += fp1
                total_FP2 += fp2
                total_FN += fn

        # Vẽ vòng tròn và ghi kết quả vào file sẽ được làm sau khi đo thời gian

    cap.release()
    out.release()
    f.close()

    # Tính toán thời gian trung bình của mỗi frame
    avg_frame_time = total_frame_time / frame_count if frame_count > 0 else 0
    print(f"[✓] Xử lý video {video_file} xong, thời gian trung bình cho mỗi frame: {avg_frame_time:.4f} giây")

    return total_TP, total_TN, total_FP1, total_FP2, total_FN, avg_frame_time

def process_all_videos(data_dir, model, num_frame, batch_size, save_dir):
    total_TP = total_TN = total_FP1 = total_FP2 = total_FN = 0
    total_frame_time_all = 0  # Tổng thời gian cho tất cả các frame
    total_frame_count = 0  # Tổng số lượng frame đã xử lý

    for root, dirs, files in os.walk(data_dir):
        video_files = [f for f in files if f.endswith('.mp4')]
        for video_file in video_files:
            video_path = os.path.join(root, video_file)
            csv_file = video_path.replace('.mp4', '.csv')
            if not os.path.exists(csv_file):
                print(f"[!] Thiếu ground truth CSV cho {video_path}")
                continue
            print(f"[✓] Đang xử lý: {video_path}")
            TP, TN, FP1, FP2, FN, avg_frame_time = process_video(video_path, csv_file, model, num_frame, batch_size, save_dir)
            total_TP += TP
            total_TN += TN
            total_FP1 += FP1
            total_FP2 += FP2
            total_FN += FN
            total_frame_time_all += avg_frame_time * (video_file.count('.mp4'))  # Lấy số lượng frame của video
            total_frame_count += video_file.count('.mp4')

    acc, prec, rec = get_metric(total_TP, total_TN, total_FP1, total_FP2, total_FN)
    avg_frame_time_all = total_frame_time_all / total_frame_count if total_frame_count > 0 else 0

    print("\n📊 Kết quả tổng hợp:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"⏱️ Thời gian trung bình mỗi frame: {avg_frame_time_all:.4f} giây")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/badminton-testset/test_dataset', help='Thư mục chứa match1/, match2/, ...')
    parser.add_argument('--model_file', type=str, default='models/model_best_v2.pt')
    parser.add_argument('--num_frame', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='pred_result')
    args = parser.parse_args()

    # Load model
    checkpoint = torch.load(args.model_file)
    param_dict = checkpoint['param_dict']
    model_name = param_dict['model_name']
    input_type = param_dict['input_type']
    model = get_model(model_name, args.num_frame, input_type).cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    process_all_videos(args.data_dir, model, args.num_frame, args.batch_size, args.save_dir)
