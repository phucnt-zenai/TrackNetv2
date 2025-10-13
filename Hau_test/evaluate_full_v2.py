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
import time

from dataset import Badminton_Dataset
from utils import *
from utils_evaluate import *

HEIGHT = 288
WIDTH = 512

def inference_model(video_path, csv_path, model, num_frame, batch_size, output_video_path):
    # Video output configuration
    video_format = video_path[-3:]
    if video_format == 'avi':
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    elif video_format == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        raise ValueError('Invalid video format.')

    # Write csv file head
    f = open(csv_path, 'w')
    f.write('file name,x,y\n')

    # Cap configuration
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    success = True
    ratio = h / HEIGHT
    # out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Bộ đếm toàn cục để sinh tên frame liên tục
    global_frame_idx = 0

    while success:
        # Sample frames để tạo input sequence
        frame_queue = []
        for _ in range(num_frame * batch_size):
            success, frame = cap.read()
            if not success:
                break
            frame_queue.append(frame)

        if not frame_queue:
            break

        # Nếu batch chưa đầy đủ thì dừng luôn (không reset lại frame_count)
        if len(frame_queue) % num_frame != 0:
            print("Incomplete mini-batch at the end. Stopping.")
            break

        x = get_frame_unit(frame_queue, num_frame)

        # Inference
        with torch.no_grad():
            y_pred = model(x.cuda())
        y_pred = y_pred.detach().cpu().numpy()
        h_pred = (y_pred > 0.5).astype('uint8') * 255
        h_pred = h_pred.reshape(-1, HEIGHT, WIDTH)

        for i in range(h_pred.shape[0]):
            img = frame_queue[i].copy()
            cx_pred, cy_pred = get_object_center(h_pred[i])
            cx_pred, cy_pred = int(ratio * cx_pred), int(ratio * cy_pred)

            # Ghi ra CSV
            frame_name = f"{global_frame_idx:04d}.jpg"
            if cx_pred == 0 and cy_pred == 0:
                f.write(f"{frame_name},,\n")
            else:
                f.write(f"{frame_name},{cx_pred},{cy_pred}\n")

            # # Vẽ bóng lên video nếu có
            # if cx_pred != 0 or cy_pred != 0:
            #     cv2.circle(img, (cx_pred, cy_pred), 5, (0, 0, 255), -1)

            # out.write(img)
            global_frame_idx += 1  # tăng bộ đếm toàn cục

    # out.release()
    f.close()
    print(f'---> Done {video_path}')

# ------------------------
# Main
# ------------------------
def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    checkpoint = torch.load(args.model_path)
    param_dict = checkpoint['param_dict']
    model_name = param_dict['model_name']
    num_frame = param_dict['num_frame']
    batch_size = args.batch_size
    input_type = param_dict['input_type']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load model
    model = get_model(model_name, num_frame, input_type).cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_tp, all_fp, all_tn, all_fn = 0, 0, 0, 0
    all_precisions, all_recalls, all_f1s, all_accs = [], [], [], []

    root_test = args.test_dir
    games = sorted(os.listdir(root_test))
    total_time = 0
    total_clip = 0

    for game in games:
        game_path = os.path.join(root_test, game)
        clips = sorted(os.listdir(game_path))

        for clip in clips:
            clip_path = os.path.join(game_path, clip)
            video_path = os.path.join(clip_path, f"{clip}.mp4")
            label_path = os.path.join(clip_path, "Label.csv")

            if not (os.path.exists(video_path) and os.path.exists(label_path)):
                continue

            print(f"\nProcessing {video_path}")

            # save outputs
            out_dir = os.path.join(args.output_dir, game, clip)
            os.makedirs(out_dir, exist_ok=True)
            predicted_path = os.path.join(out_dir, "ball_tracks.csv")
            output_video_path = os.path.join(out_dir, "video_tracking.mp4")

            # inference 
            frames, fps = read_video(video_path)
            start_time = time.time()
            inference_model(video_path, predicted_path, model, num_frame, batch_size, output_video_path)
            end_time = time.time()
            number_frames = len(frames)
            if total_clip !=0:
                total_time += (end_time - start_time) / number_frames
            total_clip += 1

            # evaluate
            precision, recall, f1, acc, tp, tn, fp1, fp2, fn = validate_from_csv(label_path, predicted_path, min_dist=args.min_dist)
            print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Acc={acc:.4f}, Time={(end_time - start_time) / number_frames}")

            all_tp += sum(tp)
            all_fp += sum(fp1)
            all_fp += sum(fp2)
            all_tn += sum(tn)
            all_fn += sum(fn)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)
            all_accs.append(acc)

    # final report
    eps = 1e-15
    final_precision = sum(all_precisions)/len(all_precisions)
    final_recall = sum(all_recalls)/len(all_recalls)
    final_f1 = sum(all_f1s)/len(all_f1s)
    final_acc = sum(all_accs)/len(all_accs)

    print("\n=== Final Evaluation on Test Set ===")
    print(f"TP={all_tp}, TN={all_tn}, FP={all_fp}, FN={all_fn}")
    print(f"Precision={final_precision:.6f}, Recall={final_recall:.6f}, F1={final_f1:.6f}, Accuracy={final_acc:.6f}")
    print(f"Inference time for each frame: {total_time / (total_clip - 1)}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, default= r"E:\VNPT\Tracknetv2\Ahau\TrackNetV2-main\exp\model_best.pt", help="Path to model .pt")
    # parser.add_argument("--test_dir", type=str, default=r"E:\VNPT\Tracknetv2\Ahau\TrackNetV2-main\datasets\data_test_checked", help="Path to test dataset root")
    # parser.add_argument("--output_dir", type=str, default=r"E:\VNPT\Tracknetv2\Ahau\TrackNetV2-main\pred_result", help="Directory to save outputs")
    # parser.add_argument("--min_dist", type=float, default=8.0, help="Max distance to ground_trouth for True Positive")
    # parser.add_argument('--num_frame', type=int, default=3)
    # parser.add_argument('--batch_size', type=int, default=8)
    # parser.add_argument("--extrapolation", action="store_true", help="Whether to use interpolation for missing tracks")
    # args = parser.parse_args()
    # main(args)
    gt_path = r"E:\VNPT\Tracknetv2\Post_processing for tracknet\check\dOne_cHeCkerViet\game17\Clip1\Label.csv"
    predicted_path = r"E:\VNPT\Tracknetv2\Post_processing for tracknet\pred_result\game17\Clip1\ball_tracks.csv"
    video_path = r"E:\VNPT\Tracknetv2\Post_processing for tracknet\pred_result\game17\Clip1\video_tracking.mp4"
    output_folder = r"E:\VNPT\Tracknetv2\Post_processing for tracknet\pred_result\game17\Clip1\visual_result"
    precision, recall, f1, acc, tp, tn, fp1, fp2, fn = validate_from_csv_visual(gt_path, predicted_path, video_path, output_folder, min_dist=8)
    print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Acc={acc:.4f}")
    print(f"TP: {tp}")
    print(f"TN: {tn}")
    print(f"FP1: {fp1}")
    print(f"FP2: {fp2}")
    print(f"FN: {fn}")