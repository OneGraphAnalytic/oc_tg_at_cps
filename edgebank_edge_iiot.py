from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import TemporalDataLoader
import numpy as np
from tqdm import tqdm
import os.path as osp
from pathlib import Path
import argparse
import torch
import pandas as pd

# internal imports
from modules.edgebank_predictor import EdgeBankPredictor
from tgb.utils.utils import set_random_seed
from sklearn.metrics import confusion_matrix, classification_report

from datasets.edge_iiot import EdgeIIoTset





def processing(args):

##############################################################################################################
# EdgeBank Utils
##############################################################################################################

    def test(test_loader):
        y_preds = []
        labels = []
        for batch in tqdm(test_loader):
            batch.src = batch.src.numpy()
            batch.dst = batch.dst.numpy()
            batch.t = batch.t.numpy()
            y_pred = edgebank.predict_link(batch.src, batch.dst)
            y_preds.append(y_pred)
            labels.append(batch.y)
    
            if EVALUATION_STRATEGY == 'predicted_knowledge':
                #update only edges, which are predicted to be benign
                update_dst, update_src, update_t = batch.dst[y_pred.round() == 1], batch.src[y_pred.round() == 1], batch.t[y_pred.round() == 1]
            elif EVALUATION_STRATEGY == 'attack_knowledge':
                batch.y = batch.y.numpy()
                mask = (batch.y == 0) # equals to benign
                update_dst, update_src, update_t = batch.dst[mask], batch.src[mask], batch.t[mask] # transform to numpy array
            elif EVALUATION_STRATEGY == 'blind':
                update_dst, update_src, update_t = batch.dst, batch.src, batch.t
            elif EVALUATION_STRATEGY == 'no':
                update_dst, update_src, update_t = [], [], []
            else:
                raise Exception('Unknown evaluation strategy')

            # only update edgebank if all three arrays are not empty
            if(len(update_src) > 0 and len(update_dst) > 0 and len(update_t) > 0):
                edgebank.update_memory(update_src, update_dst, update_t)
        y_preds = np.concatenate(y_preds)

        # switch values in y_preds, because here is a 1 = benign and 0 = malicious
        y_preds = 1 - y_preds
        labels = np.concatenate(labels)

        return y_preds, labels
    
##############################################################################################################
# Hyperparameters / Settings
##############################################################################################################

    TRAIN_DATA_SIZE = args.train_data_size # default 100000 up to 150000 is possible (then also malicious data is included)
    BATCH_SIZE = args.batch_size # default 200
    EVALUATION_STRATEGY = args.eval_strategy # 'attack_knowledge' or 'predicted_knowledge' or 'blind'
    SEED = args.seed  # set the random seed for consistency
    set_random_seed(SEED)
    MEMORY_MODE = args.memory_mode # `unlimited` or `fixed_time_window`
    TIME_WINDOW_RATIO = args.time_window_ratio # default 0.15


##############################################################################################################
# Data
##############################################################################################################
    path = osp.join(osp.dirname(osp.realpath(__file__)), './', 'data/edge-iiot_submission', 'nids')
    dataset = EdgeIIoTset(path, name='DNN-EdgeIIoT-dataset')
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    train_data = data[data.t < TRAIN_DATA_SIZE]
    eval_data = data[data.t >= TRAIN_DATA_SIZE]
    
    test_loader = TemporalDataLoader(
        eval_data,
        batch_size=BATCH_SIZE,
        neg_sampling_ratio=0.0,
    )
##############################################################################################################
# Edgebank Processing
##############################################################################################################
#data for memory in edgebank
    hist_src = train_data.src.numpy()
    hist_dst = train_data.dst.numpy()
    hist_ts = train_data.t.numpy()

    # Set EdgeBank with memory updater
    edgebank = EdgeBankPredictor(
            hist_src,
            hist_dst,
            hist_ts,
            memory_mode=MEMORY_MODE,
            time_window_ratio=TIME_WINDOW_RATIO)

    y_preds, y_trues,  = test(test_loader)
    cm = confusion_matrix(y_trues, y_preds.round())
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')
    print(classification_report(y_trues, y_preds.round()))

##############################################################################################################
# Result Logging
##############################################################################################################
    # create result folder
    result_folder = './results'
    Path(result_folder).mkdir(parents=True, exist_ok=True)

    # create result file
    result_file = 'results/edgebank_results.txt'
    f = open(result_file, "a")
    f.write(f'Evaluation Strategy: {EVALUATION_STRATEGY}\n')
    f.write(f'TRAIN_DATA_SIZE: {TRAIN_DATA_SIZE}\n')
    f.write(f'BATCH_SIZE: {BATCH_SIZE}\n')
    f.write(f'MEMORY_MODE: {MEMORY_MODE}\n')
    f.write(f'TIME_WINDOW_RATIO: {TIME_WINDOW_RATIO}\n')
    f.write(f'SEED: {SEED}\n')
    f.write(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n')
    f.write(f'Classification Report:\n')
    f.write(f'{classification_report(y_trues, y_preds.round())}\n')
    f.write(f'Confusion Matrix:\n')
    f.write(f'{cm}\n')
    f.write(f'-----------------------------------\n')
    f.write(f'-----------------------------------\n')
    f.write(f'-----------------------------------\n')
    f.close()
##############################################################################################################
# Evaluation by Attack Type
##############################################################################################################

    df = pd.DataFrame({'y_preds': y_preds.flatten().astype(int)})
    df['y_trues'] = y_trues
    df['tps'] = np.where((df['y_preds'] == 1) & (df['y_trues'] == 1), 1, 0)
    df['fps'] = np.where((df['y_preds'] == 1) & (df['y_trues'] == 0), 1, 0)
    df['tns'] = np.where((df['y_preds'] == 0) & (df['y_trues'] == 0), 1, 0)
    df['fns'] = np.where((df['y_preds'] == 0) & (df['y_trues'] == 1), 1, 0)
    
    df.to_csv(f'results/evaluation{"tmp_edge"}.csv', index=False)
    
    csv_path = f"data/edge-iiot_submission/nids/dnn-edgeiiot-dataset/raw/DNN-EdgeIIoT-dataset.csv"
    df_edge_iiot = pd.read_csv(csv_path)
    df_edge_iiot = df_edge_iiot[df_edge_iiot['Attack_type'] != 'DDoS_UDP']
    df_edge_iiot = df_edge_iiot[df_edge_iiot['Attack_type'] != 'MITM']
    df_edge_iiot.reset_index(drop=True, inplace=True)
    df_edge_iiot = df_edge_iiot[df_edge_iiot.index >= TRAIN_DATA_SIZE]
    df_type = df_edge_iiot[["Attack_type", "Attack_label"]].reset_index(drop=True)
    df_merged = pd.merge(df_type, df, left_index=True, right_index=True, how="inner")
    df_merged.to_csv(f'results/evaluation{"tmp_edge_fps_investigation"}.csv', index=False)

    grouped_stats = df_merged.groupby('Attack_type').agg({
        'tps': 'sum',
        'fps': 'sum',
        'tns': 'sum',
        'fns': 'sum'
        }).reset_index()
    
    # Display the grouped statistics
    print(f'Evaluation Strategy: {EVALUATION_STRATEGY}')

    print(grouped_stats)

##############################################################################################################
# Main
##############################################################################################################

def main(args):
    # Your main logic goes here
    print("Hyperparameters:")
    print(f"- Evaluation Strategy: {args.eval_strategy}")
    print(f"- Train Data Size: {args.train_data_size}")
    print(f"- Batch Size: {args.batch_size}")   
    print(f"- Memory Mode: {args.memory_mode}") 
    print(f"- Time Window Ratio: {args.time_window_ratio}") 
    print(f"- Seed: {args.seed}")   

    processing(args)


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Edgebank for NIDS Example Edge-IIoT Dataset")

    # Add hyperparameter arguments
    parser.add_argument("--eval_strategy", type=str, default='no', help="Evaluation Strategy") # predicted_knowledge or attack_knowledge or blind or no
    parser.add_argument("--train_data_size", type=int, default=100000, help="Train Data Size")
    parser.add_argument("--batch_size", type=int, default=2000, help="Batch Size")
    parser.add_argument("--memory_mode", type=str, default='unlimited', help="Memory Mode") # fixed_time_window or unlimited
    parser.add_argument("--time_window_ratio", type=float, default=0.15, help="Time Window Ratio")
    parser.add_argument("--seed", type=int, default=1, help="Seed")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)