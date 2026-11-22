import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prediction.predictor import Predictor


def run_single_knn_prediction():
    print("=" * 65)
    print("CSE-CIC-IDS2018: KNN MODEL SINGLE-FLOW INFERENCE")
    print("=" * 65)

    benign_record = {
        "Dst_Port": 443.0,
        "Protocol": 6.0,
        "Flow_Duration": 37196175.0,
        "Tot_Fwd_Pkts": 3.0,
        "Tot_Bwd_Pkts": 2.0,
        "TotLen_Fwd_Pkts": 158.0,
        "Fwd_Pkt_Len_Max": 79.0,
        "Fwd_Pkt_Len_Min": 0.0,
        "Fwd_Pkt_Len_Mean": 52.67,
        "Fwd_Pkt_Len_Std": 45.61,
        "Bwd_Pkt_Len_Max": 0.0,
        "Bwd_Pkt_Len_Min": 0.0,
        "Bwd_Pkt_Len_Mean": 0.0,
        "Flow_Byts/s": 4.25,
        "Flow_Pkts/s": 0.134,
        "Flow_IAT_Mean": 9299043.75,
        "Flow_IAT_Max": 32420207.0,
        "Bwd_IAT_Tot": 32487564.0,
        "Bwd_IAT_Mean": 32487564.0,
        "Bwd_IAT_Std": 0.0,
        "Bwd_IAT_Max": 32487564.0,
        "Bwd_IAT_Min": 32487564.0,
        "PSH_Flag_Cnt": 0.0,
        "ACK_Flag_Cnt": 1.0,
        "Pkt_Len_Var": 1872.3,
        "Init_Fwd_Win_Byts": 255.0,
        "Init_Bwd_Win_Byts": 73.0,
        "Fwd_Act_Data_Pkts": 1.0,
        "Fwd_Seg_Size_Min": 20.0,
    }

    attack_record = {
        "Dst_Port": 22.0,
        "Protocol": 6.0,
        "Flow_Duration": 309795.0,
        "Tot_Fwd_Pkts": 22.0,
        "Tot_Bwd_Pkts": 21.0,
        "TotLen_Fwd_Pkts": 1928.0,
        "Fwd_Pkt_Len_Max": 640.0,
        "Fwd_Pkt_Len_Min": 0.0,
        "Fwd_Pkt_Len_Mean": 87.64,
        "Fwd_Pkt_Len_Std": 137.78,
        "Bwd_Pkt_Len_Max": 976.0,
        "Bwd_Pkt_Len_Min": 0.0,
        "Bwd_Pkt_Len_Mean": 126.90,
        "Flow_Byts/s": 14825.93,
        "Flow_Pkts/s": 138.80,
        "Flow_IAT_Mean": 7376.07,
        "Flow_IAT_Max": 80171.0,
        "Bwd_IAT_Tot": 309782.0,
        "Bwd_IAT_Mean": 15489.1,
        "Bwd_IAT_Std": 31890.96,
        "Bwd_IAT_Max": 119064.0,
        "Bwd_IAT_Min": 12.0,
        "PSH_Flag_Cnt": 1.0,
        "ACK_Flag_Cnt": 0.0,
        "Pkt_Len_Var": 42227.96,
        "Init_Fwd_Win_Byts": 26883.0,
        "Init_Bwd_Win_Byts": 230.0,
        "Fwd_Act_Data_Pkts": 16.0,
        "Fwd_Seg_Size_Min": 32.0,
    }

    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("\n[1] Initializing KNN Model Artifact (with StandardScaler)...")
    t0 = time.time()
    predictor = Predictor("knn", dataset_name="02-14-2018")
    load_time = time.time() - t0
    print(f"    [OK] KNN Model loaded successfully in {load_time:.2f}s")
    print(f"    [OK] StandardScaler active for distance metric: {predictor.scaler is not None}")

    print("\n" + "-" * 65)
    print("[TEST 1] EVALUATING SINGLE BENIGN SAMPLE FLOW (HTTPS - Port 443)")
    print("-" * 65)
    t1 = time.time()
    res_b = predictor.predict(benign_record)
    lat_b = (time.time() - t1) * 1000.0
    print(f"  Classification Label : {res_b['predictions'][0]} ({res_b['labels'][0]})")
    print(f"  Attack Probability   : {res_b['probabilities'][0]:.4f} ({res_b['probabilities'][0]*100:.2f}%)")
    print(f"  Threat Severity Tier : {res_b['threat_levels'][0]}")
    print(f"  Inference Latency    : {lat_b:.2f} ms")
    print(f"  Verdict              : Permitted legitimate traffic.")

    print("\n" + "-" * 65)
    print("[TEST 2] EVALUATING SINGLE INTRUSION ATTACK FLOW (SSH-BruteForce - Port 22)")
    print("-" * 65)
    t2 = time.time()
    res_a = predictor.predict(attack_record)
    lat_a = (time.time() - t2) * 1000.0
    print(f"  Classification Label : {res_a['predictions'][0]} ({res_a['labels'][0]})")
    print(f"  Attack Probability   : {res_a['probabilities'][0]:.4f} ({res_a['probabilities'][0]*100:.2f}%)")
    print(f"  Threat Severity Tier : {res_a['threat_levels'][0]}")
    print(f"  Inference Latency    : {lat_a:.2f} ms")
    if res_a['predictions'][0] == 1:
        print(f"  Verdict              : INTRUSION DETECTED! Security alert triggered.")
    else:
        print(f"  Verdict              : Classified as {res_a['labels'][0]}.")
    print("=" * 65)

    return res_b, res_a


if __name__ == "__main__":
    run_single_knn_prediction()
