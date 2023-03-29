import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import sched
import time
import threading
import subprocess
import os
import curses

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pickle
import argparse

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"
cv = lambda x: np.std(x, ddof=1 if len(x) > 1 else 0) / np.mean(x) * 100
normDiff = lambda y: (np.max(y) - np.min(y)) / np.max(y) if np.max(y) != 0 else 0
maxRTT = lambda z: np.max(z)
minRTT = lambda w: np.min(w)
scheduler = sched.scheduler(time.time, time.sleep)
flow_data = {}
stop_event = threading.Event()

def fetch_data():
    while not stop_event.is_set():
        proc = subprocess.Popen(['bash', 'run_tcp.sh'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.wait() # Wait for the script to complete
        with open('data.csv') as file:
            process_data(file)
        time.sleep(3)

def process_data(file):
    global flow_data
    file_lst = file.read().strip(" ").split("\n")
    #remove repeated lines
    header = file_lst[0]
    default_flow = file_lst[1]
    file_lst = list(filter(lambda a: a != header, file_lst))
    file_lst = list(filter(lambda a: a != default_flow, file_lst))
    file_lst.insert(0, header)
    uuids = set()

    #iterate through and parse out tcp fields then add to the dictionary
    for line in file_lst[1:]:
        flow = line.split(",")
        if len(flow) < 105:
            continue
        src = flow[9]
        sPort = flow[7]
        dst = flow[10]
        dPort = flow[8]
        CAState = flow[26]
        RTT = int(flow[48])
        try:
            loss = int(flow[74]) / (int(flow[73]) - int(flow[74]))
        except ZeroDivisionError:
            loss = 0  # set loss to zero if denominator is zero
        uuid = (src + ":" + sPort + "->" + dst + ":" +dPort)
        uuids.add(uuid)

        if uuid in flow_data: #If this flow has already been seen just append the data to its existing pair
            flow_data[uuid]["RTT"].append(RTT)
            flow_data[uuid]["Loss"].append(loss)
            flow_data[uuid]["CAState"].append(CAState)
        else: #if this is a new flow create a pair
            flow_data[uuid] = {"RTT":[RTT], "Loss": [loss], "CAState": [CAState]}
    flow_data_uuids = list(flow_data.keys())
    for uuid in flow_data_uuids: #remove a flow that died from summary
        if uuid not in uuids: #if there is a uuid not present in new data remove it
            del flow_data[uuid]


        

def refine_data(flow_data):
    #print(flow_data)
    formatted = []
    for ip in flow_data:
        formatted.append({"IP": ip, "LossRate": np.mean(np.array(flow_data[ip]["Loss"])), "CoV": cv(np.array(flow_data[ip]["RTT"])), "NormDiff": normDiff(np.array(flow_data[ip]["RTT"])), "MaxRTT": maxRTT(np.array(flow_data[ip]["RTT"])), "MinRTT": minRTT(np.array(flow_data[ip]["RTT"])), "CAState": flow_data[ip]["CAState"]})
    return formatted



def main():
    print("|*****************************|")
    print("|******* Starting Up *********|")
    print("|*****************************|")
    subprocess.Popen(['rm', '-rf', 'data.csv'])
    subprocess.Popen(['rm', '-rf', 'predictions.csv'])
    model = pickle.load(open("congestion_classifier2.sav", 'rb'))
    thread = threading.Thread(target=fetch_data, daemon=True)
    thread.start()
    run_loop(model)
    stop_event.set()



def run_loop(model):
    while not stop_event.is_set():
        global flow_data 
        if len(flow_data) > 0:
            tcp_info = refine_data(flow_data)
            X_test = df = pd.DataFrame(tcp_info)
            ip_addrs = X_test["IP"]
            CAState = X_test["CAState"]
            X_test = X_test.drop(["IP", "CAState"], axis=1)
            predictions = model.predict(X_test)
            ts = time.time()
            CACount = 0
            for state in CAState:
                state = np.array(state).astype(int)
                if np.sum(state) == 0:
                    #this is flow has no congestion
                    predictions[CACount] = 2
                CACount +=1
            summary = "Active TCP flows:\n\n"
            count = 0
            with open('predictions.csv', 'a') as file:
                for ip in ip_addrs:
                    prediction = ""
                    csvprediction = ""
                    if predictions[count] == 0:
                        prediction = RED + "external" + RESET
                        csvprediction = "external"
                    elif predictions[count] == 1: 
                        prediction = YELLOW + "internal" + RESET
                        csvprediction = "internal"
                    else:
                        prediction = GREEN + "neither" + RESET
                        csvprediction = "neither" 
                    summary += ("Flow tuple: " + ip + "\nCongestion State: " + prediction + "\n" + "\n")
                    file.write(ip + "," + csvprediction + "," + str(ts))
                    file.write('\n')
                    count +=1
            file.close()
            import os
            os.system('clear')
            print(summary, end= "\r")
            time.sleep(1)

                

if __name__ == "__main__":
    main()
