import pandas as pd
import numpy as np
import argparse
from collections import Counter
from sklearn.preprocessing import StandardScaler

KNN = 2
TRAIN_FILE = 'C:/Users/User/Documents/GitHub/autograding-python-knn/data/IRIS.csv'
TEST_FILE = 'C:/Users/User/Documents/GitHub/autograding-python-knn/data/iris_test.csv'




# 計算所有歐式距離
def knn_calc_dists(xTrain, xTest, k):
    """
    計算測試數據與訓練數據之間的歐氏距離。
    Input:
        xTrain = n x d matrix. n=rows and d=features
        xTest = m x d matrix. m=rows and d=features
        k = 需要找到的最近鄰個數
    Output:
        dists = 訓練數據與測試數據之間的距離。大小為 n x m
        indices = k x m 矩陣，包含最近鄰的索引
    """
    distances = np.linalg.norm(xTrain[:, np.newaxis] - xTest, axis=2)  # 計算歐氏距離
    indices = np.argsort(distances, axis=0)[:k, :]  # 排序並選擇最近的 K 個鄰居
    return indices, distances


# 使用 KNN 進行預測
def knn_predict(xTrain, yTrain, xTest, k=3):
    """
    預測測試數據的標籤。
    Input:
        xTrain = n x d matrix. 訓練數據的特徵
        yTrain = n x 1 array. 訓練數據的標籤
        xTest = m x d matrix. 測試數據的特徵
        k = 最近鄰的數量
    Output:
        predictions = 預測的標籤，預測值是最常見的標籤
    """
    indices, distances = knn_calc_dists(xTrain, xTest, k)
    predictions = []

    # 對於每個測試樣本，選擇 K 個最近鄰並進行投票
    for i in rdfsaors = yTrain[indices[:, i]]  # 獲取 K 個鄰居的標籤
        most_common = Counter(neighbors.flatten()).most_common(1)  # 找到最常見的標籤
        predictions.append(most_common[0][0])

    return np.array(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K Nearest Neighbor Classifier")
    parser.add_argument("--train-csv", help="Training data in CSV format. Labels are stored in the last column.",
                        required=True)
    parser.add_argument("--test-csv", help="Test data in CSV format", required=True)
    parser.add_argument("--num_k", "-k", dest="K", help="Number of nearest neighbors", default=3, type=int)
    args = parser.parse_args()

    # Load training CSV file. The labels are stored in the last column
    train_df = pd.read_csv(args.train_csv)
    train_data = train_df.iloc[:, :-1].to_numpy()  # 特徵數據
    train_label = train_df.iloc[:, -1:].to_numpy()  # 標籤數據

    test_df = pd.read_csv(args.test_csv)
    test_data = test_df.iloc[:, :-1].to_numpy()  # 特徵數據
    test_label = test_df.iloc[:, -1:].to_numpy()  # 標籤數據

    # 標準化數據
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)

    # 預測
    predictions = knn_predict(train_data_scaled, train_label, test_data_scaled, args.K)

    # 儲存預測結果
    df = pd.DataFrame(predictions)
    df.to_csv("predictions.csv", header=False, index=False)

    # 計算準確度
    result = predictions == test_label.flatten()
    accuracy = np.mean(result)  # 計算平均準確度
    print(f'Evaluate KNN(K={args.K}) on Iris Flower dataset. Accuracy = {accuracy:.2f}')

    # 檢查準確度是否大於 0.33
    assert accuracy > 0.33

