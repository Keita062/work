import json
import pandas as pd

# ファイルのパス
file_path = '/content/eSenseData.txt'
data = []

# ファイルを開いて行ごとに読み込む
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 行をカンマで分割
        parts = line.strip().split(',', 2)
        if len(parts) < 3:
            continue

        index = parts[0].strip()  # インデックス
        timestamp = parts[1].strip()  # タイムスタンプ
        json_data = parts[2].strip()  # JSONデータ部分

        try:
            # JSONデータを解析
            eeg_data = json.loads(json_data)

            # データをフラット化して必要な情報を抽出
            flattened_data = {
                'timestamp': timestamp,
                'attention': eeg_data['eSense']['attention'],  # 注意力
                'meditation': eeg_data['eSense']['meditation'],  # 瞑想
                'delta': eeg_data['eegPower']['delta'],
                'theta': eeg_data['eegPower']['theta'],
                'lowAlpha': eeg_data['eegPower']['lowAlpha'],
                'highAlpha': eeg_data['eegPower']['highAlpha'],
                'lowBeta': eeg_data['eegPower']['lowBeta'],
                'highBeta': eeg_data['eegPower']['highBeta'],
                'lowGamma': eeg_data['eegPower']['lowGamma'],
                'highGamma': eeg_data['eegPower']['highGamma'],
                'poorSignalLevel': eeg_data['poorSignalLevel']  # 信号レベルの悪さ
            }

            # 抽出したデータをリストに追加
            data.append(flattened_data)
        except json.JSONDecodeError:
            print(f"JSONのデコードエラー: {line}")

# データをDataFrameに変換
EEG = pd.DataFrame(data)

# データの最初の数行を表示
EEG.head()

# lowAlphaとhighAlphaの平均をとって"Alpha"列を作成
EEG['Alpha'] = EEG[['lowAlpha', 'highAlpha']].mean(axis=1)

# lowGammaとhighGammaの平均をとって"Gamma"列を作成
EEG['Gamma'] = EEG[['lowGamma', 'highGamma']].mean(axis=1)

EEG["Beta"] = EEG[["highBeta","lowBeta"]].mean(axis=1)

#使う情報として(注意力{attention}、瞑想力{meditation}、シグナルレベル{poorSignalLevel}を削除)
EEG_data = EEG.drop(columns=['attention',"meditation","poorSignalLevel","lowAlpha", "lowBeta", "highAlpha", "highBeta", "lowGamma", "highGamma"])
EEG_data

# モジュールをインポート
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np

%matplotlib inline

#要約情報の確認
EEG_data.info()

#記述統計量を出力
EEG_data.describe()

x1 = pd.to_datetime(EEG_data["timestamp"])
y1 = EEG_data["delta"]
y2 = EEG_data["theta"]
y3 = EEG_data["Alpha"]
y4 = EEG_data["Gamma"]
y5 = EEG_data["Beta"]

lab1 = "delta"
lab2 = "theta"
lab3 = "Alpha"
lab4 = "Gamma"
lab5 = "Beta"

col1 = "blue"
col2 = "red"
col3 = "green"
col4 = "orange"
col5 = "purple"
col6 = "black"
col7=  "white"

linewidth = 2
linestyle = "-"
marker = "o"

fontsize1 = 14
fontsize2 = 12

upper_right="upper right"

left=0.15
right=0.95
top=0.95
bottom=0.15


import matplotlib.dates as mdates

# グラフの設定
plt.figure(figsize=(8, 6))

# 折れ線グラフの描画
plt.plot(x1, y1, label=lab1, color=col1, linewidth=linewidth, linestyle=linestyle)
plt.plot(x1, y2, label=lab2, color=col2, linewidth=linewidth, linestyle=linestyle)
plt.plot(x1, y3, label=lab3, color=col3, linewidth=linewidth, linestyle=linestyle)
plt.plot(x1, y4, label=lab4, color=col4, linewidth=linewidth, linestyle=linestyle)
plt.plot(x1, y5, label=lab5, color=col5, linewidth=linewidth, linestyle=linestyle)

# 軸ラベル
plt.xlabel('TIMESTAMP', fontsize=fontsize1)
plt.ylabel('BRAINWAVE', fontsize=fontsize2)

# 軸範囲
plt.ylim(0,)

# 上と右の枠線を削除
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 軸の太さと色
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)

# グリッド
plt.grid(True, linestyle=linestyle, alpha=0.7)

# タイムスタンプの目盛りを30分ごとに設定
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.xticks(pd.date_range(start="2024-07-06 20:26:10", end="2024-07-06 21:04:59", freq='5T'), rotation=45, fontsize=fontsize2)

# 凡例の設定(枠囲い黒、背景白)
plt.legend(fontsize=fontsize2, loc=upper_right, frameon=True, edgecolor=col6, facecolor=col7, framealpha=1)

# 余白調整
plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

# グラフの保存
plt.savefig('line_graph_paper.png', dpi=300, bbox_inches='tight', transparent=False)

# グラフの表示
plt.show()

# 移動平均関数の定義
def moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

window_size = 10  # 例えば、10サンプルのウィンドウ

EEG_data["delta_smooth"] = moving_average(EEG_data["delta"], window_size)
EEG_data["theta_smooth"] = moving_average(EEG_data["theta"], window_size)
EEG_data["Alpha_smooth"] = moving_average(EEG_data["Alpha"], window_size)
EEG_data["Gamma_smooth"] = moving_average(EEG_data["Gamma"], window_size)
EEG_data["Beta_smooth"] = moving_average(EEG_data["Beta"], window_size)

plt.figure(figsize=(10, 6))
plt.plot(EEG_data["timestamp"], EEG_data["delta_smooth"], label='Delta (smooth)')
plt.plot(EEG_data["timestamp"], EEG_data["theta_smooth"], label='Theta (smooth)')
plt.plot(EEG_data["timestamp"], EEG_data["Alpha_smooth"], label='Alpha (smooth)')
plt.plot(EEG_data["timestamp"], EEG_data["Gamma_smooth"], label='Gamma (smooth)')
plt.plot(EEG_data["timestamp"], EEG_data["Beta_smooth"], label='Beta (smooth)')

# 軸ラベル
plt.xlabel('TIMESTAMP', fontsize=fontsize1)
plt.ylabel('BRAINWAVE', fontsize=fontsize2)

# 軸範囲
plt.ylim(0,)

# 上と右の枠線を削除
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 軸の太さと色
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)

# グリッド
plt.grid(True, linestyle=linestyle, alpha=0.7)

# タイムスタンプの目盛りを30分ごとに設定
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))

# 凡例の設定(枠囲い黒、背景白)
plt.legend(fontsize=fontsize2, loc=upper_right, frameon=True, edgecolor=col6, facecolor=col7, framealpha=1)

# 余白調整
plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

# グラフの保存
plt.savefig('line_graph_paper.png', dpi=300, bbox_inches='tight', transparent=False)

plt.xlabel('Timestamp')
plt.ylabel('Brainwave')

plt.legend()

plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# タイムスタンプ列を datetime 型に変換（まだ変換されていない場合）
EEG_data['timestamp'] = pd.to_datetime(EEG_data['timestamp'])

# 5分ごとに平均を計算するため、timestamp 列を設定してリサンプリングを行う
EEG_data_resampled = EEG_data.set_index('timestamp').resample('5T').mean().reset_index()

# 指定した期間でデータをフィルタリング
start_time = "2024-07-06 20:26:10"
end_time = "2024-07-06 21:01:10"
filtered_data = EEG_data_resampled[(EEG_data_resampled['timestamp'] >= start_time) & (EEG_data_resampled['timestamp'] <= end_time)]

# プロット
plt.figure(figsize=(15, 9))
plt.plot(filtered_data['timestamp'], filtered_data["delta_smooth"], label='Delta (smooth)')
plt.plot(filtered_data['timestamp'], filtered_data["theta_smooth"], label='Theta (smooth)')
plt.plot(filtered_data['timestamp'], filtered_data["Alpha_smooth"], label='Alpha (smooth)')
plt.plot(filtered_data['timestamp'], filtered_data["Gamma_smooth"], label='Gamma (smooth)')
plt.plot(filtered_data['timestamp'], filtered_data["Beta_smooth"], label='Beta (smooth)')

# 軸ラベル
plt.xlabel('TIMESTAMP', fontsize=fontsize1)
plt.ylabel('BRAINWAVE', fontsize=fontsize2)

# 軸範囲
plt.ylim(0,)

# 上と右の枠線を削除
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 軸の太さと色
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)

# グリッド
plt.grid(True, linestyle=linestyle, alpha=0.7)

# タイムスタンプの目盛りを5分ごとに設定
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))

# 凡例の設定(枠囲い黒、背景白)
plt.legend(fontsize=fontsize2, loc='upper right', frameon=True, edgecolor='black', facecolor='white', framealpha=1)

# 余白調整
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

# グラフの保存
plt.savefig('line_graph_paper.png', dpi=300, bbox_inches='tight', transparent=False)

plt.show()

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

# 欠損値を平均値で補完するインプターを定義
imputer = SimpleImputer(strategy='mean')

# 使用するカラムのみを選択（欠損値が存在するカラム）
columns_to_cluster = ['delta_smooth', 'theta_smooth', 'Alpha_smooth', 'Gamma_smooth', 'Beta_smooth']

# 欠損値の補完を実施
EEG_data_cleaned = EEG_data[columns_to_cluster]
EEG_data_imputed = imputer.fit_transform(EEG_data_cleaned)


# 各列ごとの欠損値の数を確認
print(EEG_data_cleaned.isnull().sum())

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')  
EEG_data_cleaned_imputed = imputer.fit_transform(EEG_data_cleaned)

# エルボー法のためのクラスタ数の範囲
k_values = range(1, 10)
inertia = []

# 各クラスタ数でのK-meansの誤差（inertia）を計算
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(EEG_data_cleaned_imputed )
    inertia.append(kmeans.inertia_)

# エルボー法のプロット
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia, marker='o')

# 上と右の枠線を削除
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 軸の太さと色
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)

# グリッド
plt.grid(True, linestyle=linestyle, alpha=0.7)


plt.xlabel('Number of clusters (k)', fontsize=fontsize2)
plt.ylabel('Inertia', fontsize=fontsize2)
plt.grid(True)
plt.show()

# K-means クラスタリングの実行 (クラスタ数は3に設定)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(EEG_data_imputed)

# クラスタラベルをデータに追加
EEG_data['cluster'] = kmeans.labels_

import seaborn as sns
import pandas as pd

# クラスタリング用のカラムと補完されたデータを DataFrame に変換
columns_to_cluster = ['delta_smooth', 'theta_smooth', 'Alpha_smooth', 'Gamma_smooth', 'Beta_smooth']
EEG_data_imputed_df = pd.DataFrame(EEG_data_imputed, columns=columns_to_cluster)

# クラスタラベルを DataFrame に追加
EEG_data_imputed_df['cluster'] = kmeans.labels_

# ペアプロットで全ての特徴量のペアをプロット (色分けはクラスタラベル)
sns.pairplot(EEG_data_imputed_df, hue='cluster', palette='viridis')
plt.suptitle('K-means Clustering Pairplot', y=1.02)  # タイトル追加
plt.show()

# 特徴量の最小値を0、最大値をdelta_smoothに合わせて正規化
max_val = EEG_data_imputed_df['delta_smooth'].max()

for col in ['delta_smooth', 'theta_smooth', 'Alpha_smooth', 'Gamma_smooth', 'Beta_smooth']:
    EEG_data_imputed_df[col] = (EEG_data_imputed_df[col] - EEG_data_imputed_df[col].min()) / (EEG_data_imputed_df[col].max() - EEG_data_imputed_df[col].min()) * max_val

# 正規化後のペアプロットを表示
sns.pairplot(EEG_data_imputed_df, hue='cluster', palette='viridis')
plt.suptitle('Normalized K-means Clustering Pairplot', y=1.02)
plt.show()

import pandas as pd

# クラスタごとのデータ数を表示
cluster_counts = EEG_data_imputed_df['cluster'].value_counts()
print("各クラスタに属するデータ数:")
print(cluster_counts)

# クラスタと特徴量の組み合わせごとの出現回数を集計
cluster_combinations = EEG_data_imputed_df.groupby(
    ['cluster', 'delta_smooth', 'theta_smooth', 'Alpha_smooth', 'Gamma_smooth', 'Beta_smooth']
).size().reset_index(name='count')

print("クラスタごとの組み合わせの数:")
print(cluster_combinations)

brainwave_columns = ['delta', 'theta', 'Alpha', 'Gamma',"Beta"]

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


# ピークを検出する高さのしきい値 (必要に応じて調整)
height_threshold = 0.5

# プロットの準備
plt.figure(figsize=(12, 8))

# 脳波ごとにピークを検出し、プロット
for wave in brainwave_columns :
    peaks, _ = find_peaks(EEG_data[wave], height=height_threshold)
    plt.plot(EEG_data["timestamp"], EEG_data[wave], label=f'{wave.capitalize()} Brainwave')
    plt.plot(EEG_data["timestamp"][peaks], EEG_data[wave][peaks], "x", label=f"{wave.capitalize()} Peaks")

# グラフの設定
plt.xlabel('Timestamp')
plt.ylabel('Brainwave Amplitude')
plt.title('Brainwave Peaks Detection')
plt.legend(loc='upper right')
plt.show()

import matplotlib.pyplot as plt

# 各周波数帯域の最小値と最大値を統一
y_min = EEG_data[brainwave_columns].min().min()
y_max = EEG_data[brainwave_columns].max().max()

# プロット作成
plt.figure(figsize=(12, 15))
for i, wave in enumerate(brainwave_columns):
    plt.subplot(len(brainwave_columns), 1, i + 1)
    plt.plot(EEG_data["timestamp"], EEG_data[wave], label=f'{wave.capitalize()} Brainwave')
    plt.xlabel('Timestamp')
    plt.ylabel(wave.capitalize())  # 各周波数帯域のラベル
    plt.ylim(y_min, y_max)  # y軸の範囲を統一
    plt.legend()

# 上と右の枠を削除
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

from scipy import signal

# スペクトログラムのプロット
plt.figure(figsize=(12, 15))
for i, wave in enumerate(brainwave_columns):
    f, t, Sxx = signal.spectrogram(EEG_data[wave], fs=250)  # fsはサンプリング周波数
    plt.subplot(len(brainwave_columns), 1, i+1)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(f'{wave.capitalize()} Spectrogram')
plt.tight_layout()
plt.show()

import seaborn as sns

# ヒートマップのプロット
energy_data = EEG_data[brainwave_columns].apply(np.abs)  # エネルギーとして絶対値を使用
plt.figure(figsize=(12, 6))
sns.heatmap(energy_data.T, cmap='viridis', xticklabels=100, yticklabels=brainwave_columns)
plt.title('Brainwave Energy Heatmap')
plt.xlabel('Time')
plt.ylabel('Frequency Bands')
plt.show()

from scipy.signal import butter, filtfilt

# バンドパスフィルタの設定
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

fs = 250  # サンプリング周波数
band_limits = {
    'delta': (1, 4),         # Delta: 1–4 Hz
    'theta': (4, 8),         # Theta: 4–8 Hz
    'Alpha': (8, 12),        # Alpha: 8–12 Hz
    'Beta': (12, 30),        # Beta: 12–30 Hz
    'Gamma': (30, 100)       # Gamma: 30–100 Hz
}

# フィルタ処理後の信号をプロット
plt.figure(figsize=(12, 15))
for i, wave in enumerate(brainwave_columns):
    lowcut, highcut = band_limits[wave]
    filtered_signal = bandpass_filter(EEG_data[wave], lowcut, highcut, fs)
    peaks, _ = find_peaks(filtered_signal, height=height_threshold)
    plt.subplot(len(brainwave_columns), 1, i+1)
    plt.plot(EEG_data["timestamp"], filtered_signal, label=f'{wave.capitalize()} Filtered')
    plt.plot(EEG_data["timestamp"][peaks], filtered_signal[peaks], "x", label=f"{wave.capitalize()} Peaks")
    plt.title(f'{wave.capitalize()} Filtered Signal and Peaks')
    plt.xlabel('Timestamp')
    plt.ylabel('Amplitude')
    plt.legend()
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# 擬似ラベルの作成
def generate_pseudo_labels(EEG_data):
    conditions = [
        (EEG_data['theta'] > EEG_data['delta']) & (EEG_data['Alpha'] > EEG_data['Beta']),
        (EEG_data['Beta'] > EEG_data['theta']) & (EEG_data['Gamma'] > EEG_data['Alpha']),
        (EEG_data['Gamma'] > EEG_data['Beta']) & (EEG_data['Beta'] > EEG_data['theta'])
    ]
    labels = ['Relaxed', 'Stressed', 'Focused']
    return np.select(conditions, labels, default='Neutral')

# 仮のラベル作成
EEG_data['emotion_label'] = generate_pseudo_labels(EEG_data)

# 特徴量として各周波数帯域のパワーを使用
X = EEG_data[["delta", "theta", "Alpha", "Beta", "Gamma"]].dropna()
y = EEG_data["emotion_label"].dropna()

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# データのスケーリング
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVMモデルの作成とハイパーパラメータチューニング
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# 最適なモデルで評価
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

EEG_data["beta_alpha_ratio"] = EEG_data["Beta"] / EEG_data["Alpha"]
EEG_data["theta_alpha_ratio"] = EEG_data["theta"] / EEG_data["Alpha"]
EEG_data["gamma_beta_ratio"] = EEG_data["Gamma"] / EEG_data["Beta"]
X = EEG_data[["beta_alpha_ratio", "theta_alpha_ratio", "gamma_beta_ratio"]].dropna()

from sklearn.cluster import KMeans

# クラスタリング
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)

# クラスタ結果をデータに追加
EEG_data["stress_cluster"] = clusters

import seaborn as sns
import matplotlib.pyplot as plt

# クラスタごとのBeta/Alpha比率の分布をプロット
sns.boxplot(x="stress_cluster", y="beta_alpha_ratio", data=EEG_data)
plt.show()

# 仮ラベルをつける
EEG_data["stress_label"] = EEG_data["stress_cluster"].map({0: "Low Stress", 1: "High Stress"})

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, EEG_data["stress_label"], test_size=0.3, random_state=42)

# ロジスティック回帰モデル
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# モデル評価
y_pred_stress = logreg.predict(X_test)
print(classification_report(y_test, y_pred_stress))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# EEGデータの読み込み
# EEG_dataは既にDataFrameに格納されていると仮定
# EEG_data = pd.read_csv('path_to_your_data.csv')

# delta, theta, Alpha, Gamma, Betaの列を選択
selected_columns = ['delta', 'theta', 'Alpha', 'Gamma', 'Beta']
EEG_selected = EEG_data[selected_columns]

# 相関行列の計算
correlation_matrix = EEG_selected.corr()

# ヒートマップの作成
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar_kws={"shrink": .8})
plt.title('EEG Signal Correlation Matrix')
plt.show()

# 散布図行列の作成
sns.pairplot(EEG_selected)
plt.suptitle('EEG Signal Pairplot', y=1.02)  # タイトルを上に調整
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

def plot_autocorrelation(data, selected_columns):
    plt.figure(figsize=(15, 10))
    
    for i, column in enumerate(selected_columns):
        plt.subplot(3, 2, i + 1)
        plot_acf(data[column], lags=30, ax=plt.gca())
        plt.title(f'Autocorrelation of {column}')
        
    plt.tight_layout()
    plt.show()

# ここでは、EEG_corr_use_dataを使用することを前提とします。
selected_columns = ['delta', 'theta', 'Alpha', 'Gamma', 'Beta']
plot_autocorrelation(EEG_data, selected_columns)

import pandas as pd
import statsmodels.api as sm

# サンプルデータの作成


# 目的変数と説明変数の設定
X = EEG_selected
EEG_data['Y'] = EEG_data['Beta'] / (EEG_data['Alpha'] + 1e-6)

# 定数項を追加
X = sm.add_constant(X)

# 重回帰モデルの適用
model = sm.OLS(EEG_data['Y'], X).fit()

# 結果の表示
model.summary()









































