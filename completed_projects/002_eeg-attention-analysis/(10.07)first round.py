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

%matplotlib inline
from scipy.signal import find_peaks

import numpy as np

#要約情報の確認
EEG_data.info()

#記述統計量を出力
EEG_data.describe()

# 脳波帯を列としてbrainwave_columnsと定義
brainwave_columns = ['delta', 'theta', 'Alpha', 'Gamma',"Beta"]
brainwaves = EEG_data[brainwave_columns]
brainwaves

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


# 5分ごとに平均を計算
EEG_data_resampled = EEG_data.resample('5T').mean()

# 指定した期間でデータをフィルタリング
start_time = "2024-07-06 20:26:10"
end_time = "2024-07-06 21:01:10"
filtered_data = EEG_data_resampled[start_time:end_time]

# プロット
plt.figure(figsize=(15, 9))
plt.plot(filtered_data.index, filtered_data["delta_smooth"], label='Delta (smooth)')
plt.plot(filtered_data.index, filtered_data["theta_smooth"], label='Theta (smooth)')
plt.plot(filtered_data.index, filtered_data["Alpha_smooth"], label='Alpha (smooth)')
plt.plot(filtered_data.index, filtered_data["Gamma_smooth"], label='Gamma (smooth)')
plt.plot(filtered_data.index, filtered_data["Beta_smooth"], label='Beta (smooth)')

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

#脳波解析における感情の切り替えの変化や、感情分析、ストレス分析を行います。現状脳波解析における感情の切り替えの変化を解析する方法として特徴量の変動を確認するために移動平均法を行って平滑化しました。A1ではTIMESTAMPがすべて記載されており、A2では5m事に記載されています。これから、k-meansクラスタリングを行います。そこでクラスタ数の設定も適切に行い、クラスタリングを行います。最適な方法としてエルボー法やシルエットスコアを使用します。クラスタリングを行った後は、ピーク解析を行います。行い方は以下の通りです。

ピーク解析: 感情の切り替え時に脳波の特定の周波数帯域が急激に変化する場合があります。これを検出するためにピーク検出アルゴリズムを適用し、その変化点を解析する方法もあります。

後に脳波解析における感情分析、脳波解析におけるストレス解析を行います。

脳波解析における感情解析は以下の通りです。 感情解析では、特定の脳波周波数帯域のパワーが感情状態に対応することが多いです。例えば、次のような関係が知られています：

Alpha波 (8–12 Hz): リラックス状態や穏やかな感情 Beta波 (12–30 Hz): 集中や緊張、ストレス状態 Theta波 (4–8 Hz): 深いリラックス状態や瞑想 Gamma波 (30–100 Hz): 高度な認知処理、集中状態 この情報を基に、各周波数帯域のパワーを特徴量として機械学習モデルを訓練し、感情分類を行います。SVMやニューラルネットワークを用いて、脳波のパターンと感情ラベルを関連付けることが可能です。

脳波解析におけるストレス解析は以下の通りです。 ストレス解析は、Beta波やGamma波など高周波成分の強度に基づいて行われることが多いです。ストレス状態においては、これらの高周波成分が増加する傾向があります。

手法：

パワースペクトル解析: 特にBeta波やGamma波の増加が見られるかどうかを確認し、ストレスの増減を推定します。 ストレス指標: 周波数帯域ごとのパワーの比率（例: Beta/Alpha比率）を計算し、一定の閾値を超える場合にストレス状態にあると判定します。 これをもとに、リアルタイムでストレスを監視し、適切なフィードバックを提供することができます。

