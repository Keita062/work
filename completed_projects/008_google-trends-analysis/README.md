# Google トレンド検索数の時系列分析

元フォルダ: `2024_11 〜 2025_1`

Google トレンドのCSVを使った検索ボリュームの分析。分量が一番多い一連の作業。

- 主対象はスニーカー4系統（AIR JORDAN / AIR FORCE / AIR MAX / DUNK）
- `Help(11.25).ipynb` のみスターバックスの検索数を月次集計したもの
- 週次データに year / month / day / week_number 列を付与 → 記述統計 → 相関
- 前日差（fluctuation）の算出、対数変換・標準化、一元配置分散分析（f_oneway）まで
