#!/usr/bin/env Rscript
################################################################################
# Step 3: 業種別失業者数のLCA（潜在クラス分析）本体実施
################################################################################
#
# 本スクリプトは、全9業種の労働特性指標（倒産件数・勤続年数・超過実労働時間数・
# 所定内給与額）を対象に、潜在クラスモデル（LCA）を推定する。
#
# 連続変数はビニング処理で離散化し、poLCAで分析。K=2〜6を探索し、
# 各KのBIC・AIC・エントロピー・クラス割当を計算する。
#
# 出力：
# - step3_r_lca_results_by_k.csv（K別統計量）
# - Rコンソール出力（K比較テーブル）
#

library(tidyverse)
library(poLCA)

cat("================================================================================\n")
cat("[Step 3 / R] 業種LCA本体の実施\n")
cat("================================================================================\n\n")

# ======================================================================================
# パス設定
# ======================================================================================
data_processed <- "C:/GitHub/Data-Analysis_competition/SSDSE/03_unemployment_lca_gis/data/processed"
scripts_dir <- "C:/GitHub/Data-Analysis_competition/SSDSE/03_unemployment_lca_gis/scripts"

# ======================================================================================
# セクション1: データ読み込み
# ======================================================================================
cat("[Section 1 / Loading] データを読み込み中...\n")

df_lca_inputs <- read.csv(
  file.path(data_processed, 'step1_lca_inputs.csv'),
  encoding = 'UTF-8'
)

cat(sprintf("✓ LCA入力データ: %d 行 × %d 列\n", nrow(df_lca_inputs), ncol(df_lca_inputs)))
cat(sprintf("✓ 列名: %s\n", paste(colnames(df_lca_inputs), collapse=', ')))

# ======================================================================================
# セクション2: ビニング処理
# ======================================================================================
cat("\n[Section 2 / Binning] 連続変数をビニング処理...\n")

df_binned <- df_lca_inputs %>%
  mutate(
    # 倒産件数を3分位でビニング
    bankruptcy_binned = cut(
      bankruptcy_count,
      breaks = quantile(bankruptcy_count, probs = c(0, 1/3, 2/3, 1), na.rm = TRUE),
      labels = c('1', '2', '3'),
      include.lowest = TRUE
    ) %>% as.numeric(),

    # 勤続年数を3分位でビニング
    tenure_binned = cut(
      勤続年数,
      breaks = quantile(勤続年数, probs = c(0, 1/3, 2/3, 1), na.rm = TRUE),
      labels = c('1', '2', '3'),
      include.lowest = TRUE
    ) %>% as.numeric(),

    # 超過実労働時間数を3分位でビニング
    overtime_binned = cut(
      超過実労働時間数,
      breaks = quantile(超過実労働時間数, probs = c(0, 1/3, 2/3, 1), na.rm = TRUE),
      labels = c('1', '2', '3'),
      include.lowest = TRUE
    ) %>% as.numeric(),

    # 所定内給与額を3分位でビニング
    wage_binned = cut(
      所定内給与額,
      breaks = quantile(所定内給与額, probs = c(0, 1/3, 2/3, 1), na.rm = TRUE),
      labels = c('1', '2', '3'),
      include.lowest = TRUE
    ) %>% as.numeric()
  ) %>%
  dplyr::select(year, industry, bankruptcy_binned, tenure_binned, overtime_binned, wage_binned)

# 確認
cat("✓ ビニング完了\n")
cat(sprintf("✓ ビニング後の行数: %d\n", nrow(df_binned)))
cat(sprintf("✓ 欠損値: %d\n", sum(is.na(df_binned[, -c(1,2)]))))

# ======================================================================================
# セクション3: K=2〜6探索
# ======================================================================================
cat("\n[Section 3 / Model Selection] K=2〜6でLCAを実行...\n")

results_list <- list()

for (k in 2:6) {
  cat(sprintf("  K=%d...", k), file = stdout(), append = TRUE)
  flush(stdout())

  # poLCA実行
  model_k <- poLCA(
    cbind(bankruptcy_binned, tenure_binned, overtime_binned, wage_binned) ~ 1,
    data = df_binned,
    nclass = k,
    maxiter = 5000,
    verbose = FALSE,
    tol = 1e-6,
    na.rm = TRUE
  )

  # 相対エントロピー計算（0-1スケール）
  # エントロピー = -sum(p_ik * log(p_ik)) / log(K)
  probs <- model_k$probs.start  # クラス別の指標プロファイル確率
  entropy_raw <- -sum(model_k$posterior * log(model_k$posterior + 1e-10), na.rm = TRUE) / nrow(df_binned)
  relative_entropy <- entropy_raw / log(k)

  results_list[[as.character(k)]] <- data.frame(
    K = k,
    BIC = model_k$bic,
    AIC = model_k$aic,
    LL = model_k$llik,
    Entropy = relative_entropy,
    N_Params = k * (sum(model_k$nobs - 1) + k - 1)  # 概算パラメータ数
  )

  # クラス割当を保存（後処理用）
  df_binned[[paste0('class_k', k)]] <- model_k$predclass

  cat(sprintf(" BIC=%.1f Entropy=%.3f\n", model_k$bic, relative_entropy))
}

# 結果をデータフレームに統合
df_results <- bind_rows(results_list)

# ΔBICを計算
df_results <- df_results %>%
  mutate(ΔBIC = c(NA, diff(BIC)))

cat("\n✓ K探索完了\n")

# ======================================================================================
# セクション4: 結果表示
# ======================================================================================
cat("\n")
cat(strrep("=", 80), "\n", sep="")
cat("K選定比較表\n")
cat(strrep("=", 80), "\n", sep="")

print(df_results, digits = 3)

# CSVとして保存（後処理用）
write.csv(
  df_results,
  file.path(data_processed, 'step3_r_lca_results_by_k.csv'),
  row.names = FALSE
)
cat("\n✓ step3_r_lca_results_by_k.csv に保存\n")

# クラス割当テーブルも保存
df_assignments_output <- df_binned %>%
  dplyr::select(year, industry, dplyr::starts_with('class_k'))

write.csv(
  df_assignments_output,
  file.path(data_processed, 'step3_r_class_assignments.csv'),
  row.names = FALSE
)
cat("✓ step3_r_class_assignments.csv に保存\n")

cat("\n")
cat(strrep("=", 80), "\n", sep="")
cat("STEP 3 (Rパート) 完了\n")
cat(strrep("=", 80), "\n", sep="")
cat(sprintf("\n次: Pythonスクリプト (step3_lca_postprocess.py) で後処理を実行\n\n"))
