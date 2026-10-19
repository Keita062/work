# 病欠届提出のシーケンス図

以下のシーケンス図は、学生が病欠届を提出するプロセスを示しています。

```mermaid
sequenceDiagram
    actor s as 学生
    actor t as 教員
    actor d as 医師
    participant db as 出席データベース

    s ->> d : 診断書発行依頼
    d ->> s : 診断書発行
    s ->> t : 診断書提出
    t ->> db : 病欠登録
    db -->> t : 登録完了
