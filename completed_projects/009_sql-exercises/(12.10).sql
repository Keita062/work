CREATE TABLE Customer (
    ID CHAR(5) NOT NULL PRIMARY KEY,  -- 仕入れ先コード
    Name VARCHAR(100) NOT NULL,       -- 仕入れ先名
    Addr VARCHAR(200) NOT NULL,       -- 仕入先住所
    Tel VARCHAR(11) NOT NULL          -- 仕入先電話番号
);


CREATE TABLE Supplier (
    ID CHAR(5) NOT NULL PRIMARY KEY,  -- 仕入先コード
    Name VARCHAR(100) NOT NULL,       -- 仕入先名
    Addr VARCHAR(200) NOT NULL,       -- 仕入先住所
    Tel VARCHAR(11) NOT NULL          -- 仕入先電話番号
);

CREATE TABLE Stock (
    PrdID CHAR(5) NOT NULL PRIMARY KEY,  -- 商品コード (ProductのID)
    Stk INT NOT NULL                     -- 在庫数
);

CREATE TABLE Product (
    ID CHAR(5) NOT NULL PRIMARY KEY,  -- 商品コード
    Name VARCHAR(50) NOT NULL,        -- 商品名
    SupID CHAR(5) NOT NULL,           -- 仕入先コード (SupplierのID)
    PPrc INT NOT NULL,                -- 仕入単価
    SPrc INT NOT NULL,                -- 販売単価
    FOREIGN KEY (SupID) REFERENCES Supplier(ID)
);

CREATE TABLE SalesSlip (
    ID CHAR(5) NOT NULL PRIMARY KEY,  -- 販売伝票番号
    SDate DATE NOT NULL,              -- 販売日
    CustID CHAR(5) NOT NULL,          -- 販売先コード (CustomerのID)
    FOREIGN KEY (CustID) REFERENCES Customer(ID)
);

CREATE TABLE PurchaseSlip (
    ID CHAR(5) NOT NULL PRIMARY KEY,  -- 仕入伝票番号
    PDate DATE NOT NULL,              -- 仕入日
    PrdID CHAR(5) NOT NULL,           -- 商品コード (ProductのID)
    Pqnty INT NOT NULL,               -- 仕入数量
    FOREIGN KEY (PrdID) REFERENCES Product(ID)
);

CREATE TABLE SalesInfo (
    ID CHAR(5) NOT NULL PRIMARY KEY,  -- 販売No
    SSlpID CHAR(5) NOT NULL,          -- 販売伝票番号 (SalesSlipのID)
    PrdID CHAR(5) NOT NULL,           -- 商品コード (ProductのID)
    Sqnty INT NOT NULL,               -- 販売数量
    FOREIGN KEY (SSlpID) REFERENCES SalesSlip(ID),
    FOREIGN KEY (PrdID) REFERENCES Product(ID)
);

INSERT INTO PurchaseSlip (ID, PData, PrdID, Pqnty) VALUES
(1, '2015/1/5', 'A0001', 2),
(2, '2015/1/5', 'A0001', 3),
(3, '2015/1/5', 'A0002', 15),
(4, '2015/1/6', 'A0003', 5),
(5, '2015/1/7', 'A0011', 7),
(6, '2015/1/7', 'A0012', 20),
(7, '2015/1/7', 'A0021', 5),
(8, '2015/1/8', 'A0022', 5),
(9, '2015/1/8', 'A0023', 5),
(10, '2015/1/9', 'A0024', 7),
(11, '2015/1/9', 'A0031', 7),
(12, '2015/1/9', 'A0032', 4);

INSERT INTO Product (ID, Name, SupID, PPrc, SPrc) VALUES
('A0001', '液晶テレビ', 'S0001', 50160, 62700),
('A0002', 'プラズマテレビ', 'S0002', 81440, 101800),
('A0003', 'プロジェクタ', 'S0003', 50160, 62700),
('A0011', 'DVDプレーヤ', 'S0001', 3150, 4500),
('A0012', 'ブルーレイプレーヤ', 'S0004', 9090, 12980),
('A0021', 'MP3プレーヤ', 'S0003', 17040, 21300),
('A0022', 'スピーカ', 'S0001', 20290, 25360),
('A0023', 'CDプレーヤ', 'S0002', 21020, 26720),
('A0024', 'ICレコーダ', 'S0004', 5490, 6860),
('A0031', 'FAX', 'S0003', 14670, 16300),
('A0032', '電話機', 'S0001', 11520, 12800);

INSERT INTO Stock (PrdID, Stk) VALUES
('A0001', 10),
('A0002', 10),
('A0003', 10),
('A0011', 10),
('A0012', 10),
('A0021', 10),
('A0022', 10),
('A0023', 10),
('A0031', 10),
('A0032', 10);

INSERT INTO SalesInfo (ID, SSlpID, PrdID, Pqnty) VALUES
(1, 'D0001', 'A0001', 1),
(2, 'D0001', 'A0024', 4),
(3, 'D0001', 'A0011', 3),
(4, 'D0002', 'A0002', 4),
(5, 'D0002', 'A0003', 1),
(6, 'D0003', 'A0022', 2),
(7, 'D0003', 'A0021', 3),
(8, 'D0004', 'A0012', 2),
(9, 'D0005', 'A0023', 3),
(10, 'D0006', 'A0022', 2),
(11, 'D0007', 'A0002', 1),
(12, 'D0007', 'A0011', 1),
(13, 'D0008', 'A0031', 3);

INSERT INTO SalesSlip (ID, SData, CustID) VALUES
('D0001', '2015/1/10', 'H0001'),
('D0002', '2015/1/10', 'H0002'),
('D0003', '2015/1/10', 'H0003'),
('D0004', '2015/1/10', 'H0005'),
('D0005', '2015/1/10', 'H0006'),
('D0006', '2015/1/10', 'H0007'),
('D0007', '2015/1/10', 'H0008'),
('D0008', '2015/1/15', 'H0004');

INSERT INTO Customer (ID, Name, Addr, Tel) VALUES
('H0001', '岸田商店', '栃木県宇都宮市', '123-234-3456'),
('H0002', '勝又電気', '千葉県千葉市', '234-345-4567'),
('H0003', '佐野電器', '長崎県長崎市', '345-456-5678'),
('H0004', '堀商店', '大阪府大阪市', '456-678-6789'),
('H0005', '光谷商店', '東京都新宿区', '567-789-8901'),
('H0006', '名電気商会', '神奈川県横浜市', '789-890-9012'),
('H0007', '向井電気', '福島県福島市', '987-876-7654'),
('H0008', '松下商会', '富山県富山市', '765-654-5432');
