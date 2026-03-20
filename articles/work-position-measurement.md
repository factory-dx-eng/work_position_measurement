---
title: "搬送ライン上のワーク位置ずれをPythonで定量化する：テンプレートマッチング実践"
emoji: "📏"
type: "tech"
topics: ["python", "opencv", "画像処理", "製造業", "DX"]
published: false
---

# 搬送ライン上のワーク位置ずれをPythonで定量化する

## 現場でこんな状況ありませんか？

搬送ラインでワークの停止位置がばらつく。それ自体はよくあることです。問題は、**そのずれがどれくらいなのか、誰も数値で把握していない**ことです。

ずれが蓄積すると次工程への受け渡し精度が落ち、加工・組付け精度にも影響が出てきます。当然、定期的な設備調整が必要になります。ところが現場では「なんとなくずれてきたら調整する」という運用になりがちで、調整のタイミングも基準もベテランの感覚任せになっています。

品質不良が発生したときも困ります。「搬送の停止位置のばらつきが原因か、次工程の加工条件が原因か」の切り分けができないのです。

**ずれ量を数値化する。それだけで、調整の属人化と原因特定の難しさ、この2つに同時に向き合えるようになります。**

この記事では、カメラ1台とPythonでワークのXY方向のずれ量をmm単位で定量化する方法を紹介します。

---

## アイデアの核心

使う技術は**テンプレートマッチング**です。

```
基準画像（正常位置のワーク）を1枚だけ登録しておく
　↓
毎回の検査画像と比較して「どこにずれたか」をピクセルで取得
　↓
ピクセル → mm に換算して出力
```

難しいアルゴリズムは不要です。センサーの後付けも設備改造も要りません。カメラを固定できる場所があれば、今日から試せます。

---

## 環境

```
Python 3.9+
opencv-python
numpy
matplotlib
```

```bash
pip install opencv-python numpy matplotlib
```

---

## 実装

コードは3つのブロックで構成しています。

### ブロック1：画像の読み込み

実際の現場ではカメラで撮影した画像を使います。リポジトリには動作確認用のサンプル画像を `samples/` に同梱しています。ファイル名にずれ量（例：`test_dx+3.0_dy+0.0.png`）を含めているので、実行結果と見比べやすくなっています。

![サンプル画像一覧](https://storage.googleapis.com/zenn-user-upload/e705a4ab1b0e-20260321.png)

```python
import cv2
import numpy as np
import csv
from pathlib import Path
from datetime import datetime

# ============================================================
# 設定パラメータ
# ============================================================
PX_PER_MM = 10.0    # キャリブレーション係数 (px/mm)
                    # 求め方：既知寸法(mm) ÷ 画像上のピクセル数
                    # 例：100mmの治具が200pxで写っていれば 200/100=2.0

MARGIN = 80         # テンプレート切り出し時の余白(px)
                    # 検出可能な最大ずれ量 = MARGIN / PX_PER_MM (mm)

SCORE_THRESHOLD = 0.85  # マッチングスコアの閾値

REFERENCE_PATH = "samples/reference.png"
TEST_DIR       = "samples"
LOG_PATH       = "offset_log.csv"


def load_image(path):
    """グレースケールで画像を読み込む"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"画像が見つかりません: {path}")
    return img


def load_test_images(test_dir, reference_path):
    """検査画像を一括読み込みする（基準画像を除く）"""
    ref_name = Path(reference_path).name
    test_paths = sorted(
        p for p in Path(test_dir).glob("*.png")
        if p.name != ref_name
    )
    return [{"path": str(p), "image": load_image(str(p))} for p in test_paths]
```

### ブロック2：テンプレートマッチングでずれ量（px）を取得

ポイントは「基準画像の**中央部だけ**をテンプレートとして切り出す」ことです。

画像全体をテンプレートにしてしまうと、ずれを探す余地がなくなります。`margin` 分の余白を残すことで「この範囲内のずれを探す」という検出窓が生まれます。

下図の①が基準画像（緑枠がテンプレート切り出し範囲）、②が切り出したテンプレート、③が検査画像です。③では橙枠がマッチした位置・緑の点線が基準位置を示しています。

![テンプレートマッチングの処理フロー](https://storage.googleapis.com/zenn-user-upload/0031704dbdf2-20260321.png)


```python
def measure_offset_px(ref_img, test_img, margin=MARGIN):
    """
    テンプレートマッチングでずれ量(px)を返す

    margin が大きいほど検出できるずれ量の上限が上がる
    margin=80px・PX_PER_MM=10 のとき → 最大±8mmまで検出可能
    """
    h, w = ref_img.shape

    # 基準画像の中央部をテンプレートとして切り出す
    tmpl = ref_img[margin:h - margin, margin:w - margin]

    # テンプレートマッチング（正規化相関係数法）
    # TM_CCOEFF_NORMED は照明の明るさ変動にやや強い
    result = cv2.matchTemplate(test_img, tmpl, cv2.TM_CCOEFF_NORMED)

    # スコアが最大の位置を取得
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # ずれなし（基準）のとき max_loc は (margin, margin) になる
    # そこからのずれがワークのXYオフセット
    dx_px = max_loc[0] - margin
    dy_px = max_loc[1] - margin

    return dx_px, dy_px, max_val
```

### ブロック3：px→mm換算して結果を表示・保存

```python
def px_to_mm(dx_px, dy_px, px_per_mm=PX_PER_MM):
    """ピクセルをmmに換算する"""
    return dx_px / px_per_mm, dy_px / px_per_mm


def save_log(log_path, filename, dx_mm, dy_mm, score):
    """計測結果をCSVに追記する"""
    write_header = not Path(log_path).exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "filename", "dx_mm", "dy_mm", "score"])
        writer.writerow([datetime.now().isoformat(), filename, dx_mm, dy_mm, score])


def run(reference_path=REFERENCE_PATH, test_dir=TEST_DIR):
    ref_img = load_image(reference_path)
    test_images = load_test_images(test_dir, reference_path)

    print("=" * 60)
    print(f"{'ファイル名':<20} {'dX(mm)':>8} {'dY(mm)':>8}  {'スコア':>6}  {'判定':>4}")
    print("-" * 60)

    for case in test_images:
        dx_px, dy_px, score = measure_offset_px(ref_img, case["image"])
        dx_mm, dy_mm = px_to_mm(dx_px, dy_px)
        status = "OK" if score >= SCORE_THRESHOLD else "警告"
        filename = Path(case["path"]).name

        print(f"{filename:<20} {dx_mm:>8.2f} {dy_mm:>8.2f}  {score:>6.3f}  {status:>4}")
        if score < SCORE_THRESHOLD:
            print(f"  警告: スコア低下 - 回転ずれ・ワーク欠け等の可能性があります")

        save_log(LOG_PATH, filename, dx_mm, dy_mm, score)

    print("=" * 60)
    print(f"\nログ保存: {LOG_PATH}")


if __name__ == "__main__":
    run()
```

---

## 実行結果

```
基準画像: samples/reference.png
検査画像: 4件

============================================================
ファイル名                  dX(mm)   dY(mm)     スコア    判定
------------------------------------------------------------
test_dx+0.0_dy-2.0.png     0.00    -2.00   1.000    OK
test_dx+3.0_dy+0.0.png     3.00     0.00   1.000    OK
test_dx+4.0_dy+3.0.png     4.00     3.00   1.000    OK
test_dx-3.0_dy+2.5.png    -3.00     2.50   1.000    OK
============================================================

ログ保存: offset_log.csv
```

全4ケースでずれ量を正確に計測できました。右のベクトル図では、矢印の方向と長さでXY方向のずれが視覚的に把握できます。

![計測結果とずれベクトルの可視化](https://storage.googleapis.com/zenn-user-upload/c43ca3c885f0-20260321.png)

---

## 現場に適用するときのポイント

### キャリブレーション（px/mm換算係数の求め方）

`PX_PER_MM` は現場のカメラ環境に合わせる必要があります。求め方はシンプルです。

```
寸法が既知のワーク（または治具）をカメラで撮影する
　↓
その寸法が何ピクセルに写っているか計測する
　↓
PX_PER_MM = ピクセル数 ÷ 実寸法(mm)
```

例：100mmの治具が200pxで写っていれば `PX_PER_MM = 2.0`

### 検出範囲の上限

検出できる最大ずれ量は `margin ÷ PX_PER_MM` です。想定される最大ずれ量に合わせて `margin` を設定してください。

### カメラと照明の固定

テンプレートマッチングはカメラの位置ずれや照明変動に敏感です。現場への導入前に必ず確認したいポイントです。

- カメラはブラケットや専用治具で**完全固定**する
- 照明も固定し、外光が差し込まないよう遮光する
- 照明変動が避けられない場合は `TM_CCOEFF_NORMED`（正規化あり）を使う（今回のコードはこちら）

### ずれ量のログ保存と画像の紐づけ

計測結果をCSVに保存することで、ずれの傾向管理や調整タイミングの判断基準として活用できます。

```python
import csv
from datetime import datetime

with open("offset_log.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([datetime.now(), meas_dx, meas_dy])
```

このとき、**検査画像のファイル名にワークIDや撮影時刻を含める**ことをおすすめします。ログCSVと画像ファイルが同じ名前で紐づくので、不良が発生したときに「いつ・どのワークで・どれくらいずれていたか」をすぐに遡れます。

```python
# ワークID + 撮影時刻をファイル名に含める例
work_id = "W-00123"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{work_id}_{timestamp}.png"
# → W-00123_20240315_143022.png

cv2.imwrite(f"images/{filename}", frame)
```

---

## 実装上の課題と対応アイデア

テンプレートマッチングはシンプルな手法ですが、現場に持ち込む前に想定しておきたい課題があります。

### 課題1：多品種対応でテンプレートの管理が増える

品種ごとに基準画像を1枚ずつ用意する必要があります。品種数が増えると管理が煩雑になりますが、対応自体はシンプルです。

品種コードをキーにした辞書でテンプレートを管理し、品種切替時に参照先を差し替えるだけです。基準画像の登録作業も「正常位置に止まったときに1枚撮影する」だけなので、現場の運用フローに組み込みやすいです。

```python
# 品種コードをキーにテンプレートを管理する例
templates = {
    "PART_A": cv2.imread("templates/PART_A.png", cv2.IMREAD_GRAYSCALE),
    "PART_B": cv2.imread("templates/PART_B.png", cv2.IMREAD_GRAYSCALE),
}

# 品種切替時はキーを変えるだけ
current_part = "PART_A"
ref_img = templates[current_part]
```

### 課題2：照明変動による誤検出

工場内の照明は時間帯や季節によって変化します。照明条件が変わると輝度パターンが変化し、マッチング精度が落ちることがあります。

**まず試すこと：`TM_CCOEFF_NORMED` を使う（今回のコードはこちら）**

輝度の絶対値ではなく相関係数で比較するため、明るさの変動にある程度対応できます。

**それでも誤検出が出る場合：画像を正規化してからマッチング**

```python
def normalize(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

result = cv2.matchTemplate(
    normalize(test_img),
    normalize(tmpl),
    cv2.TM_CCOEFF_NORMED
)
```

**根本対策：照明を固定・遮光する**

ソフトウェアで吸収しようとするより、照明環境を安定させる方が確実です。外光の影響が大きい場合は遮光カバーの設置も検討してください。

### 課題3：ワークに回転ずれが発生する場合

テンプレートマッチングは並進（XY移動）のずれ検出を前提としています。ワークが回転してしまうと、マッチングスコアが下がり検出精度も落ちます。

**まず確認すること：スコアを監視して回転ずれを検知する**

スコアが閾値を下回るときは「回転ずれを含む異常」として扱い、後続処理で除外するか警告を出す運用が現実的な第一歩です。

```python
_, max_val, _, max_loc = cv2.minMaxLoc(result)

SCORE_THRESHOLD = 0.85  # 要調整
if max_val < SCORE_THRESHOLD:
    print(f"警告：マッチングスコア低下 ({max_val:.3f})。回転ずれの可能性あり")
```

**回転量も推定したい場合：回転済みテンプレートで総当たりマッチング**

基準画像を複数の角度に回転させたテンプレートを用意し、最もスコアが高い角度を採用する方法です。既存のテンプレートマッチングをそのまま使い回せるため、実装の敷居が低いです。

```python
def measure_offset_with_rotation(ref_img, test_img, angle_step=5, margin=80):
    """
    回転済みテンプレートを総当たりしてずれ量と回転角を返す
    angle_step: 探索角度の刻み幅(deg)。細かいほど精度は上がるが処理時間も増える
    """
    h, w = ref_img.shape
    cx, cy = w / 2, h / 2
    best_score, best_angle, best_loc = -1, 0, (margin, margin)

    for angle in range(-180, 180, angle_step):
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(ref_img, M, (w, h), borderValue=200)
        tmpl = rotated[margin:h - margin, margin:w - margin]
        result = cv2.matchTemplate(test_img, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(result)
        if score > best_score:
            best_score, best_angle, best_loc = score, angle, loc

    dx_px = best_loc[0] - margin
    dy_px = best_loc[1] - margin
    return dx_px, dy_px, best_angle
```

5度刻みなら最大±2.5度の誤差です。精度よりも**手軽さ・シンプルさ**を優先したい最初の一手として有効です。

**より高精度に回転も検出したい場合：ECC マッチングに切り替える**

`cv2.findTransformECC` に `MOTION_EUCLIDEAN` を指定すると、並進＋回転を同時に推定できます。計算コストは上がりますが、±50度以内の回転なら高精度に検出できます。

```python
warp_matrix = np.eye(2, 3, dtype=np.float32)
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7)

_, M = cv2.findTransformECC(
    ref_img.astype(np.float32),
    test_img.astype(np.float32),
    warp_matrix,
    cv2.MOTION_EUCLIDEAN,
    criteria,
    None,
    5,  # ガウシアンフィルタサイズ
)

# 回転角の取得
angle_deg = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
```

---

## まとめ

| 項目 | 内容 |
|---|---|
| 使用技術 | OpenCV テンプレートマッチング |
| 必要なもの | カメラ（固定）、Python環境 |
| 出力 | XY方向のずれ量（mm）＋マッチングスコア＋CSVログ |
| 設備改造 | 不要 |

テンプレートマッチングは「登録した基準画像と比べる」だけのシンプルな原理なので、不定形ワークにも業種を問わず使えます。

「どれくらいずれているか分からない」状態に数値を与えること。それが、調整の属人化を崩す最初の一手です。まずはサンプルコードを手元で動かすところから試してみてください。

---

## GitHubリポジトリ

サンプルコード・サンプル画像一式を公開しています。

🔗 [github.com/factory-dx-eng/work_position_measurement](https://github.com/factory-dx-eng/work_position_measurement)

---
