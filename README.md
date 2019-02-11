# X線画像解析
製薬協タスクフォース「AIってなに？」の報告書で用いたプログラムの実行方法について以下に示す．

## ローカル環境での実行
[X線画像データ](https://nihcc.app.box.com/v/ChestXray-NIHCC)およびプログラム（.pyファイル，.ipybnファイル）を以下のフォルダ構成で保存する．プログラムはGitHubから**`Clone or download`**ボタンをクリックして.zipファイルにしてからダウンロードすることができる．その後`CNN_x-ray.ipynb`を実行する．

```bash
.
├── BBox_List_2017.csv
├── CNN_x-ray.ipynb
├── Data_Entry_2017.csv
├── myGenerator.py
├── myMetrics.py
├── myTrainer.py
├── myUtility.py
├── test_list.txt
├── train_list.txt
└── images
|   ├── 00000001_000.png
|   └── ...
└── logs
```

## Colaboratoryでの実行
[Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb?hl=ja)とはGoogleが提供している，機械学習の教育，研究を目的とした研究用ツールである．完全にクラウドで実行される Jupyterノートブック環境が，設定不要かつ無料で利用することができる．[FAQ](https://research.google.com/colaboratory/faq.html)にあるようにPC 版の Chrome と Firefox では完全に動作するよう検証済みである．なお，起動から12時間を超える，または90分を超えるアイドル時間があると仮想マシンが破棄されるため注意が必要となる．

Googleドライブにx-rayフォルダを作成し，以下のようにファイルを保存しておく．その後，Colaboratory上で`ファイル`から`ノートブックを開く`を選択し`N_x-ray.ipynb`を開き実行する．ファイルのコピーや.tar.gzの展開などはプログラムで実行している．
```bash
x-ray
├── BBox_List_2017.csv
├── Data_Entry_2017.csv
├── myGenerator.py
├── myMetrics.py
├── myTrainer.py
├── myUtility.py
├── test_list.txt
├── train_list.txt
└── images
    ├── images_001.tar.gz
    ├── images_002.tar.gz
    ├── ...
    └── images_012.tar.gz
```
authorization codeを利用してGoogleドライブをCoraboratoryにマウントした上でファイルのコピーを行っている．不明な点については[Googleドライブの読み込みと保存](https://colab.research.google.com/notebooks/io.ipynb)を参照されたい．  
なお，直接ローカルPCからColaboratoryにファイルをアップロードすることも可能であるが，X線画像データのサイズが非常に大きく時間がかかる．繰り返し実行を行うような場合はGoogleドライブにデータを予めアップロードしておいてそれをコピーした方が早いと考えられる．