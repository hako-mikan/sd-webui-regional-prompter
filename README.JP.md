# Regional Prompter
![top](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/top.jpg)
- custom script for [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 
- Different prompts can be specified for different regions

- [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 用のスクリプトです
- 垂直/平行方向に分割された領域ごとに異なるプロンプトを指定できます

## Language control / 言語制御
ENGLISH: [![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/main/README.md)

## 更新情報
- 新機能2D領域を追加しました
- 新しい計算方式「Latent」を追加しました。生成が遅くなりますがLoRAをある程度分離できます
- 75トークン以上を入力できるようになりました
- 共通プロンプトを設定できるようになりました
- 設定がPNG infoに保存されるようになりました

[Symbiomatrix](https://github.com/Symbiomatrix)氏の協力によりより[柔軟な領域指定](#2次元領域指定実験的機能)が可能になりました。


# 概要
Latent couple extentionではプロンプトごとにU-Netの計算を行っていますが、このエクステンションではU-Netの内部でプロンプトごとの計算を行います。詳しくは[こちら](https://note.com/gcem156/n/nb3d516e376d7)をご参照ください。アイデアを発案されたfurusu様に感謝いたします。

## 使い方
次の画像の作り方を解説しつつ、使い方を説明します。  
![sample](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/sample.jpg)  
以下がプロンプトです。
```
green hair twintail BREAK
red blouse BREAK
blue skirt
```
設定
```
Active : On
Use base prompt : Off
Divide mode : Vertical
Divide Ratio : 1,1,1
Base Ratio : 
```
この設定では縦方向に三分割し、上から順にgreen hair twintail ,red blouse ,blue skirtというプロンプトを適用しています。
### Active  
ここにチェックが入っている場合有効化します。

### Prompt
領域別のプロンプト同士はBREAKで区切ります。水平の場合は左から、垂直の場合は上から順にプロンプトを入力します。
ネガティブプロンプトもBREAKで区切ることで領域ごとに設定できますが、BREAKを入力しない場合すべての領域に同一のネガティブプロンプトが設定されます。

### Use base prompt
ベースプロンプトとはすべての領域に共通のプロンプトを使用したい場合チェックを入れます。領域で一貫した場面にしたい場合などは使ってください。
ベースプロンプトを使用する場合、BREAK区切られた最初のプロンプトがベースとして扱われます。
ADDBASEが入力された場合、自動的にオンになります。

### Base ratio
ベースプロンプトの比率を設定します。0.2と入力された場合、ベースの割合が0.2になります。領域ごとにも指定可能で、0.2,0.3,0.5などと入力できます。単一の値を入力した場合はすべての領域に同じ値が適応されます。

### Divide ratio
領域の広さを指定します。1,1,1と入力した場合、三分割されます(33,3%,33,3%,33,3%)。3,1,1と入力した場合60%,20%,20%となります。小数点でも入力可能です。0.1,0.1,0.1は1,1,1と同じ結果になります。

### calcutation mode  
#### Attention  
通常はこちらを使用して下さい
#### Latent
LoRAを分離したい場合こちらを使用して下さい。生成時間は長くなりますが、ある程度LoRAを分離できます。

[ねんどろいど](https://civitai.com/models/7269/nendoroid-figures-lora),
[figma](https://civitai.com/models/7984/figma-anime-figures)LoRAを左右に分離して作成した例。  
<img src="https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/sample2.jpg" width="400">

~~三月末のweb-uiのアップデートでLoRAの適用方法が変更され、これにより生成時間が大幅に長くなります。アップデートに不具合があるというわけでは無く、普通の使い方をするなら生成時間を短縮する効果がありますが、領域別適応をする段においては逆効果になるようです。いくつか対策を考えてみましたがいまのところ回避策は思い浮かびません。~~

### Divide mode
分割方向を指定します。水平、垂直方向が指定できます。

### Use common prompt
このオプションを有効化すると最初のプロンプトをすべてのプロンプトに加算します。
`ADDCOMM`が入力された場合自動的にオンになります。
```
best quality, 20yo lady in garden BREAK
green hair twintail BREAK
red blouse BREAK
blue skirt
```
このようなプロンプトがあるときに、この機能を有効化すると以下のように扱われます。
```
best quality, 20yo lady in garden, green hair twintail BREAK
best quality, 20yo lady in garden, red blouse BREAK
best quality, 20yo lady in garden, blue skirt
```
よって、3つの領域に分ける場合4つのプロンプトをセットする必要があります。Use base promptが有効になっている場合は5つ必要になります。設定順はcommon,base, prompt1,prompt2,...となります。

### 2次元領域指定(実験的機能)
領域を2次元的に指定できます。特別なセパレイター(`ADDCOL/ADDROW`)を用いることで領域を縦横に分割することができます。左上を始点として、`ADDCOL`で区切ると横方向、`ADDROW`で区切ると縦方向に分割されます。分割の比率はセミコロンで区切られた比率で指定します。以下に例を示します。`BREAK`のみで記述し、比率のみで記述することも可能ですが、明示的にCOL/ROWを指定した方がわかりやすいです。最初のセパレーターとして`ADDBASE`を使用すると、ベースプロンプトになります。比率を指定しない場合や比率がセパレーターの数と一致しないときは自動的にすべて等倍として処理されます。`ADDCOMM`を最初のセパレーターとして入力した場合共通プロンプトになります。Divide modeで選択された方向は有効であり、上から/左から順に`ADDCOL/ADDROW`が処理されます。

```
(blue sky:1.2) ADDCOL
green hair twintail ADDCOL
(aquarium:1.3) ADDROW
(messy desk:1.2) ADDCOL
orange dress and sofa
```

```
Active : On
Use base prompt : Off
Divide mode : Horizontal
Divide Ratio : 1,2,1,1;2,4,6
Base Ratio : 
```

![2d](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/2d.jpg)

### visualise and make template
複雑な領域指定をする場合など領域を可視化して、テンプレートを作成します。

![tutorial](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/tutorial.jpg)

入力を終えてボタンを押すと、画像のように領域とテンプレートが出力されます。テンプレートをコピペして使用して下さい。以下は入力例と出力結果です。

```
fantasy ADDCOMM
sky ADDROW
castle ADDROW
street stalls ADDCOL
2girls eating and walking on street ADDCOL
street stalls
```

![tutorial](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/sample3.jpg
)

### ベースと共通の違い
```
a girl ADDROMM(or ADDBASE)
red hair BREAK
green dress
```
と言うプロンプトがあった場合、共通の場合には領域1は`a girl red hair`というプロンプトで生成されます。ベースの場合で比率が0.2の場合には` (a girl) * 0.2 + (red hair) * 0.8`というプロンプトで生成されます。基本的には共通プロンプトで問題ありません。共通プロンプトの効きが強いという場合などはベースにしてみてもいいかもしれません。

## 謝辞
Attention coupleを提案された[furusu](https://note.com/gcem156)氏、Latent coupleを提案された[opparco](https://github.com/opparco)氏、2D生成のコード作成に協力して頂いた[Symbiomatrix](https://github.com/Symbiomatrix)に感謝します。

