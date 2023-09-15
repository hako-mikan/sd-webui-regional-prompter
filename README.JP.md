# Regional Prompter
![top](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/top.jpg)
- custom script for [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 
- Different prompts can be specified for different regions

- [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 用のスクリプトです
- 垂直/平行方向に分割された領域ごとに異なるプロンプトを指定できます

## Language control / 言語制御
ENGLISH: [![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/main/README.md)

## 更新情報
- 新機能「[差分生成・差分アニメ](differential_ja.md)」
- [APIを通しての利用について](#apiを通した利用方法)
- プロンプトによる領域指定の[チュートリアル](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/main/prompt_ja.md)
- 新機能 : [インペイントによる領域指定](#inpaint) (thanks [Symbiomatrix](https://github.com/Symbiomatrix))
- 新機能 : [プロンプトによる領域指定](#divprompt) 


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

### Split mode
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
Divide mode : Columns
Divide Ratio : 1,2,1,1;2,4,6
Base Ratio : 
```

![2d](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/2d.jpg)

## <a id="inpaint">Mask regions aka inpaint+ (experimental function)</a>
 手描きマスク、またはアップロードされたマスクを使って領域を指定することができるようになりました。
- まず、`Columns` / `Rows` の横にある `mask divide mode` に切り替えていることを確認してください。そうしないと、マスクは無視され、領域は通常通り分割されます。
- キャンバスの幅と高さを希望する画像に合わせて設定し、`create mask area`を押してください。異なる比率やサイズを指定すると、マスクが正確に適用されないことがあります。（インペイントの「リサイズだけ」のように）。
- キャンバス領域に必要な領域の輪郭を描くか、完全に塗りつぶした後、`draw region`を押してください。これにより、マスクに対応する塗りつぶし多角形が追加され、`region` の番号に従って色が付けられます。
- draw region` を押すと、region が +1 ずつ増えていき、次のregion を素早く描画することができます。また、後でマスクを作るためにどのリージョンが使われたかのリストも保持されます。現在、最大で ~360~ 256 のリージョンが使用できます。
- 既存のリージョンに追加するには、以前に使用された色を選択し、通常通り描画することが可能です。現在のところ、新しいマスク領域以外の領域をクリアする方法はありません（そのうちクリア機能は追加されるかもしれません）。
- `make mask`ボタンは、以前に描いたリージョンについて、`region`の番号で指定されたマスクを表示します。マスクはリージョン固有の色によって検出されます。
- リージョンマスクの準備ができたら、いつも通りプロンプトを書きます： 分割比率は無視されます。`base ratio`は各リージョンに適用されます。すべてのオプションがサポートされ、すべての BREAK / ADDX キーワード (ROW/COL は BREAK に変換されるだけです)。アテンションモードとレイテンモードがサポートされています。
- ベースはマスクモードでは特別な変化をします： base が off のとき、色がついていない領域は最初のマスクに追加されます (したがって、最初のプロンプトで埋められるべきです)。base がオンのとき、色のついていないリージョンは base のプロンプトを反映します、色のついたリージョンは通常の base のウェイトを受け取ります。このため、baseはbase weight = 0で、シーン/背景を指定するのに特に便利なツールです。
- 描画の代わりにマスクをアップロードしたい人向けです： この機能はまだ **非常に多くのWIP** であることに注意してください。マスクを適用するためには、すべての色に何らかのタグを付ける必要があります（コードでLCOLOUR変数を変更するか、手動で各色を画像に追加してください）。色はすべて `HSV(degree,50%,50%)` の変形で、 degree (0:360) は以前のすべての色から最大に離れた値として計算されます（そのため、色は容易に区別できます）。最初のいくつかの値は、基本的に 0、180、90、270、45、135、225、315、22.5などです。色の選択によって、どの領域に対応するかが決まります。

![RegionalMaskGuideB](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/RegionalMaskGuideB.jpg)

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


'1,1;2,3,2;3,2,3'を指定してColumnsを選んだ場合、  
![flip](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/msapmle1.png)  
Rowsに変えると  
![flip](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/msapmle2.png)  
flipを有効にすると
![flip](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/msapmle3.png)	

## <a id="divprompt">region specification by prompt (experimental)</a>
プロンプトによる領域指定です。これまでの領域指定では分割された領域に対してプロンプトを設定していました。この領域指定にはいくつかの問題があり、例えば縦に分割した場合、指定したオブジェクトがそこに限定されてしまします。プロンプトによる領域指定では指定したプロンプトを反映した領域が画像生成中に作成され、そこに対応したプロンプトが適用されます。よって、より柔軟な領域指定が可能になります。以下に例を示します。`apple printed`は`shirt`にだけ効果が反映されて欲しいわけですが、shirtには反映されず、林檎の現物が出てきたりするわけです。
```
lady smiling and sitting, twintails green hair, white skirt, apple printed shirt
```
![prompt](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/psample1.png)
そこで`apple printedの強度を1.4にするとこうなるわけです。

![prompt](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/psample4.png)
プロンプトによる領域指定ではshirtに対して領域を計算して、そこに`apple printed`を適用します。
![prompt](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/psample6.png)
```
lady smiling and sitting, twintails green hair, white skirt, shirt BREAK
(apple printed:1.4),shirt
```
![prompt](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/psample2.png)
すると、目的の効果が得られるわけです。これまでの領域指定ではshirtの位置を詳細に指定しなければいけなかったわけですが、その必要がなくなりました。
### つかいかた
### 書式
```
baseprompt target1 target2 BREAK
effect1, target1 BREAK
effect2 ,target2
```
まず、ベースプロンプトを書きます。ベースプロンプトにはマスクを作成する単語（target1、target2）を書きます。次にBREAKで区切ります。次に、target1に対応するプロンプトを書きます。そしてカンマを入力しtarget1を記載します。ベースプロンプトのtargetの順番とBREAKで区切られたtargetの順番は前後しても問題ありません。targetは大まかな単語でも問題なく、例えば`tops`と指定して、`effect`に`red camisole`などと書いてもいいわけです。

```
target2 baseprompt target1  BREAK
effect1, target1 BREAK
effect2 ,target2
```
ベースプロンプトの順番は考慮されません。effectの順番は考慮されます。

### threshold
プロンプトによって作られるマスクの判定に使われる閾値です。これは対象となるプロンプトによって範囲が大きく異なるのでマスクの数だけ設定できます。複数の領域を使うときはカンマで区切って入力して下さい。例えば髪は領域が曖昧になりがちなので小さな値が必要ですが、顔は領域が大きくなりがちなので小さな値が必要です。これはBREAKで区切られた順に並べて下さい。

```
a lady ,hair, face  BREAK
red, hair BREAK
tanned ,face
```
`threshold : 0.4,0.6`
単一の値が入力された場合、すべての領域に同じ値が適用されます。

### Prompt and Prompt-EX
領域がかぶった場合の計算方式です。Promptだと加算されます。Prompt-EXだと順番に上書きされます。つまり、target1とtarget2の領域が重複していた場合、target2の領域が優先されます。target1にtopsを指定してthretholdを小さくして大きな領域にして、target2をbottomsとしてthresholdを大きくすれば良い分離が得られます。この場合、targetは領域が大きい順に記載されるべきです。

### Accuracy
12 x 512 サイズの場合、Attention modeではU-netの深い領域では 8 x 8 で計算されます。これでは小さい領域しては意味をなしません。よって領域の浸食が起きやすくなります。Latentモードでは 64*64で計算されるため領域が厳密になります。
```
girl hair twintail frills,ribbons, dress, face BREAK
girl, ,face
```
Prompt-EX/Attention
![prompt](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/psample5.png)
Prompt-EX/Latent
![prompt](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/psample3.png)



### ベースと共通の違い
```
a girl ADDROMM(or ADDBASE)
red hair BREAK
green dress
```
と言うプロンプトがあった場合、共通の場合には領域1は`a girl red hair`というプロンプトで生成されます。ベースの場合で比率が0.2の場合には` (a girl) * 0.2 + (red hair) * 0.8`というプロンプトで生成されます。基本的には共通プロンプトで問題ありません。共通プロンプトの効きが強いという場合などはベースにしてみてもいいかもしれません。

## APIを通した利用方法
APIを通してこの拡張を利用する場合には次の書式を使います。
```
  "prompt": "green hair twintail BREAK red blouse BREAK blue skirt",
	"alwayson_scripts": {
		"Regional Prompter": {
			"args": [True,False,"Matrix","Vertical","Mask","Prompt","1,1,1","",False,False,False,"Attention",False,"0","0","0",""]
}}
```
`args`の各設定は下の表を参照して下さい。No.は順番に対応します。typeがtextになっている場合は`""`で囲って下さい。3-6のモード設定は3.のモードで選択したモードに対応するサブモード以外は無視されます。17.のマスクは画像データのアドレスを指定して下さい。アドレスは絶対パスか、web-uiルートからの相対パスが利用できます。マスクはマスクの項で指定された色を使用して作成して下さい。

|  No.  |  setting  |choice| type  | default |
| ---- | ---- |---- |----| ----|
|  1  |  Active  |True, False|Bool|False| 
|  2  | debug   |True, False|Bool|False| 
|  3  | Mode  |Matrix, Mask, Prompt|Text| Matrix|
|  4  | Mode (Matrix)|Horizontal, Vertical, Columns, Rows|Text|Columns
|  5  | Mode (Mask)| Mask |Text|Mask 
|  6  | Mode (Prompt)| Prompt, Prompt-Ex |Text|Prompt
|  7 |  Ratios||Text|1,1,1
|  8 |  Base Ratios  |  |Text| 0
|  9 |  Use Base  |True, False|Bool|False| 
|  10 | Use Common |True, False|Bool|False| 
|  11 | Use Neg-Common   |True, False|Bool| False| 
|  12 | Calcmode| Attention, Latent | Text  | Attention
|  13 | Not Change AND   |True, False|Bool|False| 
|  14 | LoRA Textencoder  ||Text|0|  
|  15 | LoRA U-Net   |  | Text  | 0
|  16 | Threshold   |  |Text| 0
|  17 | Mask   |  | Text | 

### 設定例
#### Matrix
```
  "prompt": "green hair twintail BREAK red blouse BREAK blue skirt",
	"alwayson_scripts": {
		"Regional Prompter": {
			"args": [True,False,"Matrix","Vertical","Mask","Prompt","1,1,1","",False,False,False,"Attention",False,"0","0","0",""]
}}
```
結果
![sample](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/asample1.png)  

#### Mask
```
   "prompt": "masterpiece,best quality 8k photo of BREAK (red:1.2) forest BREAK yellow chair BREAK blue dress girl",
	"alwayson_scripts": {
		"Regional Prompter": {
			"args":	[True,False,"Mask","Vertical","Mask","Prompt","1,1,1","",False,True,False,"Attention",False,"0","0","0","mask.png"]
```
使用したマスク
![sample](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/mask.png)  
結果
![sample](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/asample2.png)  

### Prompt
```
 "prompt": "masterpiece,best quality 8k photo of BREAK a girl hair blouse skirt with bag BREAK (red:1.8) ,hair BREAK (green:1.5),blouse BREAK,(blue:1.7), skirt BREAK (yellow:1.7), bag",
	"alwayson_scripts": {
		"Regional Prompter": {
			"args":	[True,False,"Prompt","Vertical","Mask","Prompt-EX","1,1,1","",False,True,False,"Attention",False,"0","0","0.5,0.6,0.5",""]
}}
```
![sample](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/asample3.png)  

### 謝辞
Attention coupleを提案された[furusu](https://note.com/gcem156)氏、Latent coupleを提案された[opparco](https://github.com/opparco)氏、2D生成のコード作成に協力して頂いた[Symbiomatrix](https://github.com/Symbiomatrix)に感謝します。



- 新機能2D領域を追加しました
- 新しい計算方式「Latent」を追加しました。生成が遅くなりますがLoRAをある程度分離できます
- 75トークン以上を入力できるようになりました
- 共通プロンプトを設定できるようになりました
- 設定がPNG infoに保存されるようになりました
