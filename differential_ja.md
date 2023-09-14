# Differential Regional Prompter
このスクリプトはRegional Prompterの補助スクリプトであり、Stable Diffusion web-uiのカスタムscript（XYZ plotなどと同じ）として動作します。
このスクリプトではRegional Prompterのpromptによる領域指定を利用して、差分画像の作成や、一貫性を保ったアニメーションなどの作成が可能です。従来のpromptによる領域指定でもある程度の一貫性を保った差分は作成可能でした。しかし、denoiseの課程において指定した領域外でも差が発生してしまい、完全な差分にはなりません。このスクリプトでは初期画像から、promptで指定した領域のみの差分だけを反映させることが可能です。差分を連続しして変化させることでなめらかなアニメーションも作成できます。  
次の画像は`closed eyes` を`eyes`から計算された領域にのみ適用して、このスクリプトを用いて作られた差分です。3枚目はanime gifにしたものです。   
<img src="https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/dsample1.jpg" width="400">  
<img src="https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/dsample2.jpg" width="400">  
<img src="https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/dsample.gif" width="400">   

このように、変化させる場所以外は一貫性を保ったような画像を作成することができます。LoRAのコピー機学習法用の画像など様々な用途に使えるのではないかと思います。  
また、スケジュール機能を使用して簡単なアニメーションを作成することができます。これはRegional Prompter単体で動作し、追加のモジュールなどを必要としません。

## 動作原理
内部ではPrompt EdittingとRegional PrompterのPromptによる領域指定を使用して差分を作成しています。これは元画像との整合性をより高くするために用いています。例えば目を閉じた差分を作りたいときに、closed eyesを追加したプロンプトすると全体が大きく変わってしまう可能性があります。そこで[:closed eyes:4]として4ステップ目からclosed eyesを効かせることで元画像との整合性を得ます。設定画面のstepはこのprompt edittingの開始ステップを示しています。

## 使い方
scirptの中にあるDifferential Regional Prompterを選択します。Regional Prompterはインストールされていれば他に設定する必要はありません。

<img src="https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/gamen.jpg" width="800">  

### Options
Reversを有効にすると動画を逆再生して生成します。必要な理由は後述します。

### Additional Output
動画(mp4)やAnime Gifを生成するかどうかを選択します。選択した場合はOutputフォルダ直下に生成されます。

### Step
Prompt Edittingで使用される開始ステップを指定します。通常4~6ぐらいがちょうどいいです。
### FPS
動画作成時のフレームレートを設定します。anime gif のdurationは1000/FPSで計算されます。
### Schedule
差分プロンプトを入力します。
詳しい解説は使用例を見て下さい。
### mp4 output directory
mp4を出力するディレクトリを記入します。空欄の場合にはoutputフォルダ直下になります。ここに値を入力すると、output直下に指定のディレクトリが作成されます。
### mp4 output filename
mp4のファイルネームを指定します。空欄の場合`drp.mp4`,`drp_1.mp4`...と連番のファイルが作成されます。ここに`test`と記入すると`test.mp4`,`test_1.mp4`のような連番のファイルが作成されます。上書きはされません。
### mp4 output directory
Anime gifを出力するディレクトリを記入します。空欄の場合にはoutputフォルダ直下になります。ここに値を入力すると、output直下に指定のディレクトリが作成されます。
### mp4 output filename
Anime gifのファイルネームを指定します。空欄の場合`drp.gif`,`drp_1.gif`...と連番のファイルが作成されます。ここに`test`と記入すると`test.gif`,`test_1.gif`のような連番のファイルが作成されます。上書きはされません。

### 使用例
ここでは目を閉じた差分を作ることを想定して解説します。
まずはメインプロンプトを通常のプロンプト入力欄に入力します。ここでは
```
a girl in garden face close up, eyes
```
としましょう。ここで重要なのは`eyes`が入力されていることです。これは差分の領域を計算する際に必要です。
次に、Scheduleに次のように入力します。Regional Prompterの設定欄のthresholdには0.6程度を入力します。

```
0
closed eyes;eyes;1.3
```
この状態でGenerateすると最初に紹介したふたつの画像ができあがります。Anime gifオプションを有効にしているとanime gifもできます。
では各設定値について説明します。
```
prompt;prompt for region calculation;weight;step
```
のように「`;`」で区切られた各設定値を各行に入力します。
#### prompt
差分を作成するプロンプト
#### prompt for region
領域計算用のプロンプト
#### weight
プロンプトの強さ
#### step(省略可)
差分のプロンプトが有効になるステップ数

closed eyes;eyes;1.3;4の場合、実行時には[:(closed eyes:1.3):4]というプロンプトが入力されています。

#### 特殊な指定
0  
*10  
step=5  
th=0.45  
ex-on,0.01  
ex-off  
などの指示を入力可能です。
それぞれstep,領域指定用の閾値などを途中で変更可能です。    


ex-onとex-offはextra seedの設定です。なんやねんそれはと言う方もいると思うので説明しますが、seedは1違うと全く異なる画像になることは知っているかと思います。seedは整数値なので、1以下の値をずらすことはできませんが、それを可能にするのがextra seedで、これを使用すると、ほんの少しだけことなる画像を作ることができます。それがなんの役にたつかというと、背景やエフェクトなどに対して有効に働きます。
次の画像は以下の指示によって作られました。
```
0
ex-on,0.005
 ;lightning_thunder;1.00-1.05
```
0.005はエクストラシードの変化量です。これぐらいに設定すると雷がほとばしるようなエフェクトになります。もっと強くすると全く別のシードから作られたような画像になってしまい意味が無くなってしまうので注意して下さい。
<img src="https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/dsample4.gif" width="400">  
lightning_thunderのようにつなげることで複数の単語に対しての領域を指定できます。プロンプト入力欄にも同じくつないだ言葉が入っている必要があります。
1.00-1.05は5パターン描くように指示するために0.01刻みにしています。
```
 ;lightning_thunder;1.00
```
を5回記述しても同じ指示なので1回しか計算されないためです。


```
#smile and blink 　　　   書式にマッチしない行は無視される
0*20                             20フレーム初期画像を表示       
smile;face;1.2;13-6　　  Step 13から6まで減らしながらface領域に(smile:1.2)を指定
smile;face;1.2*10　　　  10フレームface領域にsmileを指定(stepはデフォルト値)
smile;face;1.2;6-13　　  Step 6から13まで増やしながらface領域に(smile:1.2)を指定
0*20                             20フレーム初期画像を表示
closed eyes;eyes;1.4*3   eyes領域に(closed eyes:1.4)を指定
0*20                             20フレーム初期画像を表示
```
同じ設定で生成される画像は再利用されるので、20フレーム指定しても作成されるのは1枚です。上記の場合、20フレームが3回出てきますが、1枚分を使い回します。Step 13-6とStep6-13は同じ物を逆再生しているのでこれも再利用されます。よって合計で作成される画像は1 + 8 + 1の10枚分になります。
