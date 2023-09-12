# Differential Regional Prompter
このスクリプトはRegional Prompterの補助スクリプトであり、Stable Diffusion web-uiのカスタムscript（XYZ plotなどと同じ）として動作します。
このスクリプトではRegional Prompterのpromptによる領域指定を利用して、差分画像の作成や、一貫性を保ったアニメーションなどの作成が可能です。従来のpromptによる領域指定でもある程度の一貫性を保った差分は作成可能でした。しかし、denoiseの課程において指定した領域外でも差が発生してしまい、完全な差分にはなりません。このスクリプトでは初期画像から、promptで指定した領域のみの差分だけを反映させることが可能です。差分を連続しして変化させることでなめらかなアニメーションも作成できます。  
次の画像は`closed eyes` を`eyes`から計算された領域にのみ適用して、このスクリプトを用いて作られた差分です。3枚目はanime gifにしたものです。   
<img src="https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/dsample1.jpg" width="300">
<img src="https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/dsample2.jpg" width="300">
<img src="https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/dsample.gif" width="300">

このように、変化させる場所以外は一貫性を保ったような画像を作成することができます。LoRAのコピー機学習法用の画像など様々な用途に使えるのではないかと思います。  
また、スケジュール機能を使用して簡単なアニメーションを作成することができます。これはRegional Prompter単体で動作し、追加のモジュールなどを必要としません。

## 動作原理
内部ではPrompt EdittingとRegional PrompterのPromptによる領域指定を使用して差分を作成しています。これは元画像との整合性をより高くするために用いています。例えば目を閉じた差分を作りたいときに、closed eyesを追加したプロンプトすると全体が大きく変わってしまう可能性があります。そこで[:closed eyes:4]として4ステップ目からclosed eyesを効かせることで元画像との整合性を得ます。設定画面のstepはこのprompt edittingの開始ステップを示しています。

## 使い方
scirptの中にあるDifferential Regional Prompterを選択します。Regional Prompterはインストールされていれば他に設定する必要はありません。
### Step
Prompt Edittingで使用される開始ステップを指定します。
### FPS
動画作成時のフレームレートを設定します。
### Schedule
差分プロンプトを入力します。
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

closed eyes;eyes;1.3の場合、実行時には[:(closed eyes:1.3):4]というプロンプトが入力されています。


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
