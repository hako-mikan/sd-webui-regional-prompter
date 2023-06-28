## promptで指定する領域のチュートリアル
あらかじめ領域を指定するタイプの方法には限界があります。複雑な形状や動的な構図を指定する場合には領域指定が足かせになるためです。promptで指定する領域では画像を生成し始めたあとで領域を決定します。これにより構図や複雑な領域にも対応できるようになります。

では例を見てみましょう。
下記の画像は次のプロンプトによって作成されました。まぁ盛大に色移りするわけです。
```
sfw (8k realistic masterpiece:1.3) a Asian girl ,dark green dress,pink belt,yellow bag,
blond hair, in rainy street, holding red umbrella
```
![1](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/ptutorial9.png)  
さてこれを通常の領域指定でなんとかしようとすると傘やバッグの領域指定に困ってしまうわけです。色々な場所に出てくるし、そもそも領域が小さかったりします。

プロンプトによる領域指定では1単語に対応する領域を計算します。

まずは傘から赤くしてみましょう。プロンプトを以下のように変更します。`red umbrella`を`umbrella`に変更し、`BREAK`のあとに`(red:1.7)`, `umbrella`を追記します。これは`BREAK`のあとに続くプロンプトでは最後のカンマのあとに書かれた単語の領域を計算する仕組みだからです。`(red:1.7), umbrella`の場合、`umbrella`の領域が計算され、その領域に`(red:1.7)`が掛かります。
強度調節はとても大切で、通常1.7を入力すると崩壊気味になるわけですが、プロンプトによる領域指定ではこれぐらいの値を入れないと効きません。特にあまり学習していないような色を指定しようとするなら強度を高めたほうが良いです。
```
sfw (8k realistic masterpiece:1.3) a girl, (dress:1.2), belt, bag, hair, in rainy street, holding umbrella BREAK
(red:1.7), umbrella  
```

```
Divide mode : Prompt-EX
Calcmode : Attention
threshold : 0.7
negative common prompt : Enable
```
Prompt-EXモードは複数の領域を指定する場合に有効なモードで、あとに来る領域によって領域を上書きする効果があります。よって大きな順に領域を指定すると効果的です。
するとちゃんと傘が赤くなりました。２枚目の画像は計算された領域でです。ちゃんと傘の形になっていて、かつ頭の部分は領域外になっているわけです。この領域はpromptによってまちまちなのでThresholdで調節してあげる必要があります。Thresholdは小さいと領域が広くなります。  
![1](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/ptutorial11.png)

ここではネガティブプロンプトも設定しています。
```
nsfw, (worst quality:1.6), (low quality:1.6), (normal quality:1.6), monochrome 
[(black:1.5)::3] BREAK BREAK (dark,transparent, black, blue:2)  
```
傘やバッグなどはもともと黒いものが多く学習されている傾向があるので色を指定するときには注意が必要です。ここでは傘の領域に黒くなるのを防ぐpromptを入れています。`BREAK`が２つ並んでいるのはnegative commom promptを有効にしているためです。`[(black:1.5)::3]`はpromptによる領域指定が始まる前の段階で黒くなるのを防いでいます。promptによる領域指定では3stepまでは領域計算ができていないので有効になっていません。

さて、同様の領域指定を行うことでプロンプトは下記のようになり、ちゃんと色分けできた結果が得られました。

```
sfw (8k realistic masterpiece:1.3) a girl, (dress:1.2), belt, bag, hair, in rainy street, holding umbrella BREAK
(red:1.7), umbrella  BREAK
(dark green:1.7) ,dress BREAK
(blond:1.7), hair BREAK
(pink:1.7), belt BREAK
(yellow:1.7), bag
```

![1](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/ptutorial10.png)

  　
  
次の絵は以下のプロンプトで作られました。forestはT-シャツにだけ書いてくれればいいものの、背景まで森になっています。
```
girl in street (forest printed:1.3) T-shirt, shortshorts daytime
```
![1](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/ptutorial1.png)
そこで、Tシャツのみにforest printetを効かせるためにプロンプトとRegional Prompterを以下のように設定します。
```
lady in street shirt,shortshorts daytime BREAK
(forest printed:1.3) T-shirt ,shirt
```
ここで大切なのはshirtがBREAKの前に入っていることと、BREAKのあと、最後にあることとカンマで区切られていることです。promptモードでは最後にカンマで区切られた単語を領域計算の対象にします。
```
Divide mode : Prompt
Calcmode : Attention
threshold : 0.7
```
divide ratioとbase ratioはいまのところ使用しません。
この設定で生成すると次のようになります。  
![2](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/ptutorial2.png)  
実際に生成されたマスクはこんな感じです。    
![3](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/ptutorial3.png)    
これはAttention modeの性質で、実際には12×8程度にまで縮小されて曖昧になるのでむしろこれくらいの方がよかったりします。

さて、なんだかshortshortにへんなひもがついているので次はここを変えてみましょう。プロンプトを以下のように書き換えます。
```
girl in street shirt,shortshorts daytime BREAK
(forest printed:1.3) T-shirt ,shirt BREAK
(skirt:1.7) ,shortshorts
```
```
Divide mode : Prompt
Calcmode : Attention
threshold : 0.7,0.75
```
するとこうなります。  
![4](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/ptutorial4.png)    
`shortshorts`が`skirt`になりました。今回は最初がshortshortsだったのでそのまま書き換えましたが、`bottoms`などのような単語を使った方が領域選択が簡単です。なぜ今回`shortshors`のままにしたかというと、ベースとなるプロンプトを変えたくなかったからです。`shortshorts`を`bottoms`に変えてしまうと、初期画像そのものが変わってしまうのです。

```
girl in street shirt,bottoms daytime BREAK
(forest printed:1.3) T-shirt ,shirt BREAK
(red skirt:1.7) ,bottoms
```
で作った画像を三番目に置きました。  
![5](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/ptutorial5.png)   
shortshortsのままで作成した場合はshortshorts周辺だけが変わっていますが、shortshortsをbottomsに変えた場合はベースが変わることになるので全体が変わってしまいます。つまり、プロンプト指定では一部だけを変更するインペイントのようなことができるというわけです。

少し設定を変えて着せ替えてみましょう。`(shortshorts:0.5)`としているのは、skirtなどを優先するためです。領域用のshortshortsは弱めても問題ありません。普通は同じものを対象としますが、強い単語だと影響が出すぎてしまうので弱めるのも手です。

```
girl in street shirt,shortshorts daytime BREAK
(forest printed:1.3) T-shirt ,shirt BREAK
(red skirt:1.5) ,(shortshorts:0.5)
```
```
Divide mode : Prompt
Calcmode : Attention
threshold : 0.7,0.55
```
`long skirt`などに対応するために`shortshorts`の`threshold`を広くしています。bottoms やlower bodyの方が楽だと思います。
![6](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/ptutorial6.png)  
着せ替えできましたが、pinkは浸食が出ていますね。これは強度の指定を一括で行っているからで、設定を詰めれば浸食はなくなりはずです。

さて次は背景を変えてみましょう。

```
girl in street shirt,shortshorts daytime BREAK
(forest printed:1.3) T-shirt ,shirt BREAK
(red skirt:1.5) ,(shortshorts:0.5) BREAK
(Japan landscape:1.6),street
```
```
Divide mode : Prompt
Calcmode : Attention
threshold : 0.7,0.55,0.7
```
Japanの部分を変えてみます。  
![7](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/ptutorial7.png)  
landscapeという単語はなかなかくせ者で、この単語はかなり強いので他の単語の効果を打ち消すことが多々あるわけですが、landscapeの代わりにstreetで領域計算して適用することで強く出過ぎることを抑えることができるわけです。このように、別な単語で領域を計算するということで表現の幅が広がるのではないでしょうか。

Regional Prompterを無効にするとこうなります。
これはこれでいいですね。
![8](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/ptutorial8.png)    
