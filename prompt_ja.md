## promptで指定する領域のチュートリアル
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
shortshortsがskirtになりました。今回は最初がshortshortsだったのでそのまま書き換えましたが、bottomsなどのような単語を使った方が領域選択が簡単です。なぜ今回shortshorsのままにしたかというと、ベースとなるプロンプトを変えたくなかったからです。shortshortsをbottomsに変えてしまうと、初期画像そのものが変わってしまうのです。

```
girl in street shirt,bottoms daytime BREAK
(forest printed:1.3) T-shirt ,shirt BREAK
(red skirt:1.7) ,bottoms
```
で作った画像を三番目に置きました。  
![5](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/ptutorial5.png)   
shortshortsのままで作成した場合はshortshorts周辺だけが変わっていますが、shortshortsをbottomsに変えた場合はベースが変わることになるので全体が変わってしまいます。つまり、プロンプト指定では一部だけを変更するインペイントのようなことができるというわけです。

少し設定を変えて着せ替えてみましょう。(shortshorts:0.5)としているのは、skirtなどを優先するためです。領域用のshortshortsは弱めても問題ありません。普通は同じものを対象としますが、強い単語だと影響が出すぎてしまうので弱めるのも手です。

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
long skirtなどに対応するためにshortshortsのthresholdを広くしています。bottoms やlower bodyの方が楽だと思います。
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
