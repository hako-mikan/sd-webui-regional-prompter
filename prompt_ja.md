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
