## Tutorial on specifying regions with prompts

There are limitations to methods of specifying areas in advance. This is because specifying areas can be a hindrance when designating complex shapes or dynamic compositions. In the region specified by the prompt, the area is determined after the image generation has begun. This allows us to accommodate compositions and complex areas.

Let's take a look at an example.
The following image was created by the next prompt. It's a grand color transition.
```
sfw (8k realistic masterpiece:1.3) a Asian girl ,dark green dress,pink belt,yellow bag,
blond hair, in rainy street, holding red umbrella
```
![1](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/ptutorial9.png)  
Now, if we try to manage this with the usual area designation, we will have trouble specifying the area for the umbrella and the bag. They appear in various places, and some areas are initially small.

With prompt-based area specification, we calculate the area corresponding to each word.

Let's first turn the umbrella red. We change the prompt as follows: we change red umbrella to umbrella and after BREAK, we add (red:1.7), umbrella. This is because the system calculates the area of the word written after the last comma in the prompt following BREAK. In the case of (red:1.7), umbrella, the area of umbrella is calculated, and (red:1.7) is applied to that area.

Intensity adjustment is very important. Normally, if you input 1.7, it tends to fall apart, but with prompt-based area specification, it doesn't work unless you put in about this value. It's especially better to increase the intensity if you're trying to specify a color that doesn't seem to have learned much.
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
Prompt-EX mode is an effective mode for specifying multiple areas and has the effect of overwriting areas with the ones that come later. Therefore, it is effective to specify the areas in a larger order.

Then the umbrella became properly red. The second image is the calculated area. It's properly shaped like an umbrella, and the head part is out of the area. This area varies depending on the prompt, so it's necessary to adjust it with the Threshold. If the Threshold is small, the area will be wider.
![1](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/ptutorial11.png)

Here, negative prompts are also set up.
```
nsfw, (worst quality:1.6), (low quality:1.6), (normal quality:1.6), monochrome 
[(black:1.5)::3] BREAK BREAK (dark,transparent, black, blue:2)  
```
Umbrellas and bags tend to be predominantly black in the data the model was trained on, so caution is required when specifying colors for them. In this case, we've added a prompt to prevent the umbrella area from becoming black. The reason there are two BREAKs is because nega is enabled. [(black:1.5)::3] prevents the image from becoming black before the region specification by the prompt begins. In the region specification by the prompt, the calculation of the region is not valid until the third step.

Now, with similar region specification, the prompt becomes as follows, and we were able to obtain a result where the colors were properly separated.

```
sfw (8k realistic masterpiece:1.3) a girl, (dress:1.2), belt, bag, hair, in rainy street, holding umbrella BREAK
(red:1.7), umbrella  BREAK
(dark green:1.7) ,dress BREAK
(blond:1.7), hair BREAK
(pink:1.7), belt BREAK
(yellow:1.7), bag
```

![1](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/ptutorial10.png)



The following image was created with the prompt below. Although it would be ideal if 'forest' was only applied to the T-shirt, the background has also become a forest.
```
girl in street (forest printed:1.3) T-shirt, shortshorts daytime
```
![1](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/ptutorial1.png)
To apply the 'forest' print only to the T-shirt, we configure the prompt and the Regional Prompter as follows.
```
lady in street shirt,shortshorts daytime BREAK
(forest printed:1.3) T-shirt ,shirt
```
What's important here is that 'shirt' and `,` is both placed before the `BREAK` and at the end after the 'BREAK', and is separated by a comma. In prompt mode, the word that is separated by a comma at the end is the target for region calculation.
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
