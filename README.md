# Regional Prompter
![top](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/top.jpg)
- custom script for [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 
- Different prompts can be specified for different regions

- [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 用のスクリプトです
- 垂直/平行方向に分割された領域ごとに異なるプロンプトを指定できます

## update/更新情報
- new feature "2D-Region"
- new Generation method "Latent" added. Generation is slower, but LoRA can be separated to some extent.
- 新機能2D領域を追加しました
- 新しい計算方式「Latent」を追加しました。生成が遅くなりますがLoRAをある程度分離できます

Thanks to the great cooperation of [Symbiomatrix](https://github.com/Symbiomatrix), we can now specify [more flexible areas](#2d-region-assignment-experimental-function).  
[Symbiomatrix](https://github.com/Symbiomatrix)氏の協力によりより[柔軟な領域指定](#2次元領域指定実験的機能)が可能になりました。


日本語解説は[後半](概要)です。

# Overview
Latent couple extention performs U-Net calculations on a per-prompt basis, but this extension performs per-prompt calculations inside U-Net. See [here](https://note.com/gcem156/n/nb3d516e376d7) for details.Thanks to furusu for initiating the idea.

## Usage
This section explains how to use the following image, explaining how to create the following image.  
![sample](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/sample.jpg)  
Here is the prompt.
```
green hair twintail BREAK
red blouse BREAK
blue skirt
```
setting
```
Active : On
Use base prompt : Off
Divide mode : Vertical
Divide Ratio : 1,1,1
Base Ratio : 
````
This setting divides the image vertically into three parts and applies the prompts "green hair twintail" ,"red blouse" ,"blue skirt" from the top to the bottom.

### Active  
If checked, this extention is enabled.

### Prompt
Prompts for different areas are separated by "BREAK". Enter prompts from the left for horizontal prompts and from the top for vertical prompts.
Negative prompts can also be set for each area by separating them with BREAK, but if BREAK is not entered, the same negative prompt will be set for all areas.

### Use base prompt
Check this if you want to use the base prompt, which is the same prompt for all areas. Use this option if you want the prompt to be consistent across all areas.
When using base prompt, the first prompt separated by BREAK is treated as the base prompt.
Therefore, when this option is enabled, one more BRAKE-separated prompt is required than Divide ratios.  
Automatically turned on when ADDBASE is entered.

### Base ratio
Sets the ratio of the base prompt; if 0.2 is setted, the base ratio is 0.2. It can also be specified for each region, and can be entered as 0.2, 0.3, 0.5, etc. If a single value is entered, the same value is applied to all areas.

### Divide ratio
If you enter 1,1,1, the area will be divided into three parts (33,3%, 33,3%, 33,3%); if you enter 3,1,1, the area will be divided into 60%, 20%, and 20%. Decimal points can also be entered. 0.1,0.1,0.1 is equivalent to 1,1,1.

### Divide mode
Specifies the direction of division. Horizontal and vertical directions can be specified.

### calcutation mode  
#### Attention  
Normally, use this one.  
#### Latent
If you want to separate LoRA, use this one. Although the generation time is longer, LoRA can be separated to some extent. The generation time is the number of areas x the generation time of one pic.

Example of Latent mode for [nendoorid](https://civitai.com/models/7269/nendoroid-figures-lora),
[figma](https://civitai.com/models/7984/figma-anime-figures) LoRA separated into left and right sides to create.  
<img src="https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/sample2.jpg" width="400">

### Use common prompt
If this option enabled, first part of the prompt is added to all part.  
Automatically turned on when ADDCOMM is entered.
```
best quality, 20yo lady in garden BREAK
green hair twintail BREAK
red blouse BREAK
blue skirt
```
If enabled, this prompt is treated as following,
```
best quality, 20yo lady in garden, green hair twintail BREAK
best quality, 20yo lady in garden, red blouse BREAK
best quality, 20yo lady in garden, blue skirt
```
So you need to set 4 prompts for 3 regions. If Use base prompt is also enabled 5 prompts are needed. The order is as follows, common,base, prompt1,prompt2,...

### 2D region assignment (experimental function)
You can specify a region in two dimensions. Using a special separator (ADDCOL/ADDROW), the area can be divided horizontally and vertically. Starting at the upper left corner, the area is divided horizontally when separated by ADDCOL and vertically when separated by ADDROW. The ratio of division is specified as a ratio separated by a semicolon. An example is shown below; although it is possible to use BREAK alone to describe only the ratio, it is easier to understand if COL/ROW is explicitly specified. Using ADDBASE as the first separator will result in the base prompt. If no ratio is specified or if the ratio does not match the number of separators, all regions are automatically treated as equal multiples.

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

### Divide mode
分割方向を指定します。水平、垂直方向が指定できます。

### Use common prompt
このオプションを有効化すると最初のプロンプトをすべてのプロンプトに加算します。
ADDCOMMが入力された場合自動的にオンになります。
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
領域を2次元的に指定できます。特別なセパレイター(ADDCOL/ADDROW)を用いることで領域を縦横に分割することができます。左上を始点として、ADDCOLで区切ると横方向、ADDROWで区切ると縦方向に分割されます。分割の比率はセミコロンで区切られた比率で指定します。以下に例を示します。BREAKのみで記述し、比率のみで記述することも可能ですが、明示的にCOL/ROWを指定した方がわかりやすいです。最初のセパレーターとしてADDBASEを使用すると、ベースプロンプトになります。比率を指定しない場合や比率がセパレーターの数と一致しないときは自動的にすべて等倍として処理されます。ADDCOMMを最初のセパレーターとして入力した場合共通プロンプトになります。

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


## update/更新情報
- 75トークン以上を入力できるようになりました
- 共通プロンプトを設定できるようになりました
- 設定がPNG infoに保存されるようになりました
- support over 75 tokens
- common prompts can be set
- setting parameters saved in PNG info
