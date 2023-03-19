# Regional Prompter
![top](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/top.jpg)
- custom script for [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 
- Different prompts can be specified for different regions

- [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 用のスクリプトです
- 垂直/平衡方向に分割された領域ごとに異なるプロンプトを指定できます

日本語解説は[後半](概要)です。

# Overview
Latent couple extention performs U-Net calculations on a per-prompt basis, but this extension performs per-prompt calculations inside U-Net. See [here](https://note.com/gcem156/n/nb3d516e376d7) for details.

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

### Base ratio
Sets the ratio of the base prompt; if 0.2 is setted, the base ratio is 0.2. It can also be specified for each region, and can be entered as 0.2, 0.3, 0.5, etc. If a single value is entered, the same value is applied to all areas.

### Divide ratio
If you enter 1,1,1, the area will be divided into three parts (33,3%, 33,3%, 33,3%); if you enter 3,1,1, the area will be divided into 60%, 20%, and 20%. Decimal points can also be entered. 0.1,0.1,0.1 is equivalent to 1,1,1.

### Divide mode
Specifies the direction of division. Horizontal and vertical directions can be specified.

# 概要
Latent couple extentionではプロンプトごとにU-Netの計算を行っていますが、このエクステンションではU-Netの内部でプロンプトごとの計算を行います。詳しくは[こちら](https://note.com/gcem156/n/nb3d516e376d7)をご参照ください。

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
ベースプロンプトを使用する場合、BREAK区切られた最初のプロンプトがベースとして扱われます。有効時にはBREAKで区切られたプロンプトがDivide Ratioより一つ多く必要になります。

### Base ratio
ベースプロンプトの比率を設定します。0.2と入力された場合、ベースの割合が0.2になります。領域ごとにも指定可能で、0.2,0.3,0.5などと入力できます。単一の値を入力した場合はすべての領域に同じ値が適応されます。

### Divide ratio
領域の広さを指定します。1,1,1と入力した場合、三分割されます(33,3%,33,3%,33,3%)。3,1,1と入力した場合60%,20%,20%となります。小数点でも入力可能です。0.1,0.1,0.1は1,1,1と同じ結果になります。

### Divide mode
分割方向を指定します。水平、垂直方向が指定できます。
