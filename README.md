# Regional Prompter
![top](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/top2.jpg)
- custom script for [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 
- Different prompts can be specified for different regions

- [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 用のスクリプトです
- 垂直/平行方向に分割された領域ごとに異なるプロンプトを指定できます

[<img src="https://img.shields.io/badge/言語-日本語-green.svg?style=plastic" height="25" />](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/main/README.JP.md)
[<img src="https://img.shields.io/badge/Support-%E2%99%A5-magenta.svg?logo=github&style=plastic" height="25" />](https://github.com/sponsors/hako-mikan)

## Updates 2025.01.28.2000 (JST)
- support reForge
- reForgeに対応

| Mode \ Client             |  A1111      | Forge         | reForge      
|------------------|--------------------|---------------|---------------|
| Attention Mode   | ○                 | ○            | ○            |
| Latent Mode      | ○                 | △             | △             |
| LoRA (Latent)           | ○                 | ○             | ×             |

○ : Supported  
△: Supported, but may not function properly with large batch sizes. Depends on VRAM.  
×: Not supported

## for LoHa, LoCon users 
**About LoRA/LoCon/LoHa**  
There are certain constraints due to the specifications of the Web-UI regarding the following:
These constraints arise because the Web-UI cannot perform specific optimizations when applying LoRA and does not support mid-strength changes to LoRA.
- **LoRA**: Can be applied without a decrease in speed.
- **LoCon/LoHa**: It can be used when the "Use LoHa or other" option is enabled, but this results in a slower generation speed. This constraint is based on the Web-UI's specifications.

**LoRA/LoCon/LoHaについて**
LoRAの種類別の使用条件です。
- **LoRA**: 速度低下なく適用可能です。
- **LoCon/LoHa**: "Use LoHa or other" オプションを有効にすると使用できますが、生成速度が遅くなります。この制約はWeb-UIの仕様に基づいています。Forgeの場合制限はありません。


# Overview
Latent couple extension performs U-Net calculations on a per-prompt basis, but this extension performs per-prompt calculations inside U-Net. See [here(Japanese)](https://note.com/gcem156/n/nb3d516e376d7) for details. Thanks to furusu for initiating the idea. Additional, Latent mode also supported.

## index
- [2D regions](#2D)
- [Latent mode(LoRA)](#latent) 
- [regions by inpaint](#inpaint) 
- [regions by prompt](#divprompt) 

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
This setting divides the image vertically into three parts and applies the prompts "green hair twintail" ,"red blouse" ,"blue skirt", from top to bottom in order.

### Active  
This extension is enabled only if "Active" is toggled.

### Prompt
Prompts for different regions are separated by `BREAK` keywords. 
Negative prompts can also be set for each area by separating them with `BREAK`, but if `BREAK` is not entered, the same negative prompt will be set for all areas.

Using `ADDROW` or `ADDCOL` anywhere in the prompt will automatically activate [2D region mode](#2D).

### Use base prompt
Check this if you want to use the base prompt, which is the same prompt for all areas. Use this option if you want the prompt to be consistent across all areas.
When using base prompt, the first prompt separated by `BREAK` is treated as the base prompt.
Therefore, when this option is enabled, one extra `BREAK`-separated prompt is required compared to Divide ratios.

Automatically turned on when `ADDBASE` is entered.

### Divide ratio
If you enter 1,1,1, the image will be divided into three equal regions (33,3%, 33,3%, 33,3%); if you enter 3,1,1, the image will be divided into 60%, 20%, and 20%. Fractions can also be entered: 0.1,0.1,0.1 is equivalent to 1,1,1. For greatest accuracy, enter pixel values corresponding to height / width (vertical / horizontal mode respectively), eg 300,100,112 -> 512.

Using a `;` separator will automatically activate 2D region mode.

### Base ratio
Sets the ratio of the base prompt; if base ratio is set to 0.2, then resulting images will consist of `20%*BASE_PROMPT + 80%*REGION_PROMPT`. It can also be specified for each region, in the same way as "Divide ratio" - 0.2, 0.3, 0.5, etc. If a single value is entered, the same value will be applied to all areas.

### split mode
Specifies the direction of division. Horizontal and vertical directions can be specified.
In order to specify both horizontal and vertical regions, see 2D region mode.

## calcutation mode  
Internally, system use BREAK in Attention mode and AND in Latent mode. AND/BREAK is automatically converted depending on the mode being used, but there is no problem whether you input BREAK or AND in the prompt.
### Attention  
Normally, use this one.  
### Latent
Slower, but allows separating LoRAs to some extent. The generation time is the number of areas x the generation time of one pic. See [known issues](#knownissues).

Example of Latent mode for [nendoorid](https://civitai.com/models/7269/nendoroid-figures-lora),
[figma](https://civitai.com/models/7984/figma-anime-figures) LoRA separated into left and right sides to create.  
<img src="https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/sample2.jpg" width="400">

### Use common prompt
If this option enabled, first part of the prompt is added to all region parts.

Automatically turned on when `ADDCOMM` is entered.
```
best quality, 20yo lady in garden BREAK
green hair twintail BREAK
red blouse BREAK
blue skirt
```
If common is enabled, this prompt is converted to the following:
```
best quality, 20yo lady in garden, green hair twintail BREAK
best quality, 20yo lady in garden, red blouse BREAK
best quality, 20yo lady in garden, blue skirt
```
So you must set 4 prompts for 3 regions. If `Use base prompt` is also enabled 5 prompts are needed. The order is as follows: common, base, prompt1,prompt2,...

## <a id="2D">2D region assignment</a>
You can specify a region in two dimensions. Using a special separator (`ADDCOL/ADDROW`), the area can be divided horizontally and vertically. Starting at the upper left corner, the area is splited by columns when separated by `ADDCOL` and rows when separated by `ADDROW`. The ratio of division is specified as a ratio separated by a semicolon. An example is shown below; although it is possible to use `BREAK` alone to describe only the ratio, it is easier to understand if COL/ROW is explicitly specified. Using `ADDBASE `as the first separator will result in the base prompt. If no ratio is specified or if the ratio does not match the number of separators, all regions are automatically treated as equal multiples.
In this mode, the direction selected in `Divide mode` changes which separator is applied first:
- In `Coloms` mode, the image is first split to rows with `ADDROW` or `;` in Divide ratio, then each row is split to regions with `ADDCOL` or `,` in Divide ratio.
- In `Rows` mode, the image is first split to columns with `ADDCOL` or `,` in Divide ratio, then each column is split to regions with `ADDROW` or `;` in Divide ratio.
- When the flip option is enabled, it swaps the , and ;. This allows you to obtain an area that is rotated 90 degrees while keeping the same ratios used in Columns/Rows.
  
In any case, the conversion of prompt clauses to rows and columns is from top to bottom, left to right.

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
Main splitting : Columns
Divide Ratio : 1,2,1,1;2,4,6
Base Ratio : 
```

![2d](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/2d.jpg)									

## <a id="visualize">visualize and make template</a>
Areas can be visualized and templates for prompts can be created.

![tutorial](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/tutorial.jpg)

Enter the area ratio and press the button to make the area appear. Next, copy and paste the prompt template into the prompt input field.

```
fantasy ADDCOMM
sky ADDROW
castle ADDROW
street stalls ADDCOL
2girls eating and walking on street ADDCOL
street stalls
```
Result is following,
![tutorial](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/sample3.jpg)


This is an example of an area using 1,1;2,3,2;3,2,3. In Columns, it would look like this:  
![flip](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/msapmle1.png)  
In Rows, it would appear as follows:  
![flip](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/msapmle2.png)  
When the flip option is enabled in Rows, it would appear as follows:  
![flip](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/msapmle3.png)	  			

"Overlay ratio" is primarily used in inpaint. By loading an image intended for inpaint on the visualize screen, you can view the regions. For further details, please refer to the following [issue](https://github.com/hako-mikan/sd-webui-regional-prompter/issues/234).


## <a id="inpaint">Mask regions aka inpaint+ (experimental function)</a>
It is now possible to specify regions using either multiple hand drawn masks or an uploaded image containing said masks (more on that later).
- First, make sure you switch to `mask divide mode` next to `Colums` / `Rows`. Otherwise the mask will be ignored and regions will be split by ratios as usual.
- Set `canvas width and height` according to desired image's, then press `create mask area`. If a different ratio or size is specified, the masks may be applied inaccurately (like in inpaint "just resize").
- Draw an outline / area of the region desired on the canvas, then press `draw region`. This will fill out the area, and colour it according to the `region` number you picked. **Note that the drawing is in black only, filling and colouring are performed automatically.** The region mask will be displayed below, to the right.
- Pressing `draw region` will automatically advance to the next region. It will also keep a list of which regions were used for building the masks later. Up to 360 regions can be used currently, but note that a few of them on the higher end are identical.
- It's possible to add to existing regions by reselecting the same number and drawing as usual.
- The special region number -1 will clear out (colour white) any drawn areas, and display which parts still contain regions in mask.
- Once the region masks are ready, write your prompt as usual: Divide ratios are ignored. Base ratios still apply to each region. All flags are supported, and all BREAK / ADDX keywords (ROW/COL will just be converted to BREAK). Attention and latent mode supported (loras maybe).
- `Base` has unique rules in mask mode: When base is off, any non coloured regions are added to the first mask (therefore should be filled with the first prompt). When base is on, any non coloured regions will receive the base prompt in full, whilst coloured regions will receive the usual base weight. This makes base a particularly useful tool for specifying scene / background, with base weight = 0.
- Masks are saved to and loaded from presets whose divide mode is `mask`. The mask is saved in the extension directory, under the folder `regional_masks`, as {preset}.png file.
- Masks can be uploaded from any image by using the empty component labelled `upload mask here`. It will automatically filter and tag the colours approximating matching those used for regions, and ignore the rest. The region / nonregion sections will be displayed under mask. **Do not upload directly to sketch area, and read the [known issues](#knownissues) section.**
- If you wish to draw masks in an image editor, this is how the colours correspond to regions: The colours are all variants of `HSV(degree,50%,50%)`, where degree (0:360) is calculated as the maximally distant value from all previous colours (so colours are easily distinguishable). The first few values are essentially: 0, 180, 90, 270, 45, 135, 225, 315, 22.5 and so on. The choice of colours decides to which region they correspond.
- Protip: You may upload an openpose / depthmap / any other image, then trace the regions accordingly. Masking will ignore colours which don't belong to the expected colour standard.

![RegionalMaskGuide2](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/RegionalMaskGuide2.jpg)
![RegionalMaskGuide2B](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/RegionalMaskGuide2B.jpg)


Here is sample and code  
using mask and prompt`landscape BREAK moon BREAK girl`.
Using XYZ plot prompt S/R, changed `moon BREAK girl` to others.  
![RegionalMaskSample](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/isample1.png)



## <a id="divprompt">region specification by prompt (experimental)</a>
The region is specified by the prompt. The picture below was created with the following prompt, but the prompt `apple printed` should only affect the shirt, but the actual apples are shown and so on. 
```
lady smiling and sitting, twintails green hair, white skirt, apple printed shirt
```
![prompt](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/psample1.png)
If you enhance the effect of `apple printed` to `:1.4`, you get, 

![prompt](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/psample4.png)
The prompt region specification allows you to calculate the region for the "shirt" and adapt the "printed apples".

![prompt](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/psample6.png)
```
lady smiling and sitting, twintails green hair, white skirt, shirt BREAK
(apple printed:1.4),shirt
```
![prompt](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/psample2.png)

### How to use
### syntax
```
baseprompt target1 target2 BREAK
effect1, target1 BREAK
effect2 ,target2
```

First, write the base prompt. In the base prompt, write the words (target1, target2) for which you want to create a mask. Next, separate them with BREAK. Next, write the prompt corresponding to target1. Then enter a comma and write target1. The order of the targets in the base prompt and the order of the BREAK-separated targets can be back to back.

```
target2 baseprompt target1  BREAK
effect1, target1 BREAK
effect2 ,target2
```
is also effective.

### threshold
The threshold used to determine the mask created by the prompt. This can be set as many times as there are masks, as the range varies widely depending on the target prompt. If multiple areas are used, enter them separated by commas. For example, hair tends to be ambiguous and requires a small value, while face tends to be large and requires a small value. These should be ordered by BREAK.

```
a lady ,hair, face  BREAK
red, hair BREAK
tanned ,face
```
`threshold : 0.4,0.6`
If only one input is given for multiple regions, they are all assumed to be the same value.

### Prompt and Prompt-EX
The difference is that in Prompt, duplicate areas are added, whereas in Prompt-EX, duplicate areas are overwritten sequentially. Since they are processed in order, setting a TARGET with a large area first makes it easier for the effect of small areas to remain unmuffled.

### Accuracy
In the case of a 512 x 512 image, Attention mode reduces the size of the region to about 8 x 8 pixels deep in the U-Net, so that small areas get mixed up; Latent mode calculates 64*64, so that the region is exact.  
```
girl hair twintail frills,ribbons, dress, face BREAK
girl, ,face
```
Prompt-EX/Attention
![prompt](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/psample5.png)
Prompt-EX/Latent
![prompt](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/psample3.png)


### Mask
When an image is generated, the generated mask is displayed. It is generated at the same size as the image, but is actually used at a much smaller size.

## Difference between base and common
```
a girl ADDCOMM (or ADDBASE)
red hair BREAK
green dress
```
If there is a prompt that says `a girl` in the common clause, region 1 is generated with the prompt `a girl , red hair`. In the base clause, if the base ratio is 0.2, it is generated with the prompt `a girl` * 0.2 + `red hair` * 0.8. Basically, common clause combines prompts, and base clause combines weights (like img2img denoising strength). You may want to try the base if the common prompt is too strong, or fine tune the (emphasis).
The immediate strength that corresponds to the target should be stronger than normal. Even 1.6 doesn't break anything.

## <a id="knownissues">Known issues</a>
- Due to an [issue with gradio](https://github.com/gradio-app/gradio/issues/4088), uploading a mask or loading a mask preset more than twice in a row will fail. There are two workarounds for this:
1) Before EVERY upload / load, press `create mask area`.
2) Modify the code in gradio.components.Image.preprocess; add the following at the beginning of the function (temporarily):
```
        if self.tool == "sketch" and self.source in ["upload", "webcam"]:
            if x is not None and isinstance(x, str):
                x = {"image":x, "mask": x[:]}
```
The extension cannot perform this override automatically, because gradio doesn't currently support [custom components](https://github.com/gradio-app/gradio/issues/1432). Attempting to override the component / method in the extension causes the application to not load at all.

3) Wait until a fix is published.

- Lora corruption in latent mode. Some attempts have been made to improve the output, but no solution as of yet. Suggestions below.
1) Reduce cfg, reduce lora weight, increase sampling steps.
2) Use the `negative textencoder` + `negative U-net` parameters: these are weights between 0 and 1, comma separated like base. One is applied to each lora in order of appearance in the prompt. A value of 0 (the default) will negate the effect of the lora on other regions, but may cause it to be corrupted. A value of 1 should be closer to the natural effect, but may corrupt other regions (greenout, blackout, SBAHJified etc), even if they don't contain any loras. In both cases, a higher lora weight amplifies the effect. The effect seems to vary per lora, possibly per combination.
3) It has been suggested that [lora block weight](https://github.com/hako-mikan/sd-webui-lora-block-weight) can help.
4) If all else fails, inpaint.

Here are samples of a simple prompt, two loras with negative te/unet values per lora of: (0,0) default, (1,0), (0,1), (1,1).
![MeguminMigurdiaCmp](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/MeguminMigurdiaCmp.jpg)

If you come across any useful insights on the phenomenon, do share.

## Settings
###Hide subprompt masks in prompt mode
In prompt mode, don't show the generated mask.

###Disable ImageEditor
Disable the mask editor (this is to fix an issue where CPU usage hits 100% in some browsers).

###Use old active check box
Use the old Active button (when using TabExtension).

## How to Use via API
The following format is used when utilizing this extension via the API.

```
  "prompt": "green hair twintail BREAK red blouse BREAK blue skirt",
	"alwayson_scripts": {
		"Regional Prompter": {
			"args": [True,False,"Matrix","Vertical","Mask","Prompt","1,1,1","",False,False,False,"Attention",False,"0","0","0",""]
}}
```
Please refer to the table below for each setting in `args`. No. corresponds to the order. When the type is text, please enclose it with `""`. Modes 3-6 ignore submodes that do not correspond to the mode selected in mode 3. For the mask in 17., please specify the address of the image data. Absolute paths or relative paths from the web-ui root can be used. Please create the mask using the color specified in the mask item.

|  No.  |  setting  |choice| type  | default |
| ---- | ---- |---- |----| ----|
|  1  |  Active  |True, False|Bool|False| 
|  2  | debug   |True, False|Bool|False| 
|  3  | Mode  |Matrix, Mask, Prompt|Text| Matrix|
|  4  | Mode (Matrix)|Horizontal, Vertical, Colums, Rows|Text|Columns
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
|  18 | LoRA stop step   |  | Text | 0
|  19 | LoRA Hires stop step   |  | Text | 0
|  20 | flip   |True, False| Bool | False

### Example Settings
#### Matrix
```
  "prompt": "green hair twintail BREAK red blouse BREAK blue skirt",
	"alwayson_scripts": {
		"Regional Prompter": {
			"args": [True,False,"Matrix","Vertical","Mask","Prompt","1,1,1","",False,False,False,"Attention",False,"0","0","0",""]
}}
```
Result
![sample](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/asample1.png)  

#### Mask
```
   "prompt": "masterpiece,best quality 8k photo of BREAK (red:1.2) forest BREAK yellow chair BREAK blue dress girl",
	"alwayson_scripts": {
		"Regional Prompter": {
			"args":	[True,False,"Mask","Vertical","Mask","Prompt","1,1,1","",False,True,False,"Attention",False,"0","0","0","mask.png"]
```
Mask used
![sample](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/mask.png)  
Result
![sample](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/asample2.png)  

#### Prompt
```
 "prompt": "masterpiece,best quality 8k photo of BREAK a girl hair blouse skirt with bag BREAK (red:1.8) ,hair BREAK (green:1.5),blouse BREAK,(blue:1.7), skirt BREAK (yellow:1.7), bag",
	"alwayson_scripts": {
		"Regional Prompter": {
			"args":	[True,False,"Prompt","Vertical","Mask","Prompt-EX","1,1,1","",False,True,False,"Attention",False,"0","0","0.5,0.6,0.5",""]
}}
```
![sample](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/asample3.png)  


## Troubleshooting
### Input type (struct c10::Half) and bias type (float) should be the same
If you encounter above error while using LoRA with the MIDVRAM option or similar settings, please enable the **"Use LoHa or other"** option.

### IndexError: list index out of range
#### context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
#### fil = self.filters[a + b*areas]
This happens when the number of BREAKs doesn’t match the number of areas. If "use base" is checked, try unchecking it and see if that helps.  

## Acknowledgments
I thank [furusu](https://note.com/gcem156) for suggesting the Attention couple, [opparco](https://github.com/opparco) for suggesting the Latent couple, and [Symbiomatrix](https://github.com/Symbiomatrix) for helping to create the 2D generation code.



## Updates 2025.01.27.0100 (JST)  
- Fixed several bugs (related to Forge)  
- Added support for Latent Mode and region-specific LoRA in Forge (only basic LoRA has been tested so far).  
- いくつかのバグフィックス(Forge関連)
- ForgeにおいてLatentモード対応、領域別LoRA対応（基本的なLoRAしか試してません。）

### Updates
- モード名が変更になりました。`Horizontal` -> `columns`, `Vertical` -> `Rows`
(日本語で横に分割を英訳したSplit Horizontalは英語圏では逆の意味になるようです。水平線「で」分割するという意味になるそう)
- `,`,`;`を入れ替えるオプションを追加

- Split Mode name changed, `Horizontal` -> `columns`, `Vertical` -> `Rows`
- flip `,`,`;` option added

- add LoRA stop step
LoRAを適用するのをやめるstepを指定できます。10 step程度で停止することで浸食、ノイズ等の防止、生成速度の向上を期待できます。
You can specify the step at which to stop applying LoRA. By stopping around 10 steps, you can expect to prevent erosion and noise, and to improve generation speed.
(0に設定すると無効になります。0 is disable)

- support SDXL
- support web-ui 1.5

- add [guide for API users](#how-to-use-via-api)

- prompt mode improved
- プロンプトモードの動作が改善しました  
(The process has been adjusted to generate masks in three steps, and to recommence generation from the first stage./3ステップでマスクを生成し、そこから生成を1stepからやり直すよう修正しました)

- New feature : [regions by inpaint](#inpaint) (thanks [Symbiomatrix](https://github.com/Symbiomatrix))
- New feature : [regions by prompt](#divprompt) ([Tutorial](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/main/prompt_en.md)) 
- 新機能 : [インペイントによる領域指定](#inpaint) (thanks [Symbiomatrix](https://github.com/Symbiomatrix))
- 新機能 : [プロンプトによる領域指定](#divprompt) ([チュートリアル](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/main/prompt_ja.md))

## Updates
- New feature, "2D-Region"
- New generation method "Latent" added. Generation is slower, but LoRA can be separated to some extent.
- Supports over 75 tokens
- Common prompts can be set
- Setting parameters saved in PNG info
