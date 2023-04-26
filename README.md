# Regional Prompter
![top](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/top.jpg)
- custom script for [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 
- Different prompts can be specified for different regions

- [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 用のスクリプトです
- 垂直/平行方向に分割された領域ごとに異なるプロンプトを指定できます

## Language control / 言語制御
日本語: [![jp](https://img.shields.io/badge/lang-jp-green.svg)](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/main/README.JP.md)

## Updates
- New feature, "2D-Region"
- New generation method "Latent" added. Generation is slower, but LoRA can be separated to some extent.
- Supports over 75 tokens
- Common prompts can be set
- Setting parameters saved in PNG info

Thanks to the great cooperation of [Symbiomatrix](https://github.com/Symbiomatrix), we can now specify [more flexible areas](#2d-region-assignment-experimental-function).  

# Overview
Latent couple extention performs U-Net calculations on a per-prompt basis, but this extension performs per-prompt calculations inside U-Net. See [here(Japanese)](https://note.com/gcem156/n/nb3d516e376d7) for details. Thanks to furusu for initiating the idea. Additional, Latent mode also supported.

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
This extention is enabled only if "Active" is toggled.

### Prompt
Prompts for different regions are separated by `BREAK` keywords. 
Negative prompts can also be set for each area by separating them with `BREAK`, but if `BREAK` is not entered, the same negative prompt will be set for all areas.

Using `ADDROW` or `ADDCOL` anywhere in the prompt will automatically activate [2D region mode](#2d-region-assignment-experimental-function).

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

### Divide mode
Specifies the direction of division. Horizontal and vertical directions can be specified.
In order to specify both horizontal and vertical regions, see 2D region mode.

### calcutation mode  
#### Attention  
Normally, use this one.  
#### Latent
Slower, but allows separating LoRAs to some extent. The generation time is the number of areas x the generation time of one pic.

Example of Latent mode for [nendoorid](https://civitai.com/models/7269/nendoroid-figures-lora),
[figma](https://civitai.com/models/7984/figma-anime-figures) LoRA separated into left and right sides to create.  
<img src="https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/sample2.jpg" width="400">

~~The web-ui update at the end of March will change the way LoRA is applied, which will significantly increase the generation time. It is not that there is anything wrong with the update, but that it has the effect of reducing the generation time for normal usage, but seems to have the opposite effect on the stage where region-specific adaptation is used. I have tried several countermeasures, but so far no workaround has come to mind.~~

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

### 2D region assignment
You can specify a region in two dimensions. Using a special separator (`ADDCOL/ADDROW`), the area can be divided horizontally and vertically. Starting at the upper left corner, the area is divided horizontally when separated by `ADDCOL` and vertically when separated by `ADDROW`. The ratio of division is specified as a ratio separated by a semicolon. An example is shown below; although it is possible to use `BREAK` alone to describe only the ratio, it is easier to understand if COL/ROW is explicitly specified. Using `ADDBASE `as the first separator will result in the base prompt. If no ratio is specified or if the ratio does not match the number of separators, all regions are automatically treated as equal multiples.
In this mode, the direction selected in `Divide mode` changes which separator is applied first:
- In `Horizontal` mode, the image is first split to rows with `ADDROW` or `;` in Divide ratio, then each row is split to regions with `ADDCOL` or `,` in Divide ratio.
- In `Vertical` mode, the image is first split to columns with `ADDCOL` or `,` in Divide ratio, then each column is split to regions with `ADDROW` or `;` in Divide ratio.

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
Divide mode : Horizontal
Divide Ratio : 1,2,1,1;2,4,6
Base Ratio : 
```

![2d](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/2d.jpg)

### Visualise and make template
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

### Difference between base and common
```
a girl ADDCOMM (or ADDBASE)
red hair BREAK
green dress
```
If there is a prompt that says `a girl` in the common clause, region 1 is generated with the prompt `a girl , red hair`. In the base clause, if the base ratio is 0.2, it is generated with the prompt `a girl` * 0.2 + `red hair` * 0.8. Basically, common clause combines prompts, and base clause combines weights (like img2img denoising strength). You may want to try the base if the common prompt is too strong, or fine tune the (emphasis).

### <a id="divprompt">region specification by prompt (experimental)</a>
The region is specified by the prompt. The picture below was created with the following prompt, but the prompt `apple printed` should only affect the shirt, but the actual apples are shown and so on. 
```
lady smiling and sitting, twintails green hair, white skirt, apple printed shirt
```
![prompt](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/psample1.png)
If you enhance the effect of `apple printed` to `:1.4`, you get, 
![prompt](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/psample4.png)
The prompt region specification allows you to calculate the region for the "shirt" and adapt the "printed apples".

![prompt](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/psample3.png)
```
lady smiling and sitting, twintails green hair, white skirt, shirt BREAK
(apple printed:1.4),shirt
```
![prompt](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/psample2.png)


### Acknowledgments
I thank [furusu](https://note.com/gcem156) for suggesting the Attention couple, [opparco](https://github.com/opparco) for suggesting the Latent couple, and [Symbiomatrix](https://github.com/Symbiomatrix) for helping to create the 2D generation code.
