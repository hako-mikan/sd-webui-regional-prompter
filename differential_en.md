# Differential Regional Prompter
This script is an auxiliary script for the Regional Prompter and operates as a custom script (similar to XYZ plot) for the Stable Diffusion web UI.
With this script, you can create differential images and generate consistent animations using the region specification of the Regional Prompter's prompt. Although traditional region specification using prompts could maintain some consistency in differential images, unintended differences could appear outside the specified region during the denoising process, making it impossible to achieve perfect differentials. This script allows you to reflect only the differences in the specified region from the initial image. By continuously varying the differential, smooth animations can also be created.

The following images demonstrate how `closed eyes` is applied only to the region calculated from `eyes` using this script. The third image is an animated GIF.
<img src="https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/dsample1.jpg" width="400">  
<img src="https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/dsample2.jpg" width="400">  
<img src="https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/dsample.gif" width="400">  

As shown above, you can generate images where unchanged areas remain consistent. This could be useful for various applications, such as creating images for LoRA copy learning.
Additionally, you can create simple animations using the scheduling feature. This feature operates with the Regional Prompter alone and does not require additional modules.

## How It Works
Internally, the script uses Prompt Editing and the Regional Prompter’s region specification to generate differentials. This approach ensures higher consistency with the original image. For example, if you want to create a differential where the eyes are closed, simply adding `closed eyes` to the prompt may cause significant changes to the entire image. Instead, using `[:closed eyes:4]` applies the `closed eyes` effect starting from step 4, preserving consistency with the original image. The step setting in the configuration screen indicates the starting step for prompt editing.

## How to Use
Select **Differential Regional Prompter** from the script options. If the Regional Prompter is already installed, no additional settings are required.

<img src="https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/gamen.jpg" width="800">  

### Options
- **Reverse**: Enables reverse playback of the generated video. The reason for this will be explained later.

### Additional Output
- Select whether to generate videos (mp4) or animated GIFs. If enabled, they will be saved in the Output folder.

### Step
- Specifies the starting step for Prompt Editing. Typically, values between 4 and 6 work well.

### FPS
- Sets the frame rate when creating a video. The duration for an animated GIF is calculated as `1000/FPS`.

### Schedule
- Enter differential prompts. See usage examples for more details.

### mp4 Output Directory
- Specifies the directory for saving mp4 files. If left blank, the output will be saved in the `output/txt2img-images` folder. If a value is entered, the specified directory will be created under `output`.

### mp4 Output Filename
- Specifies the filename for the mp4 file. If left blank, files will be named `drp.mp4`, `drp_1.mp4`, etc. If you enter `test`, the files will be named `test.mp4`, `test_1.mp4`, etc. Files will not be overwritten.

### Anime GIF Output Directory
- Specifies the directory for saving animated GIFs. If left blank, the output will be saved in the `output/txt2img-images` folder. If a value is entered, the specified directory will be created under `output`.

### Anime GIF Output Filename
- Specifies the filename for the animated GIF file. If left blank, files will be named `drp.gif`, `drp_1.gif`, etc. If you enter `test`, the files will be named `test.gif`, `test_1.gif`, etc. Files will not be overwritten.

## Usage Examples
### Blinking
Here, we assume that you want to create a differential where the eyes close.

First, enter the main prompt in the regular prompt input field. Let’s use:
```
a girl in garden face close up, eyes
```
It is important to include `eyes`, as this is necessary for calculating the differential region.

Next, enter the following in the **Schedule** field and set the **threshold** in the Regional Prompter settings to around 0.6:
```
0
closed eyes;eyes;1.3
```
Generating with these settings will produce the two images shown earlier. If the **Anime GIF** option is enabled, an animated GIF will also be created.

Each setting value follows this format:
```
prompt;prompt for region calculation;weight;step
```
- **prompt**: The prompt for generating the differential.
- **prompt for region**: The prompt used to calculate the region.
- **weight**: The strength of the prompt.
- **step (optional)**: The step at which the differential prompt becomes effective.

For `closed eyes;eyes;1.3;4`, the script effectively applies the prompt `[:(closed eyes:1.3):4]` during execution.

### Smiling
Next, let’s create an animation of a gradual smile.
Without changing the main prompt, enter the following in the **Schedule** field:
```
0*10
smile;face;1.2;20-6(2)
smile;face;1.2*10
```
This means:
- Display the initial image for 10 frames.
- Apply `smile` with a strength of 1.2 to the `face` region, decreasing the step from 20 to 6 in increments of 2.
- Apply `smile` with a strength of 1.2 to the `face` region for 10 frames.

The notation `20-6(2)` means that step values will change from `20, 18, 16, ..., 6` automatically, making the smile effect gradually appear.

<img src="https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/dsample5.gif" width="400">  

The same format applies to **weight** values. For example, `1.0-1.3(0.05)` increases the weight from 1.0 to 1.3 in increments of 0.05, generating 7 frames.

#### Special Commands
- **step=5**: Changes the step value.
- **th=0.45**: Changes the threshold for region specification.
- **ex-on,0.01 / ex-off**: Enables/disables **extra seed** for slight variations in generated images.

Using `extra seed`, you can create subtle changes in background or effects. For example:
```
0
ex-on,0.005
 ;lightning_thunder;1.00-1.05
```
This produces animated lightning effects.

<img src="https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/dsample4.gif" width="400">  

These techniques allow for more advanced animation sequences, such as a character blinking and smiling in sequence.

