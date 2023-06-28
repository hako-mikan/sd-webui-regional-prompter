Tutorial on specifying areas with prompts

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
What's important here is that `shirt` is both placed before the `BREAK` and at the end after the 'BREAK', and is separated by a comma. In prompt mode, the word that is separated by a comma at the end is the target for region calculation.
```
Divide mode : Prompt
Calcmode : Attention
threshold : 0.7
```

With these settings, the generation will look like this.  
![2](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/ptutorial2.png)  
実際に生成されたマスクはこんな感じです。    
![3](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/ptutorial3.png)    
This is a property of the Attention mode, where it actually shrinks to about 12×8, making it more ambiguous, which might be better.

Now, there seems to be a strange string attached to the "shortshort", so let's change this next. We rewrite the prompt as follows.
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
It turns out like this.
![4](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/ptutorial4.png)    
The `shortshorts` became a `skirt`. I wrote it as `shortshorts` because that's what it was initially, but using a word like `bottoms` can make region selection easier. The reason why I kept it as `shortshorts` this time is because I didn't want to change the base prompt. If you change `shortshorts` to `bottoms`, it changes the initial image itself.

```
girl in street shirt,bottoms daytime BREAK
(forest printed:1.3) T-shirt ,shirt BREAK
(red skirt:1.7) ,bottoms
```
で作った画像を三番目に置きました。  
![5](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/ptutorial5.png)   
Keeping it as `shortshorts` only changes the area around the `shortshorts`, but if you change `shortshorts` to `bottoms`, it changes the base, which changes the whole image. In other words, with prompt-based specifications, you can do something like inpainting where only a part is changed.

Let's change the settings a bit and try dressing up. The reason why I set it as` (shortshorts:0.5)` is to prioritize the `skirt`. Weakening the `shortshorts` for the region won't be a problem. Normally, you target the same item, but if the word is too strong, it will have too much impact, so weakening it is an option.

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
I'm broadening the `threshold` for `shortshorts` to accommodate things like long skirt. I think bottoms or lower body would be easier.
![6](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/ptutorial6.png)  
We were able to change the clothes, but there's some erosion in the pink. This is because the intensity is set uniformly. If we refine the settings, the erosion should disappear.

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
Let's change the `Japan` part.
![7](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/ptutorial7.png)  
The word `landscape` is a bit tricky. It's quite strong and often cancels out the effect of other words. However, by calculating the region using street instead of `landscape`, we can prevent it from becoming overly dominant. By calculating regions with different words like this, you might be able to expand the range of your expressions.

When the Regional Prompter is disabled, it looks like this.
This is pretty good as it is.
![8](https://github.com/hako-mikan/sd-webui-regional-prompter/blob/imgs/ptutorial8.png)    
