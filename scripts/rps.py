from unittest import result
import modules.scripts as scripts
import gradio as gr
from pprint import pprint
import os
import math
from PIL import Image, ImageFont, ImageDraw, ImageColor, PngImagePlugin
from PIL import Image
import imageio
import random
import numpy as np
import time
import glob
import time
import re

from modules.processing import process_images
from modules.shared import cmd_opts, total_tqdm, state

class Script(scripts.Script):

    def __init__(self):
        self.count = 0
        self.latent = None
        self.latent_hr= None

    def title(self):
        return "Differential Regional Prompter"

    def ui(self, is_img2img):
        with gr.Row():
            pass
            # urlguide = gr.HTML(value = fhurl(GUIDEURL, "Usage guide"))
        with gr.Row():
            # mode = gr.Radio(label="Divide mode", choices=["Horizontal", "Vertical","Mask","Prompt","Prompt-Ex"], value="Horizontal",  type="value", interactive=True)
            #outmode = gr.Radio(label="Output mode", choices=["ALL", "Only 2nd"], value="ALL",  type="value", interactive=True)
            #changes = gr.Textbox(label="original, replace, replace ;original, replace, replace...")
            pass
        with gr.Row(visible=True):
            # ratios = gr.Textbox(label="Divide Ratio",lines=1,value="1,1",interactive=True,elem_id="RP_divide_ratio",visible=True)
            options = gr.CheckboxGroup(choices=["Reverse"], label="Options",interactive=True,elem_id="RP_usecommon")
            addout = gr.CheckboxGroup(choices=["mp4","Anime Gif"], label="Additional Output",interactive=True,elem_id="RP_usecommon")
        with gr.Row(visible=True):
            step = gr.Slider(label="Step", minimum=0, maximum=150, value=4, step=1)
            duration = gr.Slider(label="FPS", minimum=1, maximum=100, value=30, step=1)
            batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=8, value=1, step=1,visible = False)
        with gr.Row(visible=True):
            plans = gr.TextArea(label="Schedule")
        with gr.Row(visible=True):
            mp4pathd = gr.Textbox(label="mp4 output directory")
            mp4pathf = gr.Textbox(label="mp4 output filename")
        with gr.Row(visible=True):
            gifpathd = gr.Textbox(label="Anime gif output directory")
            gifpathf = gr.Textbox(label="Anime gif output filename")

        with gr.Row(visible=True):
            add_filename = gr.Textbox(label="Add text to filename")

        return [options, duration, plans, step, addout, batch_size, mp4pathd, mp4pathf, gifpathd, gifpathf, add_filename]

    def run(self, p, options, duration, plans, step, addout, batch, mp4pathd, mp4pathf, gifpathd, gifpathf, add_filename):
        self.__init__()

        p.rps_diff = True

        plans = plans.splitlines()
        plans = [f.split(";") for f in plans]
        all_prompts = []
        all_prompts_hr = []

        base_prompt = p.prompt.split("BREAK")[0]

        def makesubprompt(pro, tar, wei, ste):
            a = "" if tar in base_prompt else tar
            if pro == "": return f" BREAK ,{tar}" 
            if wei == 1:
                return f"{a} BREAK {base_prompt} [:{pro}:{ste}], {tar}"
            else:
                return f"{a} BREAK {base_prompt} [:({pro}:{wei}):{ste}], {tar}" 

        def makesubprompt_hr(pro, tar, wei, ste):
            a = "" if tar in base_prompt else tar
            if pro == "": return f" BREAK ,{tar}" 
            if wei == 1:
                return f"{a} BREAK {base_prompt} {pro}, {tar}"
            else:
                return f"{a} BREAK {base_prompt} ({pro}:{wei}), {tar}" 
        #pprint(plans)

        for plan in plans:
            if 3 > len(plan):
                sets = plan[0]
                if "=" in sets:
                    change, num = sets.split("=")
                    if change == "step":
                        step = int(num)
                    if "th" in change:
                        all_prompts.append(["th",num])
                        all_prompts_hr.append(None)
                elif "*" in sets:
                    num = int(sets.replace("*",""))
                    all_prompts.extend([["th",2]]+[base_prompt + ". BREAK " + base_prompt  + f" ,."]*num + [["th",None]])
                    all_prompts_hr.extend([["th",2]]+[base_prompt + ". BREAK " + base_prompt  + f" ,."]*num + [["th",None]])
                elif "ex-on" in sets:
                    strength = float(sets.split(",")[1]) if "," in sets else None
                    all_prompts.append(["ex-on",strength])
                    all_prompts_hr.append(None)
                elif "ex-off" in sets:
                    all_prompts.append(["ex-off"])
                    all_prompts_hr.append(None)
                elif sets == "0":
                    all_prompts.extend([["th",2], base_prompt + ". BREAK " + base_prompt  + f" ,.", ["th",None]])
                    all_prompts_hr.extend([["th",2], base_prompt + ". BREAK " + base_prompt  + f" ,.", ["th",None]])
                continue
            weights = parse_weights(plan[2])
            istep = step
            if len(plan) >=4:
                asteps = parse_steps(plan[3])
                if type(asteps) is list:
                    for astep in asteps:
                        all_prompts.append(base_prompt + makesubprompt(plan[0], plan[1], weights[0], astep))
                        all_prompts_hr.append(base_prompt + makesubprompt_hr(plan[0], plan[1], weights[0], astep))
                    continue
                else:
                    istep = astep
            for weight in weights:
                all_prompts.append(base_prompt + makesubprompt(plan[0], plan[1], weight, istep))
                all_prompts_hr.append(base_prompt + makesubprompt_hr(plan[0], plan[1], weight, istep))

        #pprint(all_prompts)

        results = {}
        output = None
        index = []

        for prompt in all_prompts:
            if type(prompt) == list: continue
            if prompt not in results.keys():
                results[prompt] = None

        print(f"Differential Regional Prompter Start")
        print(f"FPS = {duration}, {len(all_prompts)} frames, {round(len(all_prompts)/duration,3)} Sec")

        job = math.ceil((len(results)))

        allstep = job * p.steps
        total_tqdm.updateTotal(allstep)
        state.job_count = job

        if p.seed == -1 : p.seed = int(random.randrange(4294967294))

        timestamp_start = time.time()

        for prompt, prompt_hr in zip(all_prompts,all_prompts_hr):
            if type(prompt) == list:
                if prompt[0] == "th":
                    p.threshold = prompt[1]
                if prompt[0] == "ex-on":
                    p.seed_enable_extras = True
                    p.subseed_strength = strength if prompt[1] else 0.1
                if prompt[0] == "ex-off":
                    p.seed_enable_extras = False
                continue
            if results[prompt] is not None:
                continue
            p.prompt = prompt
            p.hr_prompt = prompt_hr

            processed = process_images(p)
            results[prompt] = processed.images[0]
            if output is None :output = processed
            else:
                if add_filename != "":
                    def find_latest_image(directory, start_time):
                        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")
                        latest_file = None
                        second_latest_file = None
                        latest_mtime = start_time
                        second_latest_mtime = start_time
                        
                        # 再帰的にすべてのファイルを検索
                        for filepath in glob.iglob(os.path.join(directory, "**"), recursive=True):
                            if os.path.isfile(filepath) and filepath.lower().endswith(image_extensions):
                                mtime = os.path.getmtime(filepath)
                                if mtime > latest_mtime:
                                    second_latest_file = latest_file
                                    second_latest_mtime = latest_mtime
                                    latest_file = filepath
                                    latest_mtime = mtime
                                elif mtime > second_latest_mtime:
                                    second_latest_file = filepath
                                    second_latest_mtime = mtime
                        
                        return latest_file, second_latest_file

                    latest_image, second_latest_image = find_latest_image(p.outpath_samples, timestamp_start)

                    if latest_image:
                        print("Latest image file:", latest_image)
                        directory, filename = os.path.split(latest_image)
                        name, ext = os.path.splitext(filename)
                        
                        new_filename = f"{name}{add_filename}{ext}"
                        
                        if second_latest_image:
                            second_name, _ = os.path.splitext(os.path.basename(second_latest_image))

                            match = re.match(r"(\d+)-(.+)", name)
                            second_match = re.match(r"(\d+)-(.+)", second_name)
                            
                            if match and second_match and match.group(2) == second_match.group(2):
                                base_name = second_name
                            else:
                                base_name = name                            
                            
                            if f"{add_filename}" in second_name:
                                add_text_count = len(re.findall(fr"{add_filename}", second_name))
                                new_filename = f"{base_name}{add_filename}{add_text_count}{ext}"
                            else:
                                new_filename = f"{base_name}{add_filename}{ext}"
                        
                        new_filepath = os.path.join(directory, new_filename)
                        os.rename(latest_image, new_filepath)
                output.images.extend(processed.images)

        all_result = []

        for prompt in all_prompts:
            if type(prompt) == list: continue
            all_result.append(results[prompt])

        if "Reverse" in options: all_result.reverse()

        outpath = p.outpath_samples
        if "Anime Gif" in addout:
            if gifpathd != "": outpath = os.path.join(outpath,gifpathd)

            try:
                os.makedirs(outpath)
            except FileExistsError:
                pass

            if gifpathf == "": gifpathf = "dfr"

            gifpath = gifpath_t = os.path.join(outpath, gifpathf + ".gif")
            
            is_file = os.path.isfile(gifpath)
            j = 1
            while is_file:
                gifpath = gifpath_t.replace(".gif",f"_{j}.gif")
                is_file = os.path.isfile(gifpath)
                j = j + 1

            all_result[0].save(gifpath, save_all=True, append_images=all_result[1:], optimize=False, duration=(1000 / duration), loop=0)

        outpath = p.outpath_samples
        if "mp4" in addout:
            if mp4pathd != "": outpath = os.path.join(outpath,mp4pathd)
            if mp4pathf == "": mp4pathf = "dfr"
            mp4path = mp4path_t = os.path.join(outpath, mp4pathf + ".mp4")

            try:
                os.makedirs(outpath)
            except FileExistsError:
                pass

            is_file = os.path.isfile(mp4path_t)
            j = 1
            while is_file:
                mp4path = mp4path_t.replace(".mp4",f"_{j}.mp4")
                is_file = os.path.isfile(mp4path)
                j = j + 1

            numpy_frames = [np.array(frame) for frame in all_result]

            with imageio.get_writer(mp4path, fps=duration) as writer:
                for numpy_frame in numpy_frames:
                    writer.append_data(numpy_frame)

        self.__init__()
        return output

    def settest1(self,valu):
        self.test1 = valu

def parse_steps(s):
    if "(" in s:
        step = s[s.index("("):]
        s = s.replace(step,"")
        step = int(step.strip("()"))
    else:
        step = 1

    if "-" in s:
        start,end = s.split("-")
        start,end = int(start), int(end)
        step = step if end > start else -step
        return list(range(start, end + step, step))
    
    if "*" in s:
        w, m = s.split("*")
        if w == "": w = 4
        return [w] * int(m)
    
    return int(s)

def parse_weights(s):
    if s == "": return[1]
    if "*" in s:
        w, m = s.split("*")
        if w == "": w = 1
        return [w] * int(m)

    if '(' in s:
        step = s[s.index("("):]
        s = s.replace(step,"")
        step = float(step.strip("()"))
    else:
        step = None

    out = []

    if "-" in s:
        rans = [x for x in s.split("-")]
        if step is None:
            digit = len(rans[0].split(".")[1])
            step = 10 ** -digit
        rans = [float(r) for r in rans]
        for start, end in zip(rans[:-1],rans[1:]):
            #print(start,end)
            sign = 1 if end > start else -1
            now = start
            for i in range(int(abs(end-start)//step) + 1):
                out.append(now)
                now = now + step * sign
    else:
        out =[float(s)]

    if out == []:out = [1]
    out = [round(x, 5) for x in out]
    return out