from random import choices
from matplotlib.style import available
import torch
import csv
import math
import gradio as gr
import os.path
from pprint import pprint
import modules.ui
import ldm.modules.attention as atm
from modules import shared,scripts
from modules.processing import Processed,paths

#'"name","mode","divide ratios,"use base","baseratios","usecom","usencom",\n'

PRESETS =[
    ["Vertical-3", "Vertical",'"1,1,1"',"","False","False","False"],
    ["Horizontal-3", "Horizontal",'"1,1,1"',"","False","False","False"],
    ["Horizontal-7", "Horizontal",'"1,1,1,1,1,1,1"',"0.2","True","False","False"],
]

class Script(modules.scripts.Script):
    def __init__(self):
        self.mode = ""
        self.w = 0
        self.h = 0
        self.usebase = False
        self.aratios = []
        self.bratios = []
        self.divide = 0
        self.count = 0
        self.pn = True
        self.hr = False
        self.hr_scale = 0
        self.hr_w = 0
        self.hr_h = 0
        self.batch_size = 0
        self.orig_all_prompts = []
        self.orig_all_negative_prompts = []
        self.all_prompts = []
        self.all_negative_prompts = []
        self.imgcount = 0

    def title(self):
        return "Regional Prompter"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    def ui(self, is_img2img):
        path_root = scripts.basedir()
        filepath = os.path.join(path_root,"scripts", "regional_prompter_presets.csv")

        presets=[]

        presets  = loadpresets(filepath)

        with gr.Accordion("Regional Prompter", open=False):
            with gr.Row():
                active = gr.Checkbox(value=False, label="Active",interactive=True,elem_id="RP_active")
            with gr.Row():
                mode = gr.Radio(label="Divide mode", choices=["Horizontal", "Vertical"], value="Horizontal",  type="value", interactive=True)
            with gr.Row(visible=True):
                ratios = gr.Textbox(label="Divide Ratio",lines=1,value="1,1",interactive=True,elem_id="RP_divide_ratio",visible=True)
                baseratios = gr.Textbox(label="Base Ratio", lines=1,value="0.2",interactive=True,  elem_id="RP_base_ratio", visible=True)
            with gr.Row():
                usebase = gr.Checkbox(value=False, label="Use base prompt",interactive=True, elem_id="RP_usebase")
                usecom = gr.Checkbox(value=False, label="Use common prompt",interactive=True,elem_id="RP_usecommon")
                usencom = gr.Checkbox(value=False, label="Use common negative prompt",interactive=True,elem_id="RP_usecommon")
            with gr.Row():
                debug = gr.Checkbox(value=False, label="debug", interactive=True, elem_id="RP_debug")

            with gr.Accordion("Presets",open = False):
                with gr.Row():
                    availablepresets = gr.Dropdown(label="Presets", choices=[pr[0] for pr in presets], type="index")
                    applypresets = gr.Button(value="Apply Presets",variant='primary',elem_id="RP_applysetting")
                with gr.Row():
                    presetname = gr.Textbox(label="Preset Name",lines=1,value="",interactive=True,elem_id="RP_preset_name",visible=True)
                    savesets = gr.Button(value="Save to Presets",variant='primary',elem_id="RP_savesetting")

            settings = [mode, ratios, baseratios, usebase, usecom, usencom]
        
        def setpreset(select):
            presets = loadpresets(filepath)
            preset = presets[select]
            preset = preset[1:]
            def booler(text):
                return text == "TRUE" or text == "true" or text == "True"
            preset[1],preset[2] = preset[1].replace('"',""),preset[2].replace('"',"")
            preset[3],preset[4],preset[5] = booler(preset[3]),booler(preset[4]),booler(preset[5])
            return [gr.update(value = pr) for pr in preset]

        applypresets.click(fn=setpreset, inputs = availablepresets, outputs=settings)
        savesets.click(fn=savepresets, inputs = [presetname,*settings],outputs=availablepresets)
                
        return [active, debug, mode, ratios, baseratios, usebase, usecom, usencom]

    def process(self, p, active, debug, mode, aratios, bratios, usebase, usecom, usencom):
        if active:
            savepresets("lastrun",mode, aratios, usebase, bratios, usecom, usencom)
            self.__init__()
            self.mode = mode
            self.w = p.width
            self.h = p.height
            self.batch_size = p.batch_size

            self.debug = debug
            self.usebase = usebase

            self.hr = p.enable_hr
            self.hr_w = (p.hr_resize_x if p.hr_resize_x > p.width else p.width * p.hr_scale)
            self.hr_h = (p.hr_resize_y if p.hr_resize_y > p.height else p.height * p.hr_scale)

            self, p = promptdealer(self, p, aratios, bratios, usebase, usecom, usencom)

            self.handle = hook_forwards(self, p.sd_model.model.diffusion_model)

            self.pt, self.nt ,ppt,pnt= tokendealer(p)

            print(f"pos tokens : {ppt}, neg tokens : {pnt}")
            
            self.eq = True if len(self.pt) == len(self.nt) else False
        else:   
            if hasattr(self,"handle"):
                hook_forwards(self, p.sd_model.model.diffusion_model, remove=True)
                del self.handle


        return p

    def postprocess_image(self, p,pp, active, debug, mode, aratios, bratios, usebase, usecom, usencom):
        if active:
            if usecom:
                p.prompt = self.orig_all_prompt[0]
                p.all_prompts[self.imgcount] = self.orig_all_prompt[self.imgcount]  
            if usencom:
                p.negative_prompt = self.orig_all_negative_prompt[0]
                p.all_negative_prompts[self.imgcount] = self.orig_all_negative_prompt[self.imgcount] 
            self.imgcount += 1
        p.extra_generation_params["Regional Prompter"] = f"mode:{mode},divide ratio : {aratios}, Use base : {usebase}, Base ratio : {bratios}, Use common : {usecom}, Use N-common : {usencom}"
        return p


    def postprocess(self, p, processed, *args):
        if hasattr(self,"handle"):
            hook_forwards(self, p.sd_model.model.diffusion_model, remove=True)
            del self.handle

        with open(os.path.join(paths.data_path, "params.txt"), "w", encoding="utf8") as file:
            processed = Processed(p, [], p.seed, "")
            file.write(processed.infotext(p, 0))


def hook_forward(self, module):
    def forward(x, context=None, mask=None):
        if self.debug :
            print("input : ", x.size())
            print("tokens : ", context.size())
            print("module : ", module.lora_layer_name)

        height = self.h
        width = self.w

        def hr_cheker(n):
            return (n != 0) and (n & (n - 1) == 0)

        if not hr_cheker(height * width // x.size()[1]) and self.hr:
            height = self.hr_h
            width = self.hr_w

        contexts = context.clone()

        def separatecalc(x, contexts, mask, pn,divide):
            sumer = 0
            h_states = []

            tll = self.pt if pn else self.nt
            if self.debug : print(f"tokens : {tll},pn : {pn}")

            for i, tl in enumerate(tll):
                context = contexts[:, tl[0] * 77 : tl[1] * 77, :]
                if self.debug : print(f"tokens : {tl[0]*77}-{tl[1]*77}")

                if self.usebase:
                    if i != 0:
                        area = self.aratios[i - 1]
                        bweight = self.bratios[i - 1]
                else:
                    area = self.aratios[i]

                h = module.heads // divide
                q = module.to_q(x)

                context = atm.default(context, x)
                k = module.to_k(context)
                v = module.to_v(context)

                q, k, v = map(lambda t: atm.rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

                sim = atm.einsum("b i d, b j d -> b i j", q, k) * module.scale

                if atm.exists(mask):
                    mask = atm.rearrange(mask, "b ... -> b (...)")
                    max_neg_value = -torch.finfo(sim.dtype).max
                    mask = atm.repeat(mask, "b j -> (b h) () j", h=h)
                    sim.masked_fill_(~mask, max_neg_value)

                attn = sim.softmax(dim=-1)

                out = atm.einsum("b i j, b j d -> b i d", attn, v)
                out = atm.rearrange(out, "(b h) n d -> b n (h d)", h=h)
                out = module.to_out(out)

                if len(self.nt) == 1 and not pn:
                    if self.debug : print("return out for NP")
                    return out

                xs = x.size()[1]
                scale = round(math.sqrt(height * width / xs))

                dsh = round(height / scale)
                dsw = round(width / scale)
                ha, wa = xs % dsh, xs % dsw
                if ha == 0:
                    dsw = int(xs / dsh)
                elif wa == 0:
                    dsh = int(xs / dsw)

                if self.debug : print(scale, dsh, dsw, dsh * dsw, x.size()[1])

                if i == 0 and self.usebase:
                    outb = out.clone()
                    if "Horizontal" in self.mode:
                        outb = outb.reshape(outb.size()[0], dsh, dsw, outb.size()[2])
                    continue
                add = 0

                cad = 0 if self.usebase else 1

                if "Horizontal" in self.mode:
                    sumer = sumer + int(dsw * area[1]) - int(dsw * area[0])
                    if i == self.divide - cad:
                        add = sumer - dsw
                    out = out.reshape(out.size()[0], dsh, dsw, out.size()[2])
                    out = out[:, :, int(dsw * area[0] + add) : int(dsw * area[1]), :]
                    if self.debug : print(f"sumer:{sumer},dsw:{dsw},add:{add}")
                    if self.usebase:
                        outb_t = outb[:, :, int(dsw * area[0] + add) : int(dsw * area[1]), :].clone()
                        out = out * (1 - bweight) + outb_t * bweight
                elif "Vertical" in self.mode:
                    sumer = sumer + int(dsw * dsh * area[1]) - int(dsw * dsh * area[0])
                    if i == self.divide - cad:
                        add = sumer - dsw * dsh
                    out = out[:, int(dsw * dsh * area[0] + add) : int(dsw * dsh * area[1]), :]
                    if self.debug : print(f"sumer:{sumer},dsw*dsh:{dsw*dsh},add:{add}")
                    if self.usebase:
                        outb_t = outb[:,int(dsw * dsh * area[0] + add) : int(dsw * dsh * area[1]),:,].clone()
                        out = out * (1 - bweight) + outb_t * bweight
                h_states.append(out)
            if self.debug:
                for h in h_states :
                    print(f"divided : {h.size()}")

            if "Horizontal" in self.mode:
                ox = torch.cat(h_states, dim=2)
                ox = ox.reshape(x.size()[0], x.size()[1], x.size()[2])
            elif "Vertical" in self.mode:
                ox = torch.cat(h_states, dim=1)
            return ox

        if self.eq:
            ox = separatecalc(x, contexts, mask, True, 1)
            if self.debug : print("same token size and divisions")
        elif x.size()[0] == 1 * self.batch_size:
            ox = separatecalc(x, contexts, mask, self.pn, 1)
            if self.debug : print("different tokens size")
        else:
            px, nx = x.chunk(2)
            opx = separatecalc(px, contexts, mask, True, 2)
            onx = separatecalc(nx, contexts, mask, False, 2)
            ox = torch.cat([opx, onx])
            if self.debug : print("same token size and different divisions")
            
        self.count += 1

        if self.count == 16:
            self.pn = not self.pn
            self.count = 0
        if self.debug : print(f"output : {ox.size()}")
        return ox

    return forward


def hook_forwards(self, root_module: torch.nn.Module, remove=False):
    for name, module in root_module.named_modules():
        if "attn2" in name and module.__class__.__name__ == "CrossAttention":
            module.forward = hook_forward(self, module)
            if remove:
                del module.forward


def tokendealer(p):
    ppl = p.prompt.split("BREAK")
    npl = p.negative_prompt.split("BREAK")
    pt, nt, ppt, pnt = [], [], [], []

    padd = 0
    for pp in ppl:
        _, tokens = shared.sd_model.cond_stage_model.tokenize_line(pp)
        pt.append([padd, tokens // 75 + 1 + padd])
        ppt.append(tokens)
        padd = tokens // 75 + 1 + padd

    padd = 0
    for np in npl:
        _, tokens = shared.sd_model.cond_stage_model.tokenize_line(np)
        nt.append([padd, tokens // 75 + 1 + padd])
        pnt.append(tokens)
        padd = tokens // 75 + 1 + padd

    return pt, nt, ppt, pnt


def promptdealer(self, p, aratios, bratios, usebase, usecom, usencom):
    aratios = [float(a) for a in aratios.split(",")]
    aratios = [a / sum(aratios) for a in aratios]

    for i, a in enumerate(aratios):
        if i == 0:
            continue
        aratios[i] = aratios[i - 1] + a

    divide = len(aratios)
    aratios_o = [0] * divide

    for i in range(divide):
        if i == 0:
            aratios_o[i] = [0, aratios[0]]
        elif i < divide:
            aratios_o[i] = [aratios[i - 1], aratios[i]]
        else:
            aratios_o[i] = [aratios[i], ""]
    if self.debug : print("regions : ", aratios_o)

    self.aratios = aratios_o
    try:
        self.bratios = [float(b) for b in bratios.split(",")]
    except:
        self.bratios = [0]

    if divide > len(self.bratios):
        while divide >= len(self.bratios):
            self.bratios.append(self.bratios[0])

    self.divide = divide

    if usecom:
        self.orig_all_prompt = p.all_prompts
        self.prompt = p.prompt = comdealer(p.prompt)
        for pr in p.all_prompts:
            self.all_prompts.append(comdealer(pr))
        p.all_prompts = self.all_prompts

    if usencom:
        self.orig_all_negative_prompt = p.all_negative_prompts
        self.negative_prompt = p.negative_prompt = comdealer(p.negative_prompt)
        for pr in p.all_negative_prompts:
            self.all_negative_prompts.append(comdealer(pr))
        p.all_negative_prompts =self.all_negative_prompts

    return self, p


def comdealer(prompt):
    ppl = prompt.split("BREAK")
    for i in range(len(ppl)):
        if i == 0:
            continue
        ppl[i] = ppl[0] + ", " + ppl[i]
    ppl = ppl[1:]
    prompt = "BREAK ".join(ppl)
    return prompt

def savepresets(name,mode, ratios, baseratios, usebase,usecom, usencom):
    path_root = scripts.basedir()
    filepath = os.path.join(path_root,"scripts", "regional_prompter_presets.csv")
    try:
        with open(filepath,mode = 'r',encoding="utf-8") as f:
            presets = f.readlines()
            pr = f'{name},{mode},"{ratios}","{baseratios}",{str(usebase)},{str(usecom)},{str(usencom)}\n'
            written = False
            if name == "lastrun":
                for i, preset in enumerate(presets):
                    if "lastrun" in preset :
                        presets[i] = pr
                        written = True
            if not written : presets.append(pr)
        with open(filepath,mode = 'w',encoding="utf-8") as f:
            f.writelines(presets)
    except Exception as e:
        print(e)
    presets = loadpresets(filepath)
    return gr.update(choices = [pr[0] for pr in presets])

def loadpresets(filepath):
    presets = []
    try:
        with open(filepath,encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > 5:
                    presets.append(row)
            presets = presets[1:]
    except OSError as e:
        presets=PRESETS
        print("ERROR")
        if not os.path.isfile(filepath):
            try:
                with open(filepath,mode = 'w',encoding="utf-8") as f:
                    f.writelines('"name","mode","divide ratios,"use base","baseratios","usecom","usencom"\n')
                    for pr in presets:
                        text = ",".join(pr) + "\n"
                        f.writelines(text)
            except:
                pass
    return presets
