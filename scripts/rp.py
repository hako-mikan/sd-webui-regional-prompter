import torch
import math
import gradio as gr
import os.path
from pprint import pprint
import modules.ui
import ldm.modules.attention as atm
from modules import shared
from modules.processing import Processed,paths


class Script(modules.scripts.Script):
    def __init__(self):
        self.mode = ""
        self.w = 0
        self.h = 0
        self.usebase = False
        self.aratios = []
        self.bratios = []
        self.handle = None
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

        return [active, mode, ratios, usebase, baseratios, debug, usecom, usencom]

    def process(self, p, active, mode, aratios, usebase, bratios, debug, usecom, usencom):
        if active:
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
            if self.handle:
                hook_forwards(self, p.sd_model.model.diffusion_model, remove=True)

        return p

    def postprocess_image(self, p,pp, active, mode, aratios, usebase, bratios, debug, usecom, usencom):
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
        if self.handle:
            hook_forwards(self, p.sd_model.model.diffusion_model, remove=True)

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

        def separatecalc(x, contexts, mask, pn):
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

                h = module.heads

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
            ox = separatecalc(x, contexts, mask, True)
        elif x.size()[0] == 1 * self.batch_size:
            ox = separatecalc(x, contexts, mask, self.pn)
        else:
            px, nx = x.chunk(2)
            if self.debug : print(px.size(),nx.size())
            opx = separatecalc(px, contexts, mask, True)
            onx = separatecalc(nx, contexts, mask, False)
            ox = torch.cat([opx, onx])

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
    self.bratios = [float(b) for b in bratios.split(",")]

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
import torch
import math
import gradio as gr
import os.path
from pprint import pprint
import modules.ui
import ldm.modules.attention as atm
from modules import shared
from modules.processing import Processed,paths


class Script(modules.scripts.Script):
    def __init__(self):
        self.mode = ""
        self.w = 0
        self.h = 0
        self.usebase = False
        self.aratios = []
        self.bratios = []
        self.handle = None
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

        return [active, mode, ratios, usebase, baseratios, debug, usecom, usencom]

    def process(self, p, active, mode, aratios, usebase, bratios, debug, usecom, usencom):
        if active:
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
            if self.handle:
                hook_forwards(self, p.sd_model.model.diffusion_model, remove=True)

        return p

    def postprocess_image(self, p,pp, active, mode, aratios, usebase, bratios, debug, usecom, usencom):
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
        if self.handle:
            hook_forwards(self, p.sd_model.model.diffusion_model, remove=True)

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

        def separatecalc(x, contexts, mask, pn):
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

                h = module.heads

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
            ox = separatecalc(x, contexts, mask, True)
        elif x.size()[0] == 1 * self.batch_size:
            ox = separatecalc(x, contexts, mask, self.pn)
        else:
            px, nx = x.chunk(2)
            if self.debug : print(px.size(),nx.size())
            opx = separatecalc(px, contexts, mask, True)
            onx = separatecalc(nx, contexts, mask, False)
            ox = torch.cat([opx, onx])

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
    self.bratios = [float(b) for b in bratios.split(",")]

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
