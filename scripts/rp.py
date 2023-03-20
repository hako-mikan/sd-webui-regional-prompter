from sre_constants import BRANCH
import torch
import math
import gradio as gr
from pprint import pprint
import modules.ui
import ldm.modules.attention as atm

class Script(modules.scripts.Script):   

    def __init__(self):
        self.mode = ""
        self.w = 0
        self.h = 0
        self.usebase = False
        self.aratios = []
        self.bratios = []
        self.handle = None
        self.count = 0
        self.hr = False
        self.hr_scale = 0
        self.hr_w = 0
        self.hr_h = 0

    def title(self):
        return "Regional Prompter"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Regional Prompter",open = False):
            with gr.Row():
                with gr.Column(min_width = 50, scale=1):
                    active = gr.Checkbox(value = False,label="Active",interactive =True,elem_id="RP_active")
                    usebase= gr.Checkbox(value = False,label="Use base prompt",interactive =True,elem_id="RP_usebase")
                    mode = gr.Radio(label = "Divide mode",choices = ["Horizontal", "Vertical"], value ="Horizontal",type = "value",interactive =True) 
                with gr.Row(visible = True):
                    ratios = gr.Textbox(label="Divide Ratio",lines=1,value="1,1",interactive =True,elem_id="RP_divide_ratio",visible = True)
                    baseratios = gr.Textbox(label="Base Ratio",lines=1,value="0.2",interactive =True,elem_id="RP_base_ratio",visible = True)
                debug = gr.Checkbox(value = False,label="debug",interactive =True,elem_id="RP_debug")

        return [active,mode,ratios,usebase,baseratios,debug]

    def process(self, p,active,mode,aratios,usebase,bratios,debug):
        if  active :
            self.__init__()
            self.mode = mode
            self.w = p.width
            self.h = p.height
            self.debug = debug
            self.usebase = usebase
            self.hr = p.enable_hr
            self.hr_w = p.hr_resize_x if p.hr_resize_x > p.width else p.width * p.hr_scale
            self.hr_h = p.hr_resize_y if p.hr_resize_y > p.height else p.height * p.hr_scale
            breaks,nbreaks = p.prompt.count("BREAK"), p.negative_prompt.count("BREAK")
            if breaks > 0:
                np = p.negative_prompt.split("BREAK")
                ok = False
                if breaks== nbreaks:
                    ok = True
                elif breaks > nbreaks:
                   while breaks >= len(np):
                        np.append(np[0])
                elif breaks < nbreaks:
                        np = np[0:breaks+1]
                for i ,n in enumerate(np):
                    if n.isspace() or n =="":
                        np[i] = ","
                        ok = False
                if not ok : 
                    p.negative_prompt = " BREAK ".join(np)
                    p.all_negative_prompts = [p.negative_prompt] * len(p.all_negative_prompts)
            else:
                return
            aratios = [float(a) for a in aratios.split(",")]
            aratios = [a / sum(aratios) for a in aratios]
            for i,a in enumerate(aratios):
                if i == 0 : continue
                aratios[i] = aratios[i-1] + a
            self.count = len(aratios)
            aratios_o = [0]*self.count
            for i in range(self.count):
                if i == 0 : aratios_o[i] = [0,aratios[0]]
                elif i < self.count : aratios_o[i] = [aratios[i-1],aratios[i]]
                else: aratios_o[i] = [aratios[i],""] 
            if self.debug : print(aratios_o)
            self.aratios = aratios_o
            self.bratios = [float(b) for b in bratios.split(",")]
            if  self.count > len(self.bratios):
                while self.count >= len(self.bratios):
                    self.bratios.append(self.bratios[0])
            self.handle = hook_forwards(self,p.sd_model.model.diffusion_model)
        return p

    def postprocess(self, p, processed, testattn, *args):
        hook_forwards(self,p.sd_model.model.diffusion_model,remove = True)

def hook_forward(self,module):
    def forward(x, context=None, mask=None):
        if self.debug : print("input",x.size())
        if self.debug : print(context.size())
        if self.debug : pprint(module.lora_layer_name)

        height = self.h
        width =self.w

        def hr_cheker(n):
            return (n != 0) and (n & (n-1) == 0)

        if not hr_cheker(height*width//x.size()[1]) and self.hr:
            height = self.hr_h
            width = self.hr_w

        sumer = 0
        h_states= []
        contexts = context
        for i in range(contexts.size()[1]//77):
            context = contexts[:,i*77:(i+1)*77,:]
            
            if self.usebase:
                if i != 0:
                    area = self.aratios[i -1]
                    bweight = self.bratios[i-1]
            else:
                area  = self.aratios[i]

            h = module.heads

            q = module.to_q(x)
            context = atm.default(context, x)
            k = module.to_k(context)
            v = module.to_v(context)

            q, k, v = map(lambda t: atm.rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = atm.einsum('b i d, b j d -> b i j', q, k) * module.scale

            if atm.exists(mask):
                mask = atm.rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = atm.repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)

            out = atm.einsum('b i j, b j d -> b i d', attn, v)
            out = atm.rearrange(out, '(b h) n d -> b n (h d)', h=h)
            out = module.to_out(out)

            xs = x.size()[1]
            scale = round(math.sqrt(height*width/xs))

            dsh = round(height/scale)
            dsw = round(width/scale)
            ha,wa = xs%dsh,xs%dsw
            if ha ==0:
                dsw = int(xs /dsh) 
            elif wa ==0:
                dsh = int(xs/dsw)

            if self.debug : print(scale,dsh,dsw,dsh*dsw,x.size()[1])

            if i == 0 and self.usebase:
                outb = out.clone()
                if "Horizontal" in self.mode:
                    outb = outb.reshape(outb.size()[0],  dsh, dsw, outb.size()[2])
                continue
            add = 0

            cad = 0 if self.usebase else 1

            if "Horizontal" in self.mode:
                sumer = sumer + int(dsw*area[1]) - int(dsw*area[0])
                if i == self.count - cad :
                    add = sumer - dsw
                out = out.reshape(out.size()[0],  dsh, dsw, out.size()[2])
                out = out[:,:,int(dsw*area[0] + add):int(dsw*area[1]),:]
                if self.debug : print(f"sumer:{sumer},dsw:{dsw},add:{add}")
                if self.usebase : 
                    outb_t = outb[:,:,int(dsw*area[0] + add):int(dsw*area[1]),:].clone()
                    out = out * (1 - bweight) + outb_t * bweight
            elif "Vertical" in self.mode:
                sumer = sumer + int(dsw*dsh*area[1]) - int(dsw*dsh*area[0])
                if i == self.count - cad:
                    add = sumer - dsw*dsh
                out = out[:,int(dsw*dsh*area[0]+ add):int(dsw*dsh*area[1]),:]
                if self.debug : print(f"sumer:{sumer},dsw*dsh:{dsw*dsh},add:{add}")
                if self.usebase : 
                    outb_t = outb[:,int(dsw*dsh*area[0] + add):int(dsw*dsh*area[1]),:].clone()
                    out = out * (1 - bweight) + outb_t * bweight
            h_states.append(out)
            if self.debug : 
                for h in h_states:
                    print(h.size())

        if "Horizontal" in self.mode:
            ox = torch.cat(h_states,dim = 2)
            ox = ox.reshape(x.size()[0],x.size()[1],x.size()[2])
        elif "Vertical" in self.mode:
            ox = torch.cat(h_states,dim = 1)
        return ox

    return forward

def hook_forwards(self,root_module: torch.nn.Module,remove = False):
    for name, module in root_module.named_modules():
        if "attn2" in name and module.__class__.__name__ == "CrossAttention":
            module.forward = hook_forward(self,module)
            if remove: del module.forward
