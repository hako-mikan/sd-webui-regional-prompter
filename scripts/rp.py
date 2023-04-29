import os.path
from importlib import reload
from pprint import pprint
import gradio as gr
import modules.ui
from modules import paths, scripts, shared
from modules.processing import Processed
from modules.script_callbacks import (CFGDenoisedParams, CFGDenoiserParams, on_cfg_denoised, on_cfg_denoiser)
import scripts.attention
import scripts.latent
import scripts.regions
reload(scripts.regions) # update without restarting web-ui.bat
reload(scripts.attention)
reload(scripts.latent)
import json  # Presets.
from scripts.attention import (TOKENS, hook_forwards, reset_pmasks, savepmasks)
from scripts.latent import (denoised_callback_s, denoiser_callback_s, lora_namer, regioner, setloradevice, setuploras, unloadlorafowards)
from scripts.regions import (CBLACK, IDIM, KEYBRK, KEYCOMM, KEYPROMPT, create_canvas, detect_mask, detect_polygons, floatdef, inpaintmaskdealer, makeimgtmp, matrixdealer)

def lange(l):
    return range(len(l))

orig_batch_cond_uncond = shared.batch_cond_uncond

PRESETS =[
    ["Vertical-3", "Vertical",'1,1,1',"",False,False,False,"Attention",False,"0","0"],
    ["Horizontal-3", "Horizontal",'1,1,1',"",False,False,False,"Attention",False,"0","0"],
    ["Horizontal-7", "Horizontal",'1,1,1,1,1,1,1',"0.2",True,False,False,"Attention",False,"0","0"],
    ["Twod-2-1", "Horizontal",'1,2,3;1,1',"0.2",False,False,False,"Attention",False,"0","0"],
]

ATTNSCALE = 8 # Initial image compression in attention layers.

class Script(modules.scripts.Script):
    def __init__(self):
        self.active = False
        self.mode = ""
        self.calcmode = ""
        self.w = 0
        self.h = 0
        self.debug = False
        self.usebase = False
        self.usecom = False
        self.usencom = False
        self.batch_size = 0
        self.orig_all_prompts = []
        self.orig_all_negative_prompts = []

        self.cells = False
        self.aratios = []
        self.bratios = []
        self.divide = 0
        self.count = 0
        self.pn = True
        self.hr = False
        self.hr_scale = 0
        self.hr_w = 0
        self.hr_h = 0
        self.all_prompts = []
        self.all_negative_prompts = []
        self.imgcount = 0
        # for latent mode
        self.filters = []
        self.neg_filters = []
        self.anded = False
        self.lora_applied = False
        self.lactive = False
        # for inpaintmask
        self.indmaskmode = False
        self.regmasks = None
        self.regbase = None
        #for prompt region
        self.pe = []
        self.modep =False
        self.calced = False
        self.step = 0
        self.lpactive = False

    def title(self):
        return "Regional Prompter"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    infotext_fields = None
    paste_field_names = []

    def ui(self, is_img2img):
        path_root = modules.scripts.basedir()
        filepath = os.path.join(path_root,"scripts", "regional_prompter_presets.json")

        presets = []

        presets = loadpresets(filepath)

        with gr.Accordion("Regional Prompter", open=False):
            with gr.Row():
                active = gr.Checkbox(value=False, label="Active",interactive=True,elem_id="RP_active")
            with gr.Row():
                mode = gr.Radio(label="Divide mode", choices=["Horizontal", "Vertical","Mask","Prompt","Prompt-Ex"], value="Horizontal",  type="value", interactive=True)
                calcmode = gr.Radio(label="Generation mode", choices=["Attention", "Latent","None"], value="Attention",  type="value", interactive=True)
            with gr.Row(visible=True):
                ratios = gr.Textbox(label="Divide Ratio",lines=1,value="1,1",interactive=True,elem_id="RP_divide_ratio",visible=True)
                baseratios = gr.Textbox(label="Base Ratio", lines=1,value="0.2",interactive=True,  elem_id="RP_base_ratio", visible=True)
            with gr.Row():
                usebase = gr.Checkbox(value=False, label="Use base prompt",interactive=True, elem_id="RP_usebase")
                usecom = gr.Checkbox(value=False, label="Use common prompt",interactive=True,elem_id="RP_usecommon")
                usencom = gr.Checkbox(value=False, label="Use common negative prompt",interactive=True,elem_id="RP_usecommon")
            with gr.Row():
                with gr.Column():
                    maketemp = gr.Button(value="visualize and make template")
                    template = gr.Textbox(label="template",interactive=True,visible=True)
                with gr.Column():
                    areasimg = gr.Image(type="pil", show_label  = False).style(height=256,width=256)
                    threshold = gr.Textbox(label = "threshold",value = 0.4,interactive=True,)

            with gr.Row():
                with gr.Column():
                    polymask = gr.Image(source = "upload", mirror_webcam = False, type = "numpy", tool = "sketch")
                    num = gr.Slider(label="Region", minimum=0, maximum=CBLACK, step=1, value=1)
                    canvas_width = gr.Slider(label="Canvas Width", minimum=64, maximum=2048, value=512, step=8)
                    canvas_height = gr.Slider(label="Canvas Height", minimum=64, maximum=2048, value=512, step=8)
                    btn = gr.Button(value = "Draw region")
                    btn2 = gr.Button(value = "Display mask")
                    cbtn = gr.Button(value="Create mask area")
                with gr.Column():
                    showmask = gr.Image(shape=(IDIM, IDIM))
            btn.click(detect_polygons, inputs = [polymask,num], outputs = [polymask,num])
            btn2.click(detect_mask, inputs = [polymask,num], outputs = [showmask])
            cbtn.click(fn=create_canvas, inputs=[canvas_height, canvas_width], outputs=[polymask])

            with gr.Accordion("Presets",open = False):
                with gr.Row():
                    availablepresets = gr.Dropdown(label="Presets", choices=[pr["name"] for pr in presets], type="index")
                    applypresets = gr.Button(value="Apply Presets",variant='primary',elem_id="RP_applysetting")
                with gr.Row():
                    presetname = gr.Textbox(label="Preset Name",lines=1,value="",interactive=True,elem_id="RP_preset_name",visible=True)
                    savesets = gr.Button(value="Save to Presets",variant='primary',elem_id="RP_savesetting")
            with gr.Row():
                nchangeand = gr.Checkbox(value=False, label="disable convert 'AND' to 'BREAK'", interactive=True, elem_id="RP_ncand")
                debug = gr.Checkbox(value=False, label="debug", interactive=True, elem_id="RP_debug")
                lnter = gr.Textbox(label="LoRA in negative textencoder",value="0",interactive=True,elem_id="RP_ne_tenc_ratio",visible=True)
                lnur = gr.Textbox(label="LoRA in negative U-net",value="0",interactive=True,elem_id="RP_ne_unet_ratio",visible=True)
            settings = [mode, ratios, baseratios, usebase, usecom, usencom, calcmode, nchangeand, lnter, lnur, threshold]
        
        self.infotext_fields = [
                (active, "RP Active"),
                (mode, "RP Divide mode"),
                (calcmode, "RP Calc Mode"),
                (ratios, "RP Ratios"),
                (baseratios, "RP Base Ratios"),
                (usebase, "RP Use Base"),
                (usecom, "RP Use Common"),
                (usencom, "RP Use Ncommon"),
                (nchangeand,"RP Change AND"),
                (lnter,"RP LoRA Neg Te Ratios"),
                (lnur,"RP LoRA Neg U Ratios"),
                (threshold,"RP threshold"),
        ]

        for _,name in self.infotext_fields:
            self.paste_field_names.append(name)

        def setpreset(select):
            presets = loadpresets(filepath)
            preset = presets[select]
            preset = [fmt(preset.get(k, vdef)) for (k,fmt,vdef) in PRESET_KEYS]
            preset = preset[1:] # Remove name.
            # TODO: Need to grab current value from gradio. Must we send it as input?
            preset = ["" if p is None else p for p in preset]
            return [gr.update(value = pr) for pr in preset]
        

        maketemp.click(fn=makeimgtmp, inputs =[ratios,mode,usecom,usebase],outputs = [areasimg,template])
        applypresets.click(fn=setpreset, inputs = availablepresets, outputs=settings)
        savesets.click(fn=savepresets, inputs = [presetname,*settings],outputs=availablepresets)
                
        return [active, debug, mode, ratios, baseratios, usebase, usecom, usencom, calcmode, nchangeand, lnter, lnur, threshold, polymask]

    def process(self, p, active, debug, mode, aratios, bratios, usebase, usecom, usencom, calcmode, nchangeand, lnter, lnur, threshold, polymask):
        if not active:
            unloader(self,p)
            return p

        p.extra_generation_params.update({
            "RP Active":active,
            "RP Divide mode":mode,
            "RP Calc Mode":calcmode,
            "RP Ratios": aratios,
            "RP Base Ratios": bratios,
            "RP Use Base":usebase,
            "RP Use Common":usecom,
            "RP Use Ncommon": usencom,
            "RP Change AND" : nchangeand,
            "RP LoRA Neg Te Ratios": lnter,
            "RP LoRA Neg U Ratios": lnur,
            "RP threshold": threshold,
                })

        savepresets("lastrun",mode, aratios,bratios, usebase, usecom, usencom, calcmode, nchangeand, lnter, lnur, threshold, polymask)

        self.__init__()

        self.active = True
        self.calcmode = calcmode
        self.debug = debug
        self.usebase = usebase
        self.usecom = usecom
        self.usencom = usencom
        self.w = p.width
        self.h = p.height
        self.batch_size = p.batch_size
        self.orig_all_prompts = p.all_prompts
        self.orig_all_negative_prompts = p.all_negative_prompts

        comprompt = comnegprompt = None
        # SBM ddim / plms detection.
        self.isvanilla = p.sampler_name in ["DDIM", "PLMS", "UniPC"]

        if self.h % ATTNSCALE != 0 or self.w % ATTNSCALE != 0:
            # Testing shows a round down occurs in model.
            print("Warning: Nonstandard height / width.")
            self.h = self.h - self.h % ATTNSCALE
            self.w = self.w - self.w % ATTNSCALE
        
        if KEYPROMPT in p.prompt.upper():
            mode = "Prompt"
            p.replace(KEYPROMPT,KEYBRK)

        self.indmaskmode = mode == "Mask"

        self.mode = mode
  
        if not nchangeand and "AND" in p.prompt.upper():
            p.prompt = p.prompt.replace("AND",KEYBRK)
            for i in lange(p.all_prompts):
                p.all_prompts[i] = p.all_prompts[i].replace("AND",KEYBRK)
            self.anded = True
            

        if "Prompt" not in mode: # skip region assign in prompt mode
            self.cells = not "Mask" in mode

            if KEYBRK in p.prompt and not "Mask" in mode:
                keychanger = makeimgtmp(aratios,mode,usecom,usebase,inprocess = True)
                keychanger = keychanger[:-1]
                print(keychanger,p.prompt)
                for change in keychanger:
                    p.prompt= p.prompt.replace(KEYBRK,change,1)

            if KEYCOMM in p.prompt: # Automatic common toggle.
                self.usecom = True
            self.usencom = usencom
            
            if KEYCOMM in p.negative_prompt: # Automatic common toggle.
                self.usencom = True

            if hasattr(p,"enable_hr"): # Img2img doesn't have it.
                self.hr = p.enable_hr
                self.hr_w = (p.hr_resize_x if p.hr_resize_x > p.width else p.width * p.hr_scale)
                self.hr_h = (p.hr_resize_y if p.hr_resize_y > p.height else p.height * p.hr_scale)

            ##### region mode

            if self.indmaskmode:
                self, p ,breaks = inpaintmaskdealer(self, p, bratios, usebase, polymask, comprompt, comnegprompt)

            elif self.cells:
                self, p ,breaks = matrixdealer(self, p, aratios, bratios, mode, usebase, comprompt,comnegprompt)
    
            ##### calcmode 

            if calcmode == "Attention":
                self.handle = hook_forwards(self, p.sd_model.model.diffusion_model)
                shared.batch_cond_uncond = orig_batch_cond_uncond 
            else:
                self.handle = hook_forwards(self, p.sd_model.model.diffusion_model,remove = True)
                setuploras(self,p)
                if self.debug : print(p.prompt)
                regioner.reset()

            seps = KEYBRK

        elif "Prompt" in mode: #Prompt mode use both calcmode
            if not (KEYBRK in p.prompt.upper() or "AND" in p.prompt.upper() or KEYPROMPT in p.prompt.upper()):
                self.active = False
                unloader(self,p)
                return
            self.ex = "Ex" in mode
            self.modep = True
            if not usebase : bratios = "0"
            self.handle = hook_forwards(self, p.sd_model.model.diffusion_model)
            denoiserdealer(self)

            if calcmode == "Latent":
                seps = "AND"
                self.lpactive = True
            else:
                seps = KEYBRK

        self, p = commondealer(self, p, self.usecom, self.usencom)   #add commom prompt to all region
        self, p, breaks = anddealer(self, p , calcmode)                                 #replace BREAK to AND
        self, ppt, pnt = tokendealer(self, p, seps)                             #count tokens and calcrate target tokens
        self, p = thresholddealer(self, p, threshold)                          #set threshold
        self = bratioprompt(self, bratios)
                  

        print(f"pos tokens : {ppt}, neg tokens : {pnt}")
        if debug : debugall(self)

    def process_batch(self, p, active, debug, mode, aratios, bratios, usebase, usecom, usencom, calcmode,nchangeand, lnter, lnur, threshold, polymask,**kwargs):
        if active and self.modep:
            self = reset_pmasks(self)
        if active and calcmode =="Latent":
            setloradevice(self, p)
            lora_namer(self, p, lnter, lnur)

            if self.lora_applied: # SBM Don't override orig twice on batch calls.
                pass
            else:
                denoiserdealer(self)
                self.lora_applied = True
            

    # TODO: Should remove usebase, usecom, usencom - grabbed from self value.
    def postprocess_image(self, p, pp, active, debug, mode, aratios, bratios, usebase, usecom, usencom, calcmode, nchangeand, lnter, lnur, threshold, polymask):
        if not self.active:
            return p
        if self.usecom or self.cells or self.anded:
            p.prompt = self.orig_all_prompts[0]
            p.all_prompts[self.imgcount] = self.orig_all_prompts[self.imgcount]
        if self.usencom:
            p.negative_prompt = self.orig_all_negative_prompts[0]
            p.all_negative_prompts[self.imgcount] = self.orig_all_negative_prompts[self.imgcount]
        self.imgcount += 1
        return p

    def postprocess(self, p, processed, *args):
        if self.active : 
            with open(os.path.join(paths.data_path, "params.txt"), "w", encoding="utf8") as file:
                processedx = Processed(p, [], p.seed, "")
                file.write(processedx.infotext(p, 0))
        
        if self.modep:
            savepmasks(self, processed)

        if self.debug : debugall(self)

        unloader(self, p)


    def denoiser_callback(self, params: CFGDenoiserParams):
        denoiser_callback_s(self, params)

    def denoised_callback(self, params: CFGDenoisedParams):
        denoised_callback_s(self, params)


def unloader(self,p):
    if hasattr(self,"handle"):
        print("unloaded")
        hook_forwards(self, p.sd_model.model.diffusion_model, remove=True)
        del self.handle

    self.__init__()
    
    shared.batch_cond_uncond = orig_batch_cond_uncond

    unloadlorafowards(p)


def denoiserdealer(self):
    if self.calcmode =="Latent": # prompt mode use only denoiser callbacks
        if not hasattr(self,"dd_callbacks"):
            self.dd_callbacks = on_cfg_denoised(self.denoised_callback)
        shared.batch_cond_uncond = False

    if not hasattr(self,"dr_callbacks"):
        self.dr_callbacks = on_cfg_denoiser(self.denoiser_callback)


############################################################
##### prompts, tokens
def commondealer(self, p, usecom, usencom):
    def comadder(prompt):
        ppl = prompt.split(KEYBRK)
        for i in range(len(ppl)):
            if i == 0:
                continue
            ppl[i] = ppl[0] + ", " + ppl[i]
        ppl = ppl[1:]
        prompt = f"{KEYBRK} ".join(ppl)
        return prompt

    if usecom:
        self.prompt = p.prompt = comadder(p.prompt)
        for pr in p.all_prompts:
            self.all_prompts.append(comadder(pr))
        p.all_prompts = self.all_prompts

    if usencom:
        self.negative_prompt = p.negative_prompt = comadder(p.negative_prompt)
        for pr in p.all_negative_prompts:
            self.all_negative_prompts.append(comadder(pr))
        p.all_negative_prompts = self.all_negative_prompts
        
    return self, p


def anddealer(self, p, calcmode):
    breaks = p.prompt.count(KEYBRK)
    if calcmode != "Latent" : return self, p, breaks

    p.prompt = p.prompt.replace(KEYBRK, "AND")
    for i in lange(p.all_prompts):
        p.all_prompts[i] = p.all_prompts[i].replace(KEYBRK, "AND")
    p.negative_prompt = p.negative_prompt.replace(KEYBRK, "AND")
    for i in lange(p.all_negative_prompts):
        p.all_negative_prompts[i] = p.all_negative_prompts[i].replace(KEYBRK, "AND")
    self.divide = p.prompt.count("AND") + 1
    return self, p , breaks


def tokendealer(self, p, seps):
    ppl = p.prompt.split(seps)
    npl = p.negative_prompt.split(seps)
    targets =[p.split(",")[-1] for p in ppl[1:]]
    pt, nt, ppt, pnt, tt = [], [], [], [], []

    padd = 0
    for pp in ppl:
        tokens, tokensnum = shared.sd_model.cond_stage_model.tokenize_line(pp)
        pt.append([padd, tokensnum // TOKENS + 1 + padd])
        ppt.append(tokensnum)
        padd = tokensnum // TOKENS + 1 + padd

    if self.modep:
        for target in targets:
            ptokens, tokensnum = shared.sd_model.cond_stage_model.tokenize_line(ppl[0])
            ttokens, _ = shared.sd_model.cond_stage_model.tokenize_line(target)

            i = 1
            tlist = []
            while ttokens[0].tokens[i] != 49407:
                if ttokens[0].tokens[i] in ptokens[0].tokens:
                    tlist.append(ptokens[0].tokens.index(ttokens[0].tokens[i]))
                i += 1
            if tlist != [] : tt.append(tlist)

    paddp = padd
    padd = 0
    for np in npl:
        _, tokensnum = shared.sd_model.cond_stage_model.tokenize_line(np)
        nt.append([padd, tokensnum // TOKENS + 1 + padd])
        pnt.append(tokensnum)
        padd = tokensnum // TOKENS + 1 + padd
    self.eq = paddp == padd

    self.pt = pt
    self.nt = nt
    self.pe = tt

    return self, ppt, pnt


def thresholddealer(self, p ,threshold):
    if self.modep:
        threshold = threshold.split(",")
        while len(self.pe) >= len(threshold) + 1:
            threshold.append(threshold[0])
        self.th = [float(t) for t in threshold] * self.batch_size
        if self.debug :print ("threshold", self.th)
    return self, p

#####################################################
##### Save  and Load Settings

fcountbrk = lambda x: x.count(KEYBRK)
fint = lambda x: int(x)

# Json formatters.
fjstr = lambda x: x.strip()
#fjbool = lambda x: (x.upper() == "TRUE" or x.upper() == "T")
fjbool = lambda x: x # Json can store booleans reliably.

# (json_name, value_format, default)
# If default = none then will use current gradio value. 
PRESET_KEYS = [
("name",fjstr,"") , # Name is special, preset's key.
("mode", fjstr, None) ,
("ratios", fjstr, None) ,
("baseratios", fjstr, None) ,
("usebase", fjbool, None) ,
("usecom", fjbool, False) ,
("usencom", fjbool, False) ,
("calcmode", fjstr, "Attention") , # Generation mode.
("nchangeand", fjbool, False) ,
("lnter", fjstr, "0") ,
("lnur", fjstr, "0") ,
("threshold", fjstr, "0") ,
]


def savepresets(*settings):
    # NAME must come first.
    name = settings[0]
    path_root = modules.scripts.basedir()
    filepath = os.path.join(path_root, "scripts", "regional_prompter_presets.json")

    try:
        with open(filepath, mode='r', encoding="utf-8") as f:
            # presets = json.loads(json.load(f))
            presets = json.load(f)
            pr = {PRESET_KEYS[i][0]:settings[i] for i,_ in enumerate(PRESET_KEYS)}
            written = False
            # if name == "lastrun": # SBM We should check the preset is unique in any case.
            for i, preset in enumerate(presets):
                if name == preset["name"]:
                # if "lastrun" in preset["name"]:
                    presets[i] = pr
                    written = True
            if not written:
                presets.append(pr)
        with open(filepath, mode='w', encoding="utf-8") as f:
            # json.dump(json.dumps(presets), f, indent = 2)
            json.dump(presets, f, indent = 2)
    except Exception as e:
        print(e)

    presets = loadpresets(filepath)
    return gr.update(choices=[pr["name"] for pr in presets])


def loadpresets(filepath):
    presets = []
    try:
        with open(filepath, encoding="utf-8") as f:
            # presets = json.loads(json.load(f))
            presets = json.load(f)
    except OSError as e:
        print("Init / preset error.")
        presets = initpresets(filepath)
    except TypeError:
        print("Corrupted file, resetting.")
        presets = initpresets(filepath)
        
    return presets


def initpresets(filepath):
    lpr = PRESETS
    # if not os.path.isfile(filepath):
    try:
        with open(filepath, mode='w', encoding="utf-8") as f:
            lprj = []
            for pr in lpr:
                prj = {PRESET_KEYS[i][0]:pr[i] for i,_ in enumerate(PRESET_KEYS)} 
                lprj.append(prj)
            #json.dump(json.dumps(lprj), f, indent = 2)
            json.dump(lprj, f, indent = 2)
            return lprj
    except Exception as e:
        return None

def debugall(self):
    print(f"mode : {self.calcmode}\ndivide : {self.mode}\nusebase : {self.usebase}")
    print(f"base ratios : {self.bratios}\nusecommon : {self.usecom}\nusenegcom : {self.usencom}\nuse 2D : {self.cells}")
    print(f"divide : {self.divide}\neq : {self.eq}\n")
    print(f"ratios : {self.aratios}\n")
    print(f"prompt : {self.pe}\n")


def bratioprompt(self, bratios):
    bratios = bratios.split(",")
    bratios = [float(b) for b in bratios]
    while len(self.pe) >= len(bratios) + 1:
        bratios.append(bratios[0])
    self.bratios = bratios
    return self