import os.path
from importlib import reload
from pprint import pprint
import gradio as gr
import modules.ui
import modules # SBM Apparently, basedir only works when accessed directly.
from modules import paths, scripts, shared, extra_networks
from modules.processing import Processed
from modules.script_callbacks import (on_ui_settings,
                                      CFGDenoisedParams, CFGDenoiserParams, on_cfg_denoised, on_cfg_denoiser)
import scripts.attention
import scripts.latent
import scripts.regions
reload(scripts.regions) # update without restarting web-ui.bat
reload(scripts.attention)
reload(scripts.latent)
import json  # Presets.
from json.decoder import JSONDecodeError
from scripts.attention import (TOKENS, hook_forwards, reset_pmasks, savepmasks)
from scripts.latent import (denoised_callback_s, denoiser_callback_s, lora_namer,
                            restoremodel, setloradevice, setuploras, unloadlorafowards)
from scripts.regions import (MAXCOLREG, IDIM, KEYBRK, KEYBASE, KEYCOMM, KEYPROMPT,
                             create_canvas, draw_region, #detect_mask, detect_polygons,  
                             draw_image, save_mask, load_mask,
                             floatdef, inpaintmaskdealer, makeimgtmp, matrixdealer)

FLJSON = "regional_prompter_presets.json"
# Modules.basedir points to extension's dir. script_path or scripts.basedir points to root.
PTPRESET = modules.scripts.basedir()
# PTPRESET = paths.script_path
# Original, fallback.
PTPRESETALT = os.path.join(paths.script_path, "scripts")
# OVERRIDE monkey patch: gradio.Image.preprocess.
# Shallowly fixes a certain bug which causes a sketch to deteriorate to image.
# class Image_Fix(gr.Image): 
#     def preprocess(self,x):
#         """Change str (image64) to dict with a null mask.
#
#         Ugh.
#         """
#         if self.tool == "sketch" and self.source in ["upload", "webcam"]:
#             if isinstance(x, str):
#                 x = {"image":x, "mask": None}
#         return super().preprocess(x)

# gr.Image = Image_Fix

def lange(l):
    return range(len(l))

orig_batch_cond_uncond = shared.batch_cond_uncond

PRESETSDEF =[
    ["Vertical-3", "Vertical",'1,1,1',"",False,False,False,"Attention",False,"0","0"],
    ["Horizontal-3", "Horizontal",'1,1,1',"",False,False,False,"Attention",False,"0","0"],
    ["Horizontal-7", "Horizontal",'1,1,1,1,1,1,1',"0.2",True,False,False,"Attention",False,"0","0"],
    ["Twod-2-1", "Horizontal",'1,2,3;1,1',"0.2",False,False,False,"Attention",False,"0","0"],
]

ATTNSCALE = 8 # Initial image compression in attention layers.

def ui_tab(mode, submode):
    """Structures components for mode tab.
    
    Semi harcoded but it's clearer this way.
    """
    vret = None
    if mode == "Matrix":
        with gr.Row():
            mmode = gr.Radio(label="Split mode", choices=submode, value="Horizontal", type="value", interactive=True)
            ratios = gr.Textbox(label="Divide Ratio",lines=1,value="1,1",interactive=True,elem_id="RP_divide_ratio",visible=True)
        with gr.Row():
            with gr.Column():
                maketemp = gr.Button(value="visualize and make template")
                template = gr.Textbox(label="template",interactive=True,visible=True)
            with gr.Column():
                areasimg = gr.Image(type="pil", show_label  = False).style(height=256,width=256)
                    
        # Need to add maketemp function based on base / common checks.
        vret = [mmode, ratios, maketemp, template, areasimg]
    elif mode == "Mask":
        with gr.Row(): # Creep: Placeholder, should probably make this invisible.
            xmode = gr.Radio(label="Mask mode", choices=submode, value="Mask", type="value", interactive=True)
        with gr.Row(): # CREEP: Css magic to make the canvas bigger? I think it's in style.css: #img2maskimg -> height.
            polymask = gr.Image(label = "Do not upload here until bugfix",elem_id="polymask",
                                source = "upload", mirror_webcam = False, type = "numpy", tool = "sketch")#.style(height=480)
        with gr.Row():
            with gr.Column():
                num = gr.Slider(label="Region", minimum=-1, maximum=MAXCOLREG, step=1, value=1)
                canvas_width = gr.Slider(label="Canvas Width", minimum=64, maximum=2048, value=512, step=8)
                canvas_height = gr.Slider(label="Canvas Height", minimum=64, maximum=2048, value=512, step=8)
                btn = gr.Button(value = "Draw region + show mask")
                # btn2 = gr.Button(value = "Display mask") # Not needed.
                cbtn = gr.Button(value="Create mask area")
            with gr.Column():
                showmask = gr.Image(label = "Mask", shape=(IDIM, IDIM))
                # CONT: Awaiting fix for https://github.com/gradio-app/gradio/issues/4088.
                uploadmask = gr.Image(label="Upload mask here cus gradio",source = "upload", type = "numpy")
        # btn.click(detect_polygons, inputs = [polymask,num], outputs = [polymask,num])
        btn.click(draw_region, inputs = [polymask, num], outputs = [polymask, num, showmask])
        # btn2.click(detect_mask, inputs = [polymask,num], outputs = [showmask])
        cbtn.click(fn=create_canvas, inputs=[canvas_height, canvas_width], outputs=[polymask])
        uploadmask.upload(fn = draw_image, inputs = [uploadmask], outputs = [polymask, uploadmask, showmask])
        
        vret = [xmode, polymask, num, canvas_width, canvas_height, btn, cbtn, showmask, uploadmask]
    elif mode == "Prompt":
        with gr.Row():
            pmode = gr.Radio(label="Prompt mode", choices=submode, value="Prompt", type="value", interactive=True)
            threshold = gr.Textbox(label = "threshold", value = 0.4, interactive=True)
        
        vret = [pmode, threshold]

    return vret
            
# modes, submodes. Order must be maintained so dict is inadequate. Must have submode for component consistency.
RPMODES = [
("Matrix", ("Horizontal","Vertical")),
("Mask", ("Mask",)),
("Prompt", ("Prompt", "Prompt-Ex")),
]
fgrprop = lambda x: {"label": x, "id": "t" + x, "elem_id": "RP_" + x}

def mode2tabs(mode):
    """Converts mode (in preset) to gradio tab + submodes.
    
    I dunno if it's possible to nest components or make them optional (probably not),
    so this is the best we can do.
    """
    vret = ["Nope"] + [None] * len(RPMODES)
    for (i,(k,v)) in enumerate(RPMODES):
        if mode in v:
            vret[0] = k
            vret[i + 1] = mode
    return vret
    
def tabs2mode(tab, *submode):
    """Converts ui tab + submode list to a single value mode.
    
    Picks current submode based on tab, nothing clever. Submodes must be unique.
    """
    for (i,(k,_)) in enumerate(RPMODES):
        if tab == k:
            return submode[i]
    return "Nope"
    
def expand_components(l):
    """Converts json preset to component format.
    
    Assumes mode is the first value in list.
    """
    l = list(l) # Tuples cannot be altered.
    tabs = mode2tabs(l[0])
    return tabs + l[1:]

def compress_components(l):
    """Converts component values to preset format.
    
    Assumes tab + submodes are the first values in list.
    """
    l = list(l)
    mode = tabs2mode(*l[:len(RPMODES) + 1])
    return [mode] + l[len(RPMODES) + 1:]
    
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
        filepath = os.path.join(PTPRESET, FLJSON)

        presets = []

        presets = loadpresets(filepath)
        presets = LPRESET.update(presets)

        with gr.Accordion("Regional Prompter", open=False, elem_id="RP_main"):
            with gr.Row():
                active = gr.Checkbox(value=False, label="Active",interactive=True,elem_id="RP_active")
            with gr.Row():
                # mode = gr.Radio(label="Divide mode", choices=["Horizontal", "Vertical","Mask","Prompt","Prompt-Ex"], value="Horizontal",  type="value", interactive=True)
                calcmode = gr.Radio(label="Generation mode", choices=["Attention", "Latent"], value="Attention",  type="value", interactive=True)
            with gr.Row(visible=True):
                # ratios = gr.Textbox(label="Divide Ratio",lines=1,value="1,1",interactive=True,elem_id="RP_divide_ratio",visible=True)
                baseratios = gr.Textbox(label="Base Ratio", lines=1,value="0.2",interactive=True,  elem_id="RP_base_ratio", visible=True)
            with gr.Row():
                usebase = gr.Checkbox(value=False, label="Use base prompt",interactive=True, elem_id="RP_usebase")
                usecom = gr.Checkbox(value=False, label="Use common prompt",interactive=True,elem_id="RP_usecommon")
                usencom = gr.Checkbox(value=False, label="Use common negative prompt",interactive=True,elem_id="RP_usecommon")
            
            # Tabbed modes.
            with gr.Tabs(elem_id="RP_mode"):
                rp_selected_tab = gr.State("Matrix") # State component to document current tab for gen.
                # ltabs = []
                ltabp = []
                for (i, (md,smd)) in enumerate(RPMODES):
                    with gr.TabItem(**fgrprop(md)) as tab: # Tabs with a formatted id.
                        # ltabs.append(tab)
                        ltabp.append(ui_tab(md, smd))
                    # Tab switch tags state component.
                    tab.select(fn = lambda tabnum = i: RPMODES[tabnum][0], inputs=[], outputs=[rp_selected_tab])
            
            # Hardcode expansion back to components for any specific events.
            (mmode, ratios, maketemp, template, areasimg) = ltabp[0]
            (xmode, polymask, num, canvas_width, canvas_height, btn, cbtn, showmask, uploadmask) = ltabp[1]
            (pmode, threshold) = ltabp[2]
            
            with gr.Accordion("Presets",open = False):
                with gr.Row():
                    availablepresets = gr.Dropdown(label="Presets", choices=presets, type="index")
                    applypresets = gr.Button(value="Apply Presets",variant='primary',elem_id="RP_applysetting")
                with gr.Row():
                    presetname = gr.Textbox(label="Preset Name",lines=1,value="",interactive=True,elem_id="RP_preset_name",visible=True)
                    savesets = gr.Button(value="Save to Presets",variant='primary',elem_id="RP_savesetting")
            with gr.Row():
                nchangeand = gr.Checkbox(value=False, label="disable convert 'AND' to 'BREAK'", interactive=True, elem_id="RP_ncand")
                debug = gr.Checkbox(value=False, label="debug", interactive=True, elem_id="RP_debug")
                lnter = gr.Textbox(label="LoRA in negative textencoder",value="0",interactive=True,elem_id="RP_ne_tenc_ratio",visible=True)
                lnur = gr.Textbox(label="LoRA in negative U-net",value="0",interactive=True,elem_id="RP_ne_unet_ratio",visible=True)
            settings = [rp_selected_tab, mmode, xmode, pmode, ratios, baseratios, usebase, usecom, usencom, calcmode, nchangeand, lnter, lnur, threshold, polymask]
        
        self.infotext_fields = [
                (active, "RP Active"),
                # (mode, "RP Divide mode"),
                (rp_selected_tab, "RP Divide mode"),
                (mmode, "RP Matrix submode"),
                (xmode, "RP Mask submode"),
                (pmode, "RP Prompt submode"),
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

        def setpreset(select, *settings):
            """Load preset from list.
            
            SBM: The only way I know how to get the old values in gradio,
            is to pass them all as input.
            Tab mode converts ui to single value.
            """
            # Need to swap all masked images to the source,
            # getting "valueerror: cannot process this value as image".
            # Gradio bug in components.postprocess, most likely.
            settings = [s["image"] if (isinstance(s,dict) and "image" in s) else s for s in settings]
            presets = loadpresets(filepath)
            preset = presets[select]
            preset = loadblob(preset)
            preset = [fmt(preset.get(k, vdef)) for (k,fmt,vdef) in PRESET_KEYS]
            preset = preset[1:] # Remove name.
            preset = expand_components(preset)
            # Change nulls to original value.
            preset = [settings[i] if p is None else p for (i,p) in enumerate(preset)]
            # return [gr.update(value = pr) for pr in preset] # SBM Why update? Shouldn't regular return do the job? 
            return preset

        maketemp.click(fn=makeimgtmp, inputs =[ratios,mmode,usecom,usebase],outputs = [areasimg,template])
        applypresets.click(fn=setpreset, inputs = [availablepresets, *settings], outputs=settings)
        savesets.click(fn=savepresets, inputs = [presetname,*settings],outputs=availablepresets)
        
        return [active, debug, rp_selected_tab, mmode, xmode, pmode, ratios, baseratios,
                usebase, usecom, usencom, calcmode, nchangeand, lnter, lnur, threshold, polymask]

    def process(self, p, active, debug, rp_selected_tab, mmode, xmode, pmode, aratios, bratios,
                usebase, usecom, usencom, calcmode, nchangeand, lnter, lnur, threshold, polymask):
        if not active:
            unloader(self,p)
            return p

        p.extra_generation_params.update({
            "RP Active":active,
            # "RP Divide mode":mode,
            "RP Divide mode": rp_selected_tab,
            "RP Matrix submode": mmode,
            "RP Mask submode": xmode,
            "RP Prompt submode": pmode,
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

        savepresets("lastrun",rp_selected_tab, mmode, xmode, pmode, aratios,bratios,
                     usebase, usecom, usencom, calcmode, nchangeand, lnter, lnur, threshold, polymask)

        self.__init__()

        if type(p.prompt) == list:p.prompt = p.promot[0]

        self.active = True
        self.mode = tabs2mode(rp_selected_tab, mmode, xmode, pmode)
        self.calcmode = calcmode
        self.debug = debug
        self.usebase = usebase
        self.usecom = usecom
        self.usencom = usencom
        self.w = p.width
        self.h = p.height
        self.batch_size = p.batch_size
        self.prompt = p.prompt
        self.all_prompts = p.all_prompts.copy()
        self.all_negative_prompts = p.all_negative_prompts.copy()

        comprompt = comnegprompt = None

        # SBM ddim / plms detection.
        self.isvanilla = p.sampler_name in ["DDIM", "PLMS", "UniPC"]

        if self.h % ATTNSCALE != 0 or self.w % ATTNSCALE != 0:
            # Testing shows a round down occurs in model.
            print("Warning: Nonstandard height / width.")
            self.h = self.h - self.h % ATTNSCALE
            self.w = self.w - self.w % ATTNSCALE

        if hasattr(p,"enable_hr"): # Img2img doesn't have it.
            self.hr = p.enable_hr
            self.hr_w = (p.hr_resize_x if p.hr_resize_x > p.width else p.width * p.hr_scale)
            self.hr_h = (p.hr_resize_y if p.hr_resize_y > p.height else p.height * p.hr_scale)
            if self.hr_h % ATTNSCALE != 0 or self.hr_w % ATTNSCALE != 0:
                # Testing shows a round down occurs in model.
                print("Warning: Nonstandard height / width for ulscaled size")
                self.hr_h = self.hr_h - self.hr_h % ATTNSCALE
                self.hr_w = self.hr_w - self.hr_w % ATTNSCALE

        self, p = flagfromkeys(self, p)

        self.indmaskmode = (self.mode == "Mask")

        if not nchangeand and "AND" in p.prompt.upper():
            p.prompt = p.prompt.replace("AND",KEYBRK)
            for i in lange(p.all_prompts):
                p.all_prompts[i] = p.all_prompts[i].replace("AND",KEYBRK)
            self.anded = True
            

        if "Prompt" not in self.mode: # skip region assign in prompt mode
            self.cells = not "Mask" in self.mode

            #convert BREAK to ADDCOL/ADDROW
            if KEYBRK in p.prompt and not "Mask" in self.mode:
                p = keyconverter(aratios, self.mode, usecom, usebase, p)

            ##### region mode

            if self.indmaskmode:
                self, p = inpaintmaskdealer(self, p, bratios, usebase, polymask, comprompt, comnegprompt)

            elif self.cells:
                self, p = matrixdealer(self, p, aratios, bratios, self.mode, usebase, comprompt,comnegprompt)
    
            ##### calcmode 

            if calcmode == "Attention":
                self.handle = hook_forwards(self, p.sd_model.model.diffusion_model)
                shared.batch_cond_uncond = orig_batch_cond_uncond
                seps = KEYBRK 
            else:
                self.handle = hook_forwards(self, p.sd_model.model.diffusion_model,remove = True)
                setuploras(self,p)
                if self.debug : print(p.prompt)
                seps = "AND"
                # SBM It is vital to use local activation because callback registration is permanent,
                # and there are multiple script instances (txt2img / img2img). 
                self.lactive = True

            # seps = KEYBRK # SBM No longer is keybrk applied first.

        elif "Prompt" in self.mode: #Prompt mode use both calcmode
            if not (KEYBRK in p.prompt.upper() or "AND" in p.prompt.upper() or KEYPROMPT in p.prompt.upper()):
                self.active = False
                unloader(self,p)
                return
            self.ex = "Ex" in self.mode
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
        self, p = anddealer(self, p , calcmode)                                 #replace BREAK to AND
        self = tokendealer(self, p, seps)                             #count tokens and calcrate target tokens
        self, p = thresholddealer(self, p, threshold)                          #set threshold
        self = bratioprompt(self, bratios)
        p = hrdealer(p)

        print(f"pos tokens : {self.ppt}, neg tokens : {self.pnt}")
        if debug : debugall(self)

    def before_process_batch(self, p, *args, **kwargs):
        self.current_prompts = kwargs["prompts"].copy()
        p.disable_extra_networks = False

    def process_batch(self, p, active, debug, rp_selected_tab, mmode, xmode, pmode, aratios, bratios,
                      usebase, usecom, usencom, calcmode,nchangeand, lnter, lnur, threshold, polymask,**kwargs):
        # print(kwargs["prompts"])
        if active:
            # SBM Before_process_batch was added in feb-mar, adding fallback.
            if not hasattr(self,"current_prompts"):
                self.current_prompts = kwargs["prompts"].copy()
            p.all_prompts[p.iteration * p.batch_size:(p.iteration + 1) * p.batch_size] = self.all_prompts[p.iteration * p.batch_size:(p.iteration + 1) * p.batch_size]
            p.all_negative_prompts[p.iteration * p.batch_size:(p.iteration + 1) * p.batch_size] = self.all_negative_prompts[p.iteration * p.batch_size:(p.iteration + 1) * p.batch_size]

            if self.modep:
                self = reset_pmasks(self)
            if calcmode =="Latent":
                setloradevice(self) #change lora device cup to gup and restore model in new web-ui lora method
                lora_namer(self, p, lnter, lnur)

                if self.lora_applied: # SBM Don't override orig twice on batch calls.
                    pass
                else:
                    restoremodel(p)
                    denoiserdealer(self)
                    self.lora_applied = True
                #escape reload loras in hires-fix
                p.disable_extra_networks = True

    # TODO: Should remove usebase, usecom, usencom - grabbed from self value.
    def postprocess_image(self, p, pp, *args, **kwargs):
        if not self.active:
            return p
        # SBM I'm not sure if there's a prompt increment that isn't working, or that it must be done manually,
        # but either way this will force p.prompt to receive the next value rather than revert to orig in batchcount.
        # if self.imgcount + 1 < len(self.orig_all_prompts):
        #     p.prompt = p.all_prompts[self.imgcount + 1]
        #     p.negative_prompt = p.all_negative_prompts[self.imgcount + 1]
        # else:
        #     if self.usecom or self.cells or self.anded:
        #         p.prompt = self.orig_all_prompts[0]
        #         p.all_prompts[self.imgcount] = self.orig_all_prompts[self.imgcount]
        #     if self.usencom:
        #         p.negative_prompt = self.orig_all_negative_prompts[0]
        #         p.all_negative_prompts[self.imgcount] = self.orig_all_negative_prompts[self.imgcount]
        # self.imgcount += 1
        print("postprocess_image : ",self.imgcount,p.iteration,p.prompt,p.all_prompts)
        return p

    def postprocess(self, p, processed, *args):
        if self.active : 
            with open(os.path.join(paths.data_path, "params.txt"), "w", encoding="utf8") as file:
                processedx = Processed(p, [], p.seed, "")
                file.write(processedx.infotext(p, 0))
        
        if self.modep and not fseti("hidepmask"):
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
    all_prompts = []
    all_negative_prompts = []
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
            all_prompts.append(comadder(pr))
        p.all_prompts = all_prompts

    if usencom:
        self.negative_prompt = p.negative_prompt = comadder(p.negative_prompt)
        for pr in p.all_negative_prompts:
            all_negative_prompts.append(comadder(pr))
        p.all_negative_prompts = all_negative_prompts
        
    return self, p

def hrdealer(p):
    p.hr_prompt = p.prompt
    p.hr_negative_prompt = p.negative_prompt
    p.all_hr_prompts = p.all_prompts
    p.all_hr_negative_prompts = p.all_negative_prompts
    return p

def anddealer(self, p, calcmode):
    self.divide = p.prompt.count(KEYBRK)
    if calcmode != "Latent" : return self, p

    p.prompt = p.prompt.replace(KEYBRK, "AND")
    for i in lange(p.all_prompts):
        p.all_prompts[i] = p.all_prompts[i].replace(KEYBRK, "AND")
    p.negative_prompt = p.negative_prompt.replace(KEYBRK, "AND")
    for i in lange(p.all_negative_prompts):
        p.all_negative_prompts[i] = p.all_negative_prompts[i].replace(KEYBRK, "AND")
    self.divide = p.prompt.count("AND") + 1
    return self, p


def tokendealer(self, p, seps):
    text, _ = extra_networks.parse_prompt(p.all_prompts[0]) # SBM From update_token_counter.
    ppl = text.split(seps)
    npl = p.all_negative_prompts[0].split(seps)
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
                for (j, maintok) in enumerate(ptokens): # SBM Long prompt.
                    if ttokens[0].tokens[i] in maintok.tokens:
                        tlist.append(maintok.tokens.index(ttokens[0].tokens[i]) + 75 * j)
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
    self.ppt = ppt
    self.pnt = pnt

    return self


def thresholddealer(self, p ,threshold):
    if self.modep:
        threshold = threshold.split(",")
        while len(self.pe) >= len(threshold) + 1:
            threshold.append(threshold[0])
        self.th = [floatdef(t, 0.4) for t in threshold] * self.batch_size
        if self.debug :print ("threshold", self.th)
    return self, p


def bratioprompt(self, bratios):
    if not self.modep: return self
    bratios = bratios.split(",")
    bratios = [floatdef(b, 0) for b in bratios]
    while len(self.pe) >= len(bratios) + 1:
        bratios.append(bratios[0])
    self.bratios = bratios
    return self
#####################################################
##### Presets - Save  and Load Settings

fimgpt = lambda flnm, fext, *dirparts: os.path.join(*dirparts, flnm + fext)

class PresetList():
    """Preset list must be the same object throughout its lifetime, otherwise updates will err.

    See gradio issue #4210 for details.
    """
    def __init__(self):
        self.lpr = []
    
    def update(self, newpr):
        """Replace all values, return the reference.
        
        Will convert dicts to the names only.
        Might be more efficient to add the new names only, but meh.
        """
        if len(newpr) > 0 and isinstance(newpr[0],dict):
            newpr = [pr["name"] for pr in newpr] 
        self.lpr.clear()
        self.lpr.extend(newpr)
        return self.lpr
        
    def get(self):
        return self.lpr

class JsonMask():
    """Mask saved as image with some editing work.
    
    """
    blobdir = "regional_masks"
    ext = ".png"
    
    def __init__(self, img):
        self.img = img
    
    def makepath(self, name):
        pt = fimgpt(name, self.ext, PTPRESET, self.blobdir)
        os.makedirs(os.path.dirname(pt), exist_ok = True)
        return pt
    
    def save(self, name, preset = None):
        """Save image to subdir.
        
        Only saved when in mask mode - Hardcoded, don't have a better idea atm.
        """
        if (preset is None) or (preset[1] == "Mask"): # Check mode.  
            save_mask(self.img, self.makepath(name))
            return name
        return None
    
    def load(self, name, preset = None):
        """Load image from subdir (no editing, that comes later).
        
        Prefer to use the given key, rather than name. SBM CONT: Load / save in dict mode? Debugging needed.
        """
        if name is None or self.img is None:
            return None
        return load_mask(self.makepath(self.img))

LPRESET = PresetList()

fcountbrk = lambda x: x.count(KEYBRK)
fint = lambda x: int(x)

# Json formatters.
fjstr = lambda x: x.strip()
#fjbool = lambda x: (x.upper() == "TRUE" or x.upper() == "T")
fjbool = lambda x: x # Json can store booleans reliably.
fjmask = lambda x: draw_image(x, inddict = False)[0] # Ignore mask reset value. 

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
("polymask", fjmask, "") , # Mask has special corrections and logging.
]
# (json_name,blob_class)
# Handles save + lazy load of blob data outside of presets.
BLOB_KEYS = {
"polymask": JsonMask
}

def saveblob(preset):
    """Preset variables saved externally (blob).
    
    Returns modified list containing the refernces instead of data.
    Currently, this includes polymask, which is saved as an image,
    with a filename = preset.
    A blob class should contain a save method which returns the reference. 
    """
    preset = list(preset) # Tuples don't have copy.
    for (i,(vkey,vfun,vdef)) in enumerate(PRESET_KEYS):
        if vkey in BLOB_KEYS:
            # Func should accept raw form and convert it to a class.
            x = BLOB_KEYS[vkey](preset[i])
            # Class should have a save func given identifier, returning an access key.
            x = x.save(preset[0], preset)
            # Update the preset.
            preset[i] = x
    return preset

def loadblob(preset):
    """Load blob presets based on key.
    
    Returns modified list containing the refernces instead of 
    Currently, this includes polymask, which is saved as an image,
    with a filename = preset.
    A blob class should contain a load method which retrieves the data based on reference. 
    """
    for (vkey,vval) in BLOB_KEYS.items():
        # Func should accept refrence form and convert it to a class.
        x = vval(preset.get(vkey))
        # Class should have a load func given identifier, returning data.
        x = x.load(preset["name"], preset)
        # Update the preset.
        preset[vkey] = x
    return preset

def savepresets(*settings):
    # NAME must come first.
    name = settings[0]
    settings = [name] + compress_components(settings[1:])
    settings = saveblob(settings)
    
    # path_root = modules.scripts.basedir()
    # filepath = os.path.join(path_root, "scripts", "regional_prompter_presets.json")
    filepath = os.path.join(PTPRESET, FLJSON)

    try:
        with open(filepath, mode='r', encoding="utf-8") as f:
            # presets = json.loads(json.load(f))
            presets = json.load(f)
            pr = {PRESET_KEYS[i][0]:settings[i] for i,_ in enumerate(PRESET_KEYS)}
            # SBM Ordereddict might be better than list, quick search.
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
    presets = LPRESET.update(presets)
    return gr.update(choices=presets)

def presetfallback():
    """Swaps main json dir to alt if exists, attempts reload.
    
    """
    global PTPRESET
    global PTPRESETALT
    
    if PTPRESETALT is not None:
        print("Unknown preset error, fallback.")
        PTPRESET = PTPRESETALT
        PTPRESETALT = None
        return loadpresets(PTPRESET)
    else: # Already attempted swap.
        print("Presets could not be loaded.") 
        return None

def loadpresets(filepath):
    presets = []
    try:
        with open(filepath, encoding="utf-8") as f:
            # presets = json.loads(json.load(f))
            presets = json.load(f)
            # presets = loadblob(presets) # DO NOT load all blobs - that's the point.
    except OSError as e:
        print("Init / preset error.")
        presets = initpresets(filepath)
    except TypeError:
        print("Corrupted preset file, resetting.")
        presets = initpresets(filepath)
    except JSONDecodeError:
        print("Preset file could not be decoded.")
        presets = initpresets(filepath)
    return presets

def initpresets(filepath):
    lpr = PRESETSDEF
    # if not os.path.isfile(filepath):
    try:
        with open(filepath, mode='w', encoding="utf-8") as f:
            lprj = []
            for pr in lpr:
                plen = min(len(PRESET_KEYS), len(pr)) # Future setting additions ignored.
                prj = {PRESET_KEYS[i][0]:pr[i] for i in range(plen)}
                lprj.append(prj)
            #json.dump(json.dumps(lprj), f, indent = 2)
            json.dump(lprj, f, indent = 2)
            return lprj
    except Exception as e:
        return presetfallback()

#####################################################
##### Global settings

EXTKEY = "regprp"
EXTNAME = "Regional Prompter"
# (id, label, type, extra_parms)
EXTSETS = [
("debug", "(PLACEHOLDER, USE THE ONE IN 2IMG) Enable debug mode", "check", dict()),
("hidepmask", "Hide subprompt masks in prompt mode", "check", dict()),

]
# Dynamically constructed list of default values, because shared doesn't allocate a value automatically.
# (id: def)
DEXTSETV = dict()
fseti = lambda x: shared.opts.data.get(EXTKEY + "_" + x, DEXTSETV[x])

class Setting_Component():
    """Creates gradio components with some standard req values.
    
    All must supply an id (used in code), label, component type. 
    Default value and specific type settings can be overridden. 
    """
    section = (EXTKEY, EXTNAME)
    def __init__(self, cid, clabel, ctyp, vdef = None, **kwargs):
        self.cid = EXTKEY + "_" + cid
        self.clabel = clabel
        self.ctyp = ctyp
        method = getattr(self, self.ctyp)
        method(**kwargs)
        if vdef is not None:
            self.vdef = vdef
        
    def get(self):
        """Get formatted setting.
        
        Input for shared.opts.add_option().
        """
        if self.ctyp == "textb":
            return (self.cid, shared.OptionInfo(self.vdef, self.clabel, section = self.section))
        return (self.cid, shared.OptionInfo(self.vdef, self.clabel,
                                            self.ccomp, self.cparms, section = self.section))
    
    def textb(self, **kwargs):
        """Textbox unusually requires no component.
        
        """
        self.ccomp = gr.Textbox
        self.vdef = ""
        self.cparms = {}
        self.cparms.update(kwargs)
    
    def check(self, **kwargs):
        self.ccomp = gr.Checkbox
        self.vdef = False
        self.cparms = {"interactive": True}
        self.cparms.update(kwargs)
        
    def slider(self, **kwargs):
        self.ccomp = gr.Slider
        self.vdef = 0
        self.cparms = {"minimum": 1, "maximum": 10, "step": 1}
        self.cparms.update(kwargs)

def ext_on_ui_settings():
    for (cid, clabel, ctyp, kwargs) in EXTSETS:
        comp = Setting_Component(cid, clabel, ctyp, **kwargs)
        opt = comp.get()
        shared.opts.add_option(*opt)
        DEXTSETV[cid] = comp.vdef

def debugall(self):
    print(f"mode : {self.calcmode}\ndivide : {self.mode}\nusebase : {self.usebase}")
    print(f"base ratios : {self.bratios}\nusecommon : {self.usecom}\nusenegcom : {self.usencom}\nuse 2D : {self.cells}")
    print(f"divide : {self.divide}\neq : {self.eq}\n")
    print(f"tokens : {self.ppt},{self.pnt},{self.pt},{self.nt}\n")
    print(f"ratios : {self.aratios}\n")
    print(f"prompt : {self.pe}\n")

def flagfromkeys(self, p):
    '''
    detect COMM/BASE keys and set flags
    '''
    if KEYCOMM in p.prompt:
        self.usecom = True
    
    if KEYCOMM in p.negative_prompt:
        self.usencom = True
    
    if KEYBASE in p.prompt:
        self.usebase = True

        
    if KEYPROMPT in p.prompt.upper():
        self.mode = "Prompt"
        p.replace(KEYPROMPT,KEYBRK)

    return self, p

def keyconverter(aratios,mode,usecom,usebase,p):
    '''convert BREAKS to ADDCOMM/ADDBASE/ADDCOL/ADDROW'''
    keychanger = makeimgtmp(aratios,mode,usecom,usebase,inprocess = True)
    keychanger = keychanger[:-1]
    #print(keychanger,p.prompt)
    for change in keychanger:
        if change == KEYCOMM and KEYCOMM in p.prompt: continue
        if change == KEYBASE and KEYBASE in p.prompt: continue
        p.prompt= p.prompt.replace(KEYBRK,change,1)

    return p

on_ui_settings(ext_on_ui_settings)
