from random import choices
from typing import Union
from matplotlib.style import available
import PIL
import inspect
import random
import copy
from regex import R
import torch
import csv
import math
import gradio as gr
import numpy as np
import os.path
from pprint import pprint
import modules.ui
import ldm.modules.attention as atm
from modules import shared,scripts,extra_networks,devices,paths
from modules.processing import Processed
from modules.script_callbacks import CFGDenoisedParams, on_cfg_denoised ,CFGDenoiserParams,on_cfg_denoiser


#'"name","mode","divide ratios,"use base","baseratios","usecom","usencom",\n'
"""
SBM mod: Two dimensional regions (of variable size, NOT a matrix).
- Adds keywords ADDROW, ADDCOL and respective delimiters for aratios.
- A/bratios become list dicts: Inner dict of cols (varying length list) + start/end + number of breaks,
  outer layer is rows list.
  First value in each row is the row's ratio, the rest are col ratios.
  This fits prompts going left -> right, top -> down. 
- Unrelated BREAKS are counted per cell, and later extracted as multiple context indices.
- Each layer is cut up by both row + col ratios.
- Style improvements: Created classes for rows + cells and functions for some of the splitting.
- Base prompt overhaul: Added keyword ADDBASE, when present will trigger "use_base" automatically;
  base is excluded from the main prompt for dim calcs; returned to start before hook (+ base break count);
  during hook, context index skips base break count + 1. Rest is applied normally.
- To specify cols first, use "vertical" mode. eg 1st col:2 rows, 2nd col:1 row.
  In effect, this merely reverses the order of iteration for every row/col loop and whatnot.
"""
def lange(l):
    return range(len(l))

orig_lora_forward = None
orig_lora_apply_weights = None
orig_lora_Linear_forward = None
orig_lora_Conv2d_forward = None
lactive = False
labug =False

PRESETS =[
    ["Vertical-3", "Vertical",'"1,1,1"',"","False","False","False"],
    ["Horizontal-3", "Horizontal",'"1,1,1"',"","False","False","False"],
    ["Horizontal-7", "Horizontal",'"1,1,1,1,1,1,1"',"0.2","True","False","False"],
]
# SBM Keywords and delimiters for region breaks, following matlab rules.
# BREAK keyword is now passed through,  
KEYROW = "ADDROW"
KEYCOL = "ADDCOL"
KEYBASE = "ADDBASE"
KEYCOMM = "ADDCOMM"
KEYBRK = "BREAK"
DELIMROW = ";"
DELIMCOL = ","
#MATMODE = "Matrix"
TOKENSCON = 77
TOKENS = 75
fidentity = lambda x: x
fcountbrk = lambda x: x.count(KEYBRK)
ffloat = lambda x: float(x)
fint = lambda x: int(x)
fspace = lambda x: " {} ".format(x)

class RegionCell():
    """Cell used to split a layer to single prompts."""
    def __init__(self, st, ed, base, breaks):
        """Range with start and end values, base weight and breaks count for context splitting."""
        self.st = st # Range for the cell (cols only).
        self.ed = ed
        self.base = base # How much of the base prompt is applied (difference).
        self.breaks = breaks # How many unrelated breaks the prompt contains.
        
class RegionRow():
    """Row containing cell refs and its own ratio range."""
    def __init__(self, st, ed, cols):
        """Range with start and end values, base weight and breaks count for context splitting."""
        self.st = st # Range for the row.
        self.ed = ed
        self.cols = cols # List of cells.

def split_l2(s, kr, kc, indsingles = False, fmap = fidentity, basestruct = None, indflip = False):
    """Split string to 2d list (ie L2) per row and col keys.
    
    The output is a list of lists, each of varying length.
    If a L2 basestruct is provided,
    will adhere to its structure using the following broadcast rules:
    - Basically matches row by row of base and new.
    - If a new row is shorter than base, the last value is repeated to fill the row.
    - If both are the same length, copied as is.
    - If new row is longer, then additional values will overflow to the next row.
      This might be unintended sometimes, but allows making all items col separated,
      then the new structure is simply adapted to the base structure.
    - If there are too many values in new, they will be ignored.
    - If there are too few values in new, the last one is repeated to fill base. 
    For mixed row + col ratios, singles flag is provided -
    will extract the first value of each row to a separate list,
    and output structure is (row L1,cell L2).
    There MUST be at least one value for row, one value for col when singles is on;
    to prevent errors, the row value is copied to col if it's alone (shouldn't affect results).
    Singles still respects base broadcast rules, and repeats its own last value.
    The fmap function is applied to each cell before insertion to L2.
    If flipped, the keyword for columns is applied before rows.
    TODO: Needs to be a case insensitive split. Use re.split.
    """
    if indflip:
        tmp = kr
        kr = kc
        kc = tmp
    lret = []
    if basestruct is None:
        lrows = s.split(kr)
        lrows = [row.split(kc) for row in lrows]
        for r in lrows:
            cell = [fmap(x) for x in r]
            lret.append(cell)
        if indsingles:
            lsingles = [row[0] for row in lret]
            lcells = [row[1:] if len(row) > 1 else row for row in lret]
            lret = (lsingles,lcells)
    else:
        lrows = s.split(kr)
        r = 0
        lcells = []
        lsingles = []
        vlast = 1
        for row in lrows:
            row2 = row.split(kc)
            row2 = [fmap(x) for x in row2]
            vlast = row2[-1]
            indstop = False
            while not indstop:
                if (r >= len(basestruct) # Too many cell values, ignore.
                or (len(row2) == 0 and len(basestruct) > 0)): # Cell exhausted.
                    indstop = True
                if not indstop:
                    if indsingles: # Singles split.
                        lsingles.append(row2[0]) # Row ratio.
                        if len(row2) > 1:
                            row2 = row2[1:]
                    if len(basestruct[r]) >= len(row2): # Repeat last value.
                        indstop = True
                        broadrow = row2 + [row2[-1]] * (len(basestruct[r]) - len(row2))
                        r = r + 1
                        lcells.append(broadrow)
                    else: # Overfilled this row, cut and move to next.
                        broadrow = row2[:len(basestruct[r])]
                        row2 = row2[len(basestruct[r]):]
                        r = r + 1
                        lcells.append(broadrow)
        # If not enough new rows, repeat the last one for entire base, preserving structure.
        cur = len(lcells)
        while cur < len(basestruct):
            lcells.append([vlast] * len(basestruct[cur]))
            cur = cur + 1
        lret = lcells
        if indsingles:
            lsingles = lsingles + [lsingles[-1]] * (len(basestruct) - len(lsingles))
            lret = (lsingles,lcells)
    return lret

def is_l2(l):
    return isinstance(l[0],list) 

def l2_count(l):
    cnt = 0
    for row in l:
        cnt + cnt + len(row)
    return cnt

def list_percentify(l):
    """Convert each row in L2 to relative part of 100%. 
    
    Also works on L1, applying once globally.
    """
    lret = []
    if is_l2(l):
        for row in l:
            # row2 = [float(v) for v in row]
            row2 = [v / sum(row) for v in row]
            lret.append(row2)
    else:
        row = l[:]
        # row2 = [float(v) for v in row]
        row2 = [v / sum(row) for v in row]
        lret = row2
    return lret

def list_cumsum(l):
    """Apply cumsum to L2 per row, ie newl[n] = l[0:n].sum .
    
    Works with L1.
    Actually edits l inplace, idc.
    """
    lret = []
    if is_l2(l):
        for row in l:
            for (i,v) in enumerate(row):
                if i > 0:
                    row[i] = v + row[i - 1]
            lret.append(row)
    else:
        row = l[:]
        for (i,v) in enumerate(row):
            if i > 0:
                row[i] = v + row[i - 1]
        lret = row
    return lret

def list_rangify(l):
    """Merge every 2 elems in L2 to a range, starting from 0.  
    
    """
    lret = []
    if is_l2(l):
        for row in l:
            row2 = [0] + row
            row3 = []
            for i in range(len(row2) - 1):
                row3.append([row2[i],row2[i + 1]]) 
            lret.append(row3)
    else:
        row2 = [0] + l
        row3 = []
        for i in range(len(row2) - 1):
            row3.append([row2[i],row2[i + 1]]) 
        lret = row3
    return lret

def round_dim(x,y):
    """Return division of two numbers, rounding 0.5 up.
    
    Seems that dimensions which are exactly 0.5 are rounded up - see 680x488, second iter.
    A simple mod check should get the job done.
    If not, can always brute force the divisor with +-1 on each of h/w.
    """
    return x // y + (x % y >= y // 2)

def repeat_div(x,y):
    """Imitates dimension halving common in convolution operations.
    
    This is a pretty big assumption of the model,
    but then if some model doesn't work like that it will be easy to spot.
    """
    while y > 0:
        x = math.ceil(x / 2)
        y = y - 1
    return x

def main_forward(module,x,context,mask,divide):
    
    # Forward.
    h = module.heads // divide
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
    
    return out

def isfloat(t):
    try:
        float(t)
        return True
    except:
        return False

class Script(modules.scripts.Script):
    def __init__(self):
        self.mode = ""
        self.calcmode = ""
        self.indexperiment = False
        self.w = 0
        self.h = 0
        self.usebase = False
        self.usecom = False
        self.usencom = False
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
        self.filters = []

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
                calcmode = gr.Radio(label="Generation mode", choices=["Attention", "Latent"], value="Attention",  type="value", interactive=True)
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

            with gr.Accordion("Presets",open = False):
                with gr.Row():
                    availablepresets = gr.Dropdown(label="Presets", choices=[pr[0] for pr in presets], type="index")
                    applypresets = gr.Button(value="Apply Presets",variant='primary',elem_id="RP_applysetting")
                with gr.Row():
                    presetname = gr.Textbox(label="Preset Name",lines=1,value="",interactive=True,elem_id="RP_preset_name",visible=True)
                    savesets = gr.Button(value="Save to Presets",variant='primary',elem_id="RP_savesetting")
            with gr.Row():
                nchangeand = gr.Checkbox(value=False, label="disable convert 'AND' to 'BREAK'", interactive=True, elem_id="RP_ncand")
                debug = gr.Checkbox(value=False, label="debug", interactive=True, elem_id="RP_debug")
            settings = [mode, ratios, baseratios, usebase, usecom, usencom, calcmode]
        
        def setpreset(select):
            presets = loadpresets(filepath)
            preset = presets[select]
            preset = preset[1:]
            def booler(text):
                return text == "TRUE" or text == "true" or text == "True"
            preset[1],preset[2] = preset[1].replace('"',""),preset[2].replace('"',"")
            preset[3],preset[4],preset[5] = booler(preset[3]),booler(preset[4]),booler(preset[5])
            while 7 > len(preset):
                preset.append("")
            if preset[6] == "" : preset[6] = "Attention"
            return [gr.update(value = pr) for pr in preset]
        
        def makeimgtmp(aratio,mode,usecom,usebase):
            aratios = aratio.split(";")
            if len(aratios) == 1 : aratios[0] = "1," + aratios[0]
            h = w = 128
            icells = []
            ocells = []
            def startend(lst):
                o = []
                s = 0
                lst = [l/sum(lst) for l in lst]
                for i in lange(lst):
                    if i == 0 :o.append([0,lst[0]])
                    else : o.append([s, s + lst[i]])
                    s = s + lst[i]
                return o
            for rc in aratios:
                rc = rc.split(",")
                rc = [float(r) for r in rc]
                if len(rc) == 1 : rc = [rc[0]]*2
                ocells.append(rc[0])
                icells.append(startend(rc[1:]))
            fx = np.zeros((h,w, 3), np.uint8)
            ocells = startend(ocells)
            print(ocells,icells)
            for i,ocell in enumerate(ocells):
                for icell in icells[i]:
                    if "Horizontal" in mode:
                        fx[int(h*ocell[0]):int(h*ocell[1]),int(w*icell[0]):int(w*icell[1]),:] = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
                    elif "Vertical" in mode: 
                        fx[int(h*icell[0]):int(h*icell[1]),int(w*ocell[0]):int(w*ocell[1]),:] = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
            img = PIL.Image.fromarray(fx)
            draw = PIL.ImageDraw.Draw(img)
            c = 0
            def coldealer(col):
                if sum(col) > 380:return "black"
                else:return "white"
            temp = ""
            for i,ocell in enumerate(ocells):
                for j,icell in enumerate(icells[i]):
                    if "Horizontal" in mode:
                        draw.text((int(w*icell[0]),int(h*ocell[0])),f"{c}",coldealer(fx[int(h*ocell[0]),int(w*icell[0])]))
                        if j != len(icells[i]) -1 : temp = temp + " " + KEYCOL + "\n"
                    elif "Vertical" in mode: 
                        draw.text((int(w*ocell[0]),int(h*icell[0])),f"{c}",coldealer(fx[int(h*icell[0]),int(w*ocell[0])]))
                        if j != len(icells[i]) -1 : temp = temp + " " + KEYROW + "\n"
                    c += 1
                if "Horizontal" in mode and i !=len(ocells)-1 :
                    temp = temp+ " " + KEYROW + "\n"
                elif "Vertical" in mode and i !=len(ocells) -1 :
                    temp = temp + " " + KEYCOL + "\n"
            if usebase : temp = " " + KEYBASE + "\n" +temp
            if usecom : temp = " " + KEYCOMM + "\n" +temp
            return img,gr.update(value = temp)

        maketemp.click(fn=makeimgtmp, inputs =[ratios,mode,usecom,usebase],outputs = [areasimg,template])
        applypresets.click(fn=setpreset, inputs = availablepresets, outputs=settings)
        savesets.click(fn=savepresets, inputs = [presetname,*settings],outputs=availablepresets)
                
        return [active, debug, mode, ratios, baseratios, usebase, usecom, usencom, calcmode, nchangeand]

    def process(self, p, active, debug, mode, aratios, bratios, usebase, usecom, usencom, calcmode, nchangeand):
        if active:
            savepresets("lastrun",mode, aratios,bratios, usebase, usecom, usencom, calcmode)
            self.__init__()
            self.active = True
            self.mode = mode
            comprompt = comnegprompt = None
            # SBM matrix mode detection.

            self.orig_all_prompts = p.all_prompts
            self.orig_all_negative_prompts = p.all_negative_prompts 

            if not nchangeand and "AND" in p.prompt.upper():
                p.prompt = p.prompt.replace("AND",KEYBRK)
                for i in lange(p.all_prompts):
                    p.all_prompts[i] = p.all_prompts[i].replace("AND",KEYBRK)
            if (KEYROW in p.prompt.upper() or KEYCOL in p.prompt.upper() or DELIMROW in aratios):
                self.indexperiment = True
            elif KEYBRK not in p.prompt.upper():
                self.active = False
                unloader(self,p)
                return
            self.w = p.width
            self.h = p.height
            self.batch_size = p.batch_size
            self.batch_cond_uncond = shared.batch_cond_uncond
            
            self.calcmode = calcmode
            if not hasattr(self,"batch_cond_uncond") : self.batch_cond_uncond = shared.batch_cond_uncond

            self.debug = debug
            self.usebase = usebase
            self.usecom = usecom
            if KEYCOMM in p.prompt: # Automatic common toggle.
                self.usecom = True
            if KEYCOMM in p.negative_prompt: # Automatic common toggle.
                self.usencom = True

            if hasattr(p,"enable_hr"): # Img2img doesn't have it.
                self.hr = p.enable_hr
                self.hr_w = (p.hr_resize_x if p.hr_resize_x > p.width else p.width * p.hr_scale)
                self.hr_h = (p.hr_resize_y if p.hr_resize_y > p.height else p.height * p.hr_scale)

            # SBM In matrix mode, the ratios are broken up 
            if self.indexperiment:
                if self.usecom and KEYCOMM in p.prompt:
                    comprompt = p.prompt.split(KEYCOMM,1)[0]
                    p.prompt = p.prompt.split(KEYCOMM,1)[1]
                elif self.usecom and KEYBRK in p.prompt:
                    comprompt = p.prompt.split(KEYBRK,1)[0]
                    p.prompt = p.prompt.split(KEYBRK,1)[1]
                if self.usencom and KEYCOMM in p.negative_prompt:
                    comnegprompt = p.negative_prompt.split(KEYCOMM,1)[0]
                    p.negative_prompt = p.negative_prompt.split(KEYCOMM,1)[1]
                elif self.usencom and KEYBRK in p.negative_prompt:
                    comnegprompt = p.negative_prompt.split(KEYBRK,1)[0]
                    p.negative_prompt = p.negative_prompt.split(KEYBRK,1)[1]
                # The addrow/addcol syntax is better, cannot detect regular breaks without it.
                # In any case, the preferred method will anchor the L2 structure. 
                if (KEYBASE in p.prompt.upper()): # Designated base.
                    self.usebase = True
                    baseprompt = p.prompt.split(KEYBASE,1)[0]
                    mainprompt = p.prompt.split(KEYBASE,1)[1] 
                    self.basebreak = fcountbrk(baseprompt)
                elif usebase: # Get base by first break as usual.
                    baseprompt = p.prompt.split(KEYBRK,1)[0]
                    mainprompt = p.prompt.split(KEYBRK,1)[1]
                else:
                    baseprompt = ""
                    mainprompt = p.prompt
                indflip = (mode == "Vertical")
                if (KEYCOL in mainprompt.upper() or KEYROW in mainprompt.upper()):
                    breaks = mainprompt.count(KEYROW) + mainprompt.count(KEYCOL) + int(self.usebase)
                    # Prompt anchors, count breaks between special keywords.
                    lbreaks = split_l2(mainprompt, KEYROW, KEYCOL, fmap = fcountbrk, indflip = indflip)
                    # Standard ratios, split to rows and cols.
                    (aratios2r,aratios2) = split_l2(aratios, DELIMROW, DELIMCOL, 
                                                    indsingles = True, fmap = ffloat, basestruct = lbreaks, indflip = indflip)
                    # More like "bweights", applied per cell only.
                    bratios2 = split_l2(bratios, DELIMROW, DELIMCOL, fmap = ffloat, basestruct = lbreaks)
                else:
                    breaks = mainprompt.count(KEYBRK) + int(self.usebase)
                    (aratios2r,aratios2) = split_l2(aratios, DELIMROW, DELIMCOL, indsingles = True, fmap = ffloat, indflip = indflip)
                    # Cannot determine which breaks matter.
                    lbreaks = split_l2("0", KEYROW, KEYCOL, fmap = fint, basestruct = aratios2, indflip = indflip)
                    bratios2 = split_l2(bratios, DELIMROW, DELIMCOL, fmap = ffloat, basestruct = lbreaks, indflip = indflip)
                    # If insufficient breaks, try to broadcast prompt - a bit dumb.
                    breaks = fcountbrk(mainprompt)
                    lastprompt = mainprompt.rsplit(KEYBRK)[-1]
                    if l2_count(aratios2) > breaks: 
                        mainprompt = mainprompt + (fspace(KEYBRK) + lastprompt) * (l2_count(aratios2) - breaks) 
                
                # Change all splitters to breaks.
                aratios2 = list_percentify(aratios2)
                aratios2 = list_cumsum(aratios2)
                aratios = list_rangify(aratios2)
                aratios2r = list_percentify(aratios2r)
                aratios2r = list_cumsum(aratios2r)
                aratiosr = list_rangify(aratios2r)
                bratios = bratios2 
                
                # Merge various L2s to cells and rows.
                drows = []
                for r,_ in enumerate(lbreaks):
                    dcells = []
                    for c,_ in enumerate(lbreaks[r]):
                        d = RegionCell(aratios[r][c][0], aratios[r][c][1], bratios[r][c], lbreaks[r][c])
                        dcells.append(d)
                    drow = RegionRow(aratiosr[r][0], aratiosr[r][1], dcells)
                    drows.append(drow)
                self.aratios = drows
                # Convert all keys to breaks, and expand neg to fit.
                mainprompt = mainprompt.replace(KEYROW,KEYBRK) # Cont: Should be case insensitive.
                mainprompt = mainprompt.replace(KEYCOL,KEYBRK)
                p.prompt = mainprompt
                if self.usebase:
                    p.prompt = baseprompt + fspace(KEYBRK) + p.prompt 
                p.all_prompts = [p.prompt] * len(p.all_prompts)
                np = p.negative_prompt
                np.replace(KEYROW,KEYBRK)
                np.replace(KEYCOL,KEYBRK)
                np = np.split(KEYBRK)
                nbreaks = len(np) - 1
                if breaks >= nbreaks: # Repeating the first neg as in orig code.
                    np.extend([np[0]] * (breaks - nbreaks))
                else: # Cut off the excess negs.
                    np = np[0:breaks + 1]
                for i ,n in enumerate(np):
                    if n.isspace() or n =="":
                        np[i] = ","
                # p.negative_prompt = fspace(KEYBRK).join(np)
                # p.all_negative_prompts = [p.negative_prompt] * len(p.all_negative_prompts)
                if comprompt is not None : 
                    p.prompt = comprompt + fspace(KEYBRK) + p.prompt
                    for i in lange(p.all_prompts):
                        p.all_prompts[i] = comprompt + fspace(KEYBRK) + p.all_prompts[i]
                if comnegprompt is not None :
                    p.negative_prompt = comnegprompt + fspace(KEYBRK) + p.negative_prompt
                    for i in lange(p.all_negative_prompts):
                        p.all_negative_prompts[i] = comnegprompt + fspace(KEYBRK) + p.all_negative_prompts[i]
                self, p = commondealer(self, p, self.usecom, self.usencom)
            else:
                self, p = promptdealer(self, p, aratios, bratios, usebase, usecom, usencom)
                self, p = commondealer(self, p, usecom, usencom)
    
            self.pt, self.nt ,ppt,pnt, self.eq = tokendealer(p)

            #self.eq = True if len(self.pt) == len(self.nt) else False
            
            if calcmode == "Attention":
                self.handle = hook_forwards(self, p.sd_model.model.diffusion_model)
                if hasattr(self,"batch_cond_uncond") : shared.batch_cond_uncond = self.batch_cond_uncond
            else:
                if not hasattr(self,"dd_callbacks"):
                    self.dd_callbacks = on_cfg_denoised(self.denoised_callback)
                if not hasattr(self,"dr_callbacks"):
                    self.dr_callbacks = on_cfg_denoiser(self.denoiser_callback)
                self.handle = hook_forwards(self, p.sd_model.model.diffusion_model,remove = True)
                shared.batch_cond_uncond = False
                del self.handle
                self, p = calcdealer(self, p,calcmode)
                global regioner
                regioner.reset()
                regioner.divide = self.divide if not self.usebase else self.divide  +1
                regioner.batch = p.batch_size
                if self.debug : print(p.prompt)

            print(f"pos tokens : {ppt}, neg tokens : {pnt}")
            if debug : 
                print(f"mode : {self.calcmode}\ndivide : {mode}\nratios : {aratios}\nusebase : {self.usebase}")
                print(f"base ratios : {self.bratios}\nusecommon : {self.usecom}\nusenegcom : {self.usencom}\nuse 2D : {self.indexperiment}")
                print(f"divide : {self.divide}\neq : {self.eq}")
                if self.indexperiment:
                    for row in self.aratios:
                        print(f"row : {row.st,row.ed},cell : {[[c.st,c.ed] for c in row.cols]}")
        else:
            unloader(self,p)
        return p

    def process_batch(self, p, active, debug, mode, aratios, bratios, usebase, usecom, usencom, calcmode,nchangeand,**kwargs):
        global lactive,labug
        if self.active and calcmode =="Latent":
            import lora
            global orig_lora_forward,orig_lora_apply_weights,lactive, orig_lora_Linear_forward, orig_lora_Conv2d_forward
            if hasattr(lora,"lora_apply_weights"): # for new LoRA applying
                if self.debug : print("hijack lora_apply_weights")
                orig_lora_apply_weights = lora.lora_apply_weights
                orig_lora_Linear_forward = torch.nn.Linear.forward
                orig_lora_Conv2d_forward = torch.nn.Conv2d.forward
                lora.lora_apply_weights = lora_apply_weights
                torch.nn.Linear.forward = lora_Linear_forward
                torch.nn.Conv2d.forward = lora_Conv2d_forward

                for l in lora.loaded_loras:
                    for key in l.modules.keys():
                        changethedevice(l.modules[key])
                restoremodel(p)

            elif hasattr(lora,"lora_forward"):
                if self.debug : print("hijack lora_forward")
                orig_lora_forward = lora.lora_forward
                lora.lora_forward = lora_forward
            lactive = True
            labug = self.debug
            self = lora_namer(self,p)
        else:
            lactive = False


    # TODO: Should remove usebase, usecom, usencom - grabbed from self value.
    def postprocess_image(self, p, pp, active, debug, mode, aratios, bratios, usebase, usecom, usencom,calcmode,nchangeand):
        if not self.active:
            return p
        if self.usecom or self.indexperiment:
            p.prompt = self.orig_all_prompts[0]
            p.all_prompts[self.imgcount] = self.orig_all_prompts[self.imgcount]
        if self.usencom:
            p.negative_prompt = self.orig_all_negative_prompts[0]
            p.all_negative_prompts[self.imgcount] = self.orig_all_negative_prompts[self.imgcount]
        self.imgcount += 1
        p.extra_generation_params["Regional Prompter"] = f"mode:{mode},divide ratio : {aratios}, Use base : {self.usebase}, Base ratio : {bratios}, Use common : {self.usecom}, Use N-common : {self.usencom}"
        return p

    def postprocess(self, p, processed, *args):
        if self.active : 
            with open(os.path.join(paths.data_path, "params.txt"), "w", encoding="utf8") as file:
                processed = Processed(p, [], p.seed, "")
                file.write(processed.infotext(p, 0))
        
        if hasattr(self,"handle"):
            hook_forwards(self, p.sd_model.model.diffusion_model, remove=True)
            del self.handle

        global orig_lora_Linear_forward, orig_lora_Conv2d_forward,orig_lora_apply_weights,orig_lora_forward
        import lora
        if orig_lora_apply_weights != None :
            lora.lora_apply_weights = orig_lora_apply_weights
            orig_lora_apply_weights = None

        if orig_lora_forward != None :
            lora.lora_forward = orig_lora_forward
            orig_lora_forward = None

        if orig_lora_Linear_forward != None :
            torch.nn.Linear.forward = orig_lora_Linear_forward
            orig_lora_Linear_forward = None

        if orig_lora_Conv2d_forward != None :
            torch.nn.Conv2d.forward = orig_lora_Conv2d_forward
            orig_lora_Conv2d_forward = None

###################################################
###### Latent Method denoise call back
# Using the AND syntax with shared.batch_cond_uncond = False
# the U-NET is calculated (the number of prompts divided by AND) + 1 times.
# This means that the calculation is performed for the area + 1 times.
# This mechanism is used to apply LoRA by region by changing the LoRA application rate for each U-NET calculation.
# The problem here is that in the web-ui system, if more than two batch sizes are set, 
# a problem will occur if the number of areas and the batch size are not the same.
# If the batch is 1 for 3 areas, the calculation is performed 4 times: Area1, Area2, Area3, and Negative. 
# However, if the batch is 2, 
# [Batch1-Area1, Batch1-Area2]
# [Batch1-Area3, Batch2-Area1]
# [Batch2-Area2, Batch2-Area3]
# [Batch1-Negative, Batch2-Negative]
# and the areas of simultaneous computation will be different. 
# Therefore, it is necessary to change the order in advance.
# [Batch1-Area1, Batch1-Area2] -> [Batch1-Area1, Batch2-Area1] 
# [Batch1-Area3, Batch2-Area1] -> [Batch1-Area2, Batch2-Area2] 
# [Batch2-Area2, Batch2-Area3] -> [Batch1-Area3, Batch2-Area3] 

    def denoiser_callback(self, params: CFGDenoiserParams):
        if lactive:
            xt = params.x.clone()
            ict = params.image_cond.clone()
            st =  params.sigma.clone()
            ct =  params.text_cond.clone()
            areas = xt.shape[0] // self.batch_size -1

            for a in range(areas):
                for b in range(self.batch_size):
                    params.x[b+a*self.batch_size] = xt[a + b * areas]
                    params.image_cond[b+a*self.batch_size] = ict[a + b * areas]
                    params.sigma[b+a*self.batch_size] = st[a + b * areas]
                    params.text_cond[b+a*self.batch_size] = ct[a + b * areas]

    def denoised_callback(self, params: CFGDenoisedParams):
        if lactive:
            x = params.x
            batch = self.batch_size
            # x.shape = [batch_size, C, H // 8, W // 8]
            if self.filters == [] :
                self.filters = makefilters(x.shape[1], x.shape[2], x.shape[3],self.aratios,self.mode,self.usebase,self.bratios,self.indexperiment)
                self.neg_filters = [1- f for f in self.filters]
            else:
                if self.filters[0].size() != x[0].size():
                    self.filters = makefilters(x.shape[1], x.shape[2], x.shape[3],self.aratios,self.mode,self.usebase,self.bratios,self.indexperiment)
                    self.neg_filters = [1- f for f in self.filters]

            if self.debug : print("filterlength : ",len(self.filters))

            x = params.x
            xt = params.x.clone()

            areas = xt.shape[0] // batch -1

            if labug : 
                for i in range(params.x.shape[0]):
                    print(torch.max(params.x[i]))

            for b in range(batch):
                for a in range(areas):
                    x[a + b * areas] = xt[b+a*batch]

            for b in range(batch):
                for a in range(areas) :
                    #print(f"x = {x.size()}f = {r}, b={b}, count = {r + b*areas}, uncon = {x.size()[0]+(b-batch)}")
                    x[a + b*areas, :, :, :] =  x[a + b*areas, :, :, :] * self.filters[a] + x[x.size()[0]+(b-batch), :, :, :] * self.neg_filters[a]

################################################################################
##### Attention mode 

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

        sumer = 0
        h_states = []
        contexts = context.clone()
        # SBM Matrix mode.
        def matsepcalc(x,contexts,mask,pn,divide):
            add = 0 # TEMP
            # Completely independent size calc.
            # Basically: sqrt(hw_ratio*x.size[1])

            # And I think shape is better than size()?
            # My guesstimate is that the the formula is a repeated ceil(d/2),
            # there doesn't seem to be a function to calculate this directly,
            # but we can either brute force it with distance, 
            # or assume the model ONLY transforms by x2 at a time.
            # Repeated ceils may cause the value to veer too greatly from rounding. 
            xs = x.size()[1]
            # scale = round(math.sqrt(height*width/xs))
            scale = math.ceil(math.log2(math.sqrt(height * width / xs))) # SBM Assumed halving transpose.
            
            # dsh = round_dim(height, scale)
            # dsw = round_dim(width, scale)
            dsh = repeat_div(height,scale)
            dsw = repeat_div(width,scale)
            
            if "Horizontal" in self.mode: # Map columns / rows first to outer / inner.
                dsout = dsw
                dsin = dsh
            elif "Vertical" in self.mode:
                dsout = dsh
                dsin = dsw

            tll = self.pt if pn else self.nt

            if self.debug : print(scale,dsh,dsw,dsh*dsw,x.size()[1])
            
            # Base forward.
            cad = 0 if self.usebase else 1 # 1 * self.usebase is shorter.
            i = 0
            outb = None
            if self.usebase:
                context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
                i = i + 1 + self.basebreak
                out = main_forward(module, x, context, mask, divide)

                if len(self.nt) == 1 and not pn:
                    if self.debug : print("return out for NP")
                    return out
                # if self.usebase:
                outb = out.clone()
                outb = outb.reshape(outb.size()[0], dsh, dsw, outb.size()[2]) 

            sumout = 0

            if self.debug : print(f"tokens : {tll},pn : {pn}")
            if self.debug : print([r for r in self.aratios])

            for drow in self.aratios:
                v_states = []
                sumin = 0
                for dcell in drow.cols:
                    # Grabs a set of tokens depending on number of unrelated breaks.
                    context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
                    if self.debug : print(f"tokens : {tll[i][0]*TOKENSCON}-{tll[i][1]*TOKENSCON}")
                    i = i + 1 + dcell.breaks
                    # if i >= contexts.size()[1]: 
                    #     indlast = True
                    out = main_forward(module, x, context, mask, divide)
                    if self.debug : print(f" dcell.breaks : {dcell.breaks}, dcell.ed : {dcell.ed}, dcell.st : {dcell.st}")
                    if len(self.nt) == 1 and not pn:
                        if self.debug : print("return out for NP")
                        return out
                    # Actual matrix split by region.
                    
                    out = out.reshape(out.size()[0], dsh, dsw, out.size()[2]) # convert to main shape.
                    # if indlast:
                    addout = 0
                    addin = 0
                    sumin = sumin + int(dsin*dcell.ed) - int(dsin*dcell.st)
                    if dcell.ed >= 0.999:
                        addin = sumin - dsin
                        sumout = sumout + int(dsout*drow.ed) - int(dsout*drow.st)
                        if drow.ed >= 0.999:
                            addout = sumout - dsout
                    if "Horizontal" in self.mode:
                        out = out[:,int(dsh*drow.st) + addout:int(dsh*drow.ed),
                                    int(dsw*dcell.st) + addin:int(dsw*dcell.ed),:]
                        if self.usebase : 
                            # outb_t = outb[:,:,int(dsw*drow.st):int(dsw*drow.ed),:].clone()
                            outb_t = outb[:,int(dsh*drow.st) + addout:int(dsh*drow.ed),
                                            int(dsw*dcell.st) + addin:int(dsw*dcell.ed),:].clone()
                            out = out * (1 - dcell.base) + outb_t * dcell.base
                    elif "Vertical" in self.mode: # Cols are the outer list, rows are cells.
                        out = out[:,int(dsh*dcell.st) + addin:int(dsh*dcell.ed),
                                  int(dsw*drow.st) + addout:int(dsw*drow.ed),:]
                        if self.usebase : 
                            # outb_t = outb[:,:,int(dsw*drow.st):int(dsw*drow.ed),:].clone()
                            outb_t = outb[:,int(dsh*dcell.st) + addin:int(dsh*dcell.ed),
                                          int(dsw*drow.st) + addout:int(dsw*drow.ed),:].clone()
                            out = out * (1 - dcell.base) + outb_t * dcell.base
                    if self.debug : print(f"sumer:{sumer},dsw:{dsw},add:{add}")
            
                    v_states.append(out)
                    if self.debug : 
                        for h in v_states:
                            print(h.size())
                            
                if "Horizontal" in self.mode:
                    ox = torch.cat(v_states,dim = 2) # First concat the cells to rows.
                elif "Vertical" in self.mode:
                    ox = torch.cat(v_states,dim = 1) # Cols first mode, concat to cols.
                h_states.append(ox)
            if "Horizontal" in self.mode:
                ox = torch.cat(h_states,dim = 1) # Second, concat rows to layer.
            elif "Vertical" in self.mode:
                ox = torch.cat(h_states,dim = 2) # Or cols.
            ox = ox.reshape(x.size()[0],x.size()[1],x.size()[2]) # Restore to 3d source.  
            return ox

        def regsepcalc(x, contexts, mask, pn,divide):
            sumer = 0
            h_states = []

            tll = self.pt if pn else self.nt
            if self.debug : print(f"tokens : {tll},pn : {pn}")

            for i, tl in enumerate(tll):
                context = contexts[:, tl[0] * TOKENSCON : tl[1] * TOKENSCON, :]
                if self.debug : print(f"tokens : {tl[0]*TOKENSCON}-{tl[1]*TOKENSCON}")

                if self.usebase:
                    if i != 0:
                        area = self.aratios[i - 1]
                        bweight = self.bratios[i - 1]
                else:
                    area = self.aratios[i]

                out = main_forward(module, x, context, mask, divide)

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
            if self.debug : print("same token size and divisions")
            if self.indexperiment:
                ox = matsepcalc(x, contexts, mask, True, 1)
            else:
                ox = regsepcalc(x, contexts, mask, True, 1)
        elif x.size()[0] == 1 * self.batch_size:
            if self.debug : print("different tokens size")
            if self.indexperiment:
                ox = matsepcalc(x, contexts, mask, self.pn, 1)
            else:
                ox = regsepcalc(x, contexts, mask, self.pn, 1)
        else:
            if self.debug : print("same token size and different divisions")
            px, nx = x.chunk(2)
            if self.indexperiment:
                opx = matsepcalc(px, contexts, mask, True, 2)
                onx = matsepcalc(nx, contexts, mask, False, 2)
            else:
                opx = regsepcalc(px, contexts, mask, True, 2)
                onx = regsepcalc(nx, contexts, mask, False, 2)
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

############################################################
##### prompts, tokens

def tokendealer(p):
    ppl = p.prompt.split(KEYBRK)
    npl = p.negative_prompt.split(KEYBRK)
    pt, nt, ppt, pnt = [], [], [], []

    padd = 0
    for pp in ppl:
        _, tokens = shared.sd_model.cond_stage_model.tokenize_line(pp)
        pt.append([padd, tokens // TOKENS + 1 + padd])
        ppt.append(tokens)
        padd = tokens // TOKENS + 1 + padd
    paddp = padd
    padd = 0
    for np in npl:
        _, tokens = shared.sd_model.cond_stage_model.tokenize_line(np)
        nt.append([padd, tokens // TOKENS + 1 + padd])
        pnt.append(tokens)
        padd = tokens // TOKENS + 1 + padd
    eq = paddp == padd
    return pt, nt, ppt, pnt, eq

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
    return self, p

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

def calcdealer(self, p, calcmode):
    if calcmode == "Latent":
        p.prompt = p.prompt.replace("BREAK", "AND")
        for i in lange(p.all_prompts):
            p.all_prompts[i] = p.all_prompts[i].replace("BREAK", "AND")
        p.negative_prompt = p.negative_prompt.replace("BREAK", "AND")
        for i in lange(p.all_negative_prompts):
            p.all_negative_prompts[i] = p.all_negative_prompts[i].replace("BREAK", "AND")
    self.divide = p.prompt.count("AND") + 1
    return self, p

def unloader(self,p):
    if hasattr(self,"handle"):
        print("unloaded")
        hook_forwards(self, p.sd_model.model.diffusion_model, remove=True)
        del self.handle
        if hasattr(self,"batch_cond_uncond") : shared.batch_cond_uncond = self.batch_cond_uncond
    global lactive
    lactive = False
    self.active = False

#############################################################
##### Preset save and load

def savepresets(name,mode, ratios, baseratios, usebase,usecom, usencom, calcmode):
    path_root = scripts.basedir()
    filepath = os.path.join(path_root,"scripts", "regional_prompter_presets.csv")
    try:
        with open(filepath,mode = 'r',encoding="utf-8") as f:
            presets = f.readlines()
            pr = f'{name},{mode},"{ratios}","{baseratios}",{str(usebase)},{str(usecom)},{str(usencom)},{str(calcmode)}\n'
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
                    f.writelines('"name","mode","divide ratios,"use base","baseratios","usecom","usencom","calcmode"\n')
                    for pr in presets:
                        text = ",".join(pr) + "\n"
                        f.writelines(text)
            except:
                pass
    return presets

######################################################
##### Latent Method

def lora_namer(self,p):
    ldict = {}
    import lora as loraclass
    for lora in loraclass.loaded_loras:
        ldict[lora.name] = lora.multiplier

    subprompts = p.prompt.split("AND")
    llist =[ldict.copy() for i in range(len(subprompts)+1)]
    for i, prompt in enumerate(subprompts):
        _, extranets = extra_networks.parse_prompts([prompt])
        calledloras = extranets["lora"]

        names = ""
        tdict = {}

        for called in calledloras:
            names = names + called.items[0]
            tdict[called.items[0]] = called.items[1]

        for key in llist[i].keys():
            if key.split("added_by_lora_block_weight")[0] not in names:
                llist[i+1][key] = 0
            elif key in names:
                llist[i+1][key] = float(tdict[key])
                
    global regioner
    regioner.te_llist = llist
    regioner.u_llist = llist[1:]
    regioner.u_llist.append(llist[0])
    regioner.ndeleter()
    if self.debug:
        print(regioner.te_llist)
        print(regioner.u_llist)


def makefilters(c,h,w,masks,mode,usebase,bratios,xy): 
    filters = []
    x =  torch.zeros(c, h, w).to(devices.device)
    if usebase:
        x0 = torch.zeros(c, h, w).to(devices.device)
    i=0
    if xy:
        for drow in masks:
            for dcell in drow.cols:
                fx = x.clone()
                if "Horizontal" in mode:
                    if usebase:
                        fx[:,int(h*drow.st):int(h*drow.ed),int(w*dcell.st):int(w*dcell.ed)] = 1 - dcell.base
                        x0[:,int(h*drow.st):int(h*drow.ed),int(w*dcell.st):int(w*dcell.ed)] = dcell.base
                    else:
                        fx[:,int(h*drow.st):int(h*drow.ed),int(w*dcell.st):int(w*dcell.ed)] = 1    
                elif "Vertical" in mode: 
                    if usebase:
                        fx[:,int(h*dcell.st):int(h*dcell.ed),int(w*drow.st):int(w*drow.ed)] = 1 - dcell.base
                        x0[:,int(h*dcell.st):int(h*dcell.ed),int(w*drow.st):int(w*drow.ed)] = dcell.base
                    else:
                        fx[:,int(h*drow.st):int(h*drow.ed),int(w*dcell.st):int(w*dcell.ed)] = 1  
                filters.append(fx)
                i +=1
    else:
        if "Horizontal" in mode:
            for mask, bratio in zip(masks,bratios):
                fx = x.clone()
                if usebase:
                    fx[:,:,int(mask[0]*w):int(mask[1]*w)] = 1 - bratio
                    x0[:,:,int(mask[0]*w):int(mask[1]*w)] = bratio
                else:
                    fx[:,:,int(mask[0]*w):int(mask[1]*w)] = 1
                filters.append(fx)
        elif "Vertical" in mode:
            for mask, bratio in zip(masks,bratios):
                fx = x.clone()
                if usebase:
                    fx[:,int(mask[0]*h):int(mask[1]*h),:] = 1 -bratio
                    x0[:,int(mask[0]*h):int(mask[1]*h),:] = bratio
                else:
                    fx[:,int(mask[0]*h):int(mask[1]*h),:] = 1
                filters.append(fx)
    if usebase : filters.insert(0,x0)
    if labug : print(i,len(filters))

    return filters

######################################################
##### Latent Method LoRA changer

TE_START_NAME = "transformer_text_model_encoder_layers_0_self_attn_q_proj"
UNET_START_NAME = "diffusion_model_time_embed_0"

class LoRARegioner:

    def __init__(self):
        self.te_count = 0
        self.u_count = 0
        self.te_llist = [{}]
        self.u_llist = [{}]
        self.mlist = {}

    def ndeleter(self):
        for key in self.te_llist[0].keys():
            self.te_llist[0][key] = 0
        for key in self.u_llist[-1].keys():
            self.u_llist[-1][key] = 0

    def te_start(self):
        self.mlist = self.te_llist[self.te_count % len(self.te_llist)]
        self.te_count += 1
        import lora
        for i in range(len(lora.loaded_loras)):
            lora.loaded_loras[i].multiplier = self.mlist[lora.loaded_loras[i].name]

    def u_start(self):
        if labug : print("u_count",self.u_count ,"divide",{self.divide},"u_count '%' divide",  self.u_count % len(self.u_llist))
        self.mlist = self.u_llist[self.u_count % len(self.u_llist)]
        self.u_count  += 1
        import lora
        for i in range(len(lora.loaded_loras)):
            lora.loaded_loras[i].multiplier = self.mlist[lora.loaded_loras[i].name]
    
    def reset(self):
        self.te_count = 0
        self.u_count = 0
    
regioner = LoRARegioner()


def lora_forward(module, input, res):
    import lora

    if len(lora.loaded_loras) == 0:
        return res

    lora_layer_name = getattr(module, 'lora_layer_name', None)

    if lactive:
        global regioner

        if lora_layer_name == TE_START_NAME:
            regioner.te_start()
        elif lora_layer_name == UNET_START_NAME:
            regioner.u_start()

    for lora_m in lora.loaded_loras:
        module = lora_m.modules.get(lora_layer_name, None)
        if labug and lora_layer_name is not None :
            if "9" in lora_layer_name and ("_attn1_to_q" in lora_layer_name or "self_attn_q_proj" in lora_layer_name): print(lora_m.multiplier,lora_m.name,lora_layer_name)
        if module is not None and lora_m.multiplier:
            if hasattr(module, 'up'):
                scale = lora_m.multiplier * (module.alpha / module.up.weight.size(1) if module.alpha else 1.0)
            else:
                scale = lora_m.multiplier * (module.alpha / module.dim if module.alpha else 1.0)
            
            if hasattr(shared.opts,"lora_apply_to_outputs"):
                if shared.opts.lora_apply_to_outputs and res.shape == input.shape:
                    x = res
                else:
                    x = input    
            else:
                x = input
        
            if hasattr(module, 'inference'):
                res = res + module.inference(x) * scale
            elif hasattr(module, 'up'):
                res = res + module.up(module.down(x)) * scale

    return res

def lora_apply_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.MultiheadAttention]):
    import lora as loramodule

    lora_layer_name = getattr(self, 'lora_layer_name', None)
    if lora_layer_name is None:
        return

    if lactive:
        global regioner

        if lora_layer_name == TE_START_NAME:
            regioner.te_start()
        elif lora_layer_name == UNET_START_NAME:
            regioner.u_start()

    current_names = getattr(self, "lora_current_names", ())
    wanted_names = tuple((x.name, x.multiplier) for x in loramodule.loaded_loras)

    if lactive : current_names = None

    weights_backup = getattr(self, "lora_weights_backup", None)
    if weights_backup is None:
        if isinstance(self, torch.nn.MultiheadAttention):
            weights_backup = (self.in_proj_weight.to(devices.cpu, copy=True), self.out_proj.weight.to(devices.cpu, copy=True))
        else:
            weights_backup = self.weight.to(devices.cpu, copy=True)

        self.lora_weights_backup = weights_backup

    if current_names != wanted_names:
        if weights_backup is not None:
            if isinstance(self, torch.nn.MultiheadAttention):
                self.in_proj_weight.copy_(weights_backup[0])
                self.out_proj.weight.copy_(weights_backup[1])
            else:
                self.weight.copy_(weights_backup)

        for lora in loramodule.loaded_loras:
            module = lora.modules.get(lora_layer_name, None)
            if module is not None and hasattr(self, 'weight'):
                self.weight += loramodule.lora_calc_updown(lora, module, self.weight)
                continue

            module_q = lora.modules.get(lora_layer_name + "_q_proj", None)
            module_k = lora.modules.get(lora_layer_name + "_k_proj", None)
            module_v = lora.modules.get(lora_layer_name + "_v_proj", None)
            module_out = lora.modules.get(lora_layer_name + "_out_proj", None)

            if isinstance(self, torch.nn.MultiheadAttention) and module_q and module_k and module_v and module_out:
                updown_q = loramodule.lora_calc_updown(lora, module_q, self.in_proj_weight)
                updown_k = loramodule.lora_calc_updown(lora, module_k, self.in_proj_weight)
                updown_v = loramodule.lora_calc_updown(lora, module_v, self.in_proj_weight)
                updown_qkv = torch.vstack([updown_q, updown_k, updown_v])

                self.in_proj_weight += updown_qkv
                self.out_proj.weight += loramodule.lora_calc_updown(lora, module_out, self.out_proj.weight)
                continue

            if module is None:
                continue

            print(f'failed to calculate lora weights for layer {lora_layer_name}')

        setattr(self, "lora_current_names", wanted_names)

############################################################
##### for new lora apply method in web-ui

def lora_Linear_forward(self, input):
    return lora_forward(self, input, torch.nn.Linear_forward_before_lora(self, input))

def lora_Conv2d_forward(self, input):
    return lora_forward(self, input, torch.nn.Conv2d_forward_before_lora(self, input))

def changethedevice(module):
    if type(module).__name__ == "LoraUpDownModule":
        if hasattr(module,"up_model.weight") :
            module.up_model.weight = torch.nn.Parameter(module.up_model.weight.to(devices.device, dtype = torch.float))
            module.down_model.weight = torch.nn.Parameter(module.down_model.weight.to(devices.device, dtype=torch.float))
        else:
            module.up.weight = torch.nn.Parameter(module.up.weight.to(devices.device, dtype = torch.float))
            module.down.weight = torch.nn.Parameter(module.down.weight.to(devices.device, dtype=torch.float))
        
    elif type(module).__name__ == "LoraHadaModule":
        module.w1a = torch.nn.Parameter(module.w1a.to(devices.device, dtype=torch.float))
        module.w1b = torch.nn.Parameter(module.w1b.to(devices.device, dtype=torch.float))
        module.w2a = torch.nn.Parameter(module.w2a.to(devices.device, dtype=torch.float))
        module.w2b = torch.nn.Parameter(module.w2b.to(devices.device, dtype=torch.float))
        
        if module.t1 is not None:
            module.t1 = torch.nn.Parameter(module.t1.to(devices.device, dtype=torch.float))

        if module.t2 is not None:
            module.t2 = torch.nn.Parameter(module.t2.to(devices.device, dtype=torch.float))
        
    elif type(module).__name__ == "FullModule":
        module.weight = torch.nn.Parameter(module.weight.to(devices.device, dtype=torch.float))
    
    if hasattr(module, 'bias') and module.bias != None:
        module.bias = torch.nn.Parameter(module.bias.to(devices.device, dtype=torch.float))

def restoremodel(p):
    model = p.sd_model
    for name,module in model.named_modules():
        if hasattr(module, "lora_weights_backup"):
            if module.lora_weights_backup is not None:
                if isinstance(module, torch.nn.MultiheadAttention):
                    module.in_proj_weight.copy_(module.lora_weights_backup[0])
                    module.out_proj.weight.copy_(module.lora_weights_backup[1])
                else:
                    module.weight.copy_(module.lora_weights_backup)
                module.lora_weights_backup = None
