#from random import choices # SBM Unused?
from typing import Union
from matplotlib.style import available
import PIL
import inspect
#import copy # SBM Unused?
#from regex import R # SBM Unused?
import torch
# import csv # SBM Replaced.
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
import json # Presets.
from torchvision.transforms import Resize, InterpolationMode # Mask.
import cv2 # Polygon regions.
import colorsys # Polygon regions.

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

orig_batch_cond_uncond = shared.batch_cond_uncond
orig_lora_forward = None
orig_lora_apply_weights = None
orig_lora_Linear_forward = None
orig_lora_Conv2d_forward = None
lactive = False
labug =False

PRESETS =[
    ["Vertical-3", "Vertical",'1,1,1',"",False,False,False,"Attention",False,"0","0"],
    ["Horizontal-3", "Horizontal",'1,1,1',"",False,False,False,"Attention",False,"0","0"],
    ["Horizontal-7", "Horizontal",'1,1,1,1,1,1,1',"0.2",True,False,False,"Attention",False,"0","0"],
    ["Twod-2-1", "Horizontal",'1,2,3;1,1',"0.2",False,False,False,"Attention",False,"0","0"],
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
NLN = "\n"
#MATMODE = "Matrix"
TOKENSCON = 77
TOKENS = 75
MCOLOUR = 256
ATTNSCALE = 8 # Initial image compression in attention layers.
DKEYINOUT = { # Out/in, horizontal/vertical or row/col first.
("out",False): KEYROW,
("in",False): KEYCOL,
("out",True): KEYCOL,
("in",True): KEYROW,
}
fidentity = lambda x: x
fcountbrk = lambda x: x.count(KEYBRK)
#ffloat = lambda x: float(x)
fint = lambda x: int(x)
fspace = lambda x: " {} ".format(x)
fcolourise = lambda: np.random.randint(0,MCOLOUR,size = 3)
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
]

class RegionCell():
    """Cell used to split a layer to single prompts."""
    def __init__(self, st, ed, base, breaks):
        """Range with start and end values, base weight and breaks count for context splitting."""
        self.st = st # Range for the cell (cols only).
        self.ed = ed
        self.base = base # How much of the base prompt is applied (difference).
        self.breaks = breaks # How many unrelated breaks the prompt contains.
        
    def __repr__(self):
        """Debug print."""
        return "({:.2f}:{:.2f})".format(self.st,self.ed) 
        
class RegionRow():
    """Row containing cell refs and its own ratio range."""
    def __init__(self, st, ed, cols):
        """Range with start and end values, base weight and breaks count for context splitting."""
        self.st = st # Range for the row.
        self.ed = ed
        self.cols = cols # List of cells.
        
    def __repr__(self):
        """Debug print."""
        return "Outer ({:.2f}:{:.2f}), contains {}".format(self.st, self.ed, self.cols) + NLN

def floatdef(x, vdef):
    """Attempt conversion to float, use default value on error.
    
    Mainly for empty ratios, double commas.
    """
    try:
        return float(x)
    except ValueError:
        print("'{}' is not a number, converted to {}".format(x,vdef))
        return vdef
    
ffloatd = lambda c: (lambda x: floatdef(x,c))

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
    The fmap function is applied to each cell before insertion to L2;
    if it fails, a default value is used.
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

def split_dims(xs, height, width, **kwargs):
    """Split an attention layer dimension to height + width.
    
    Originally, the estimate was dsh = sqrt(hw_ratio*xs),
    rounding to the nearest value. But this proved inaccurate.
    What seems to be the actual operation is as follows:
    - Divide h,w by 8, rounding DOWN. 
      (However, webui forces dims to be divisible by 8 unless set explicitly.)
    - For every new layer (of 4), divide both by 2 and round UP (then back up)
    - Multiply h*w to yield xs.
    There is no inverse function to this set of operations,
    so instead we mimic them sans the multiplication part with orig h+w.
    The only alternative is brute forcing integer guesses,
    which might be inaccurate too.
    No known checkpoints follow a different system of layering,
    but it's theoretically possible. Please report if encountered.
    """
    # OLD METHOD.
    # scale = round(math.sqrt(height*width/xs))
    # dsh = round_dim(height, scale)
    # dsw = round_dim(width, scale) 
    scale = math.ceil(math.log2(math.sqrt(height * width / xs)))
    dsh = repeat_div(height,scale)
    dsw = repeat_div(width,scale)
    if kwargs.get("debug",False) : print(scale,dsh,dsw,dsh*dsw,xs)
    
    return dsh,dsw

def main_forward(module,x,context,mask,divide,isvanilla = False):
    
    # Forward.
    h = module.heads
    if isvanilla: # SBM Ddim / plms have the context split ahead along with x.
        pass
    else: # SBM I think divide may be redundant.
        h = h // divide
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
    except Exception:
        return False
    
"""
SBM mod: Mask polygon region.
- Basically a version of inpainting, where polygon outlines are drawn and added to a coloured image.
- Colours from the image are picked apart for masks corresponding to regions.
- In new mask mode, masks are stored instead of aratios, and applied to each region forward.
- Mask can be uploaded (alpha, no save), and standard colours are detected from it.
- Uncoloured regions default to the first colour detected;
  however, if base mode is used, instead base will be applied to the remainder at 100% strength.
  I think this makes it far more useful. At 0 strength, it will apply ONLY to said regions.
Some sketch code shamelessly copied from controlnet, thanks. 
"""

POLYFACTOR = 1.5 # Small lines are detected as shapes.
LCOLOUR = set() # List of used colours. CONT: Empty when image is deleted, interpret from colours on upload. 
COLREG = None # Computed colour regions. Array. Extended whenever a new colour is requested. 
IDIM = 512
CBLACK = 255
VARIANT = 0 # Ensures that the sketch canvas is actually refreshed.

def generate_unique_colors(n):
    """Generate n visually distinct colors as a list of RGB tuples.
    
    Uses the hue of hsv, with balanced saturation & value.
    """
    hsv_colors = [(x*1.0/n, 0.5, 0.5) for x in range(n)]
    rgb_colors = [tuple(int(i * CBLACK) for i in colorsys.hsv_to_rgb(*hsv)) for hsv in hsv_colors]
    return rgb_colors

def deterministic_colours(n, lcol = None):
    """Generate n visually distinct & consistent colours as a list of RGB tuples.
    
    Uses the hue of hsv, with balanced saturation & value.
    Goes around the cyclical 0-256 and picks each /2 value for every round.
    Continuation rules: If pcyv != ccyv in next round, then we don't care.
    If pcyv == ccyv, we want to get the cval + delta of last elem.
    If lcol > n, will return it as is.
    """
    if n <= 0:
        return None
    pcyc = -1
    cval = 0
    if lcol is None:
        st = 0
    elif n <= len(lcol):
        # return lcol[:n] # Truncating the list is accurate, but pointless.
        return lcol
    else:
        st = len(lcol)
        if st > 0:
            pcyc = np.ceil(np.log2(st))
            # This is erroneous on st=2^n, but we don't care.
            dlt = 1 / (2 ** pcyc)
            cval = dlt + 2 * dlt * (st % (2 ** (pcyc - 1)) - 1)

    lhsv = []
    for i in range(st,n):
        ccyc = np.ceil(np.log2(i + 1))
        if ccyc == 0: # First col = 0.
            cval = 0
            pcyc = ccyc
        elif pcyc != ccyc: # New cycle, start from the half point between 0 and first point.
            dlt = 1 / (2 ** ccyc)
            cval = dlt
            pcyc = ccyc
        else:
            cval = cval + 2 * dlt # Jumps over existing vals.
        lhsv.append(cval)
    lhsv = [(v, 0.5, 0.5) for v in lhsv] # Hsv conversion only works 0:1.
    lrgb = [colorsys.hsv_to_rgb(*hsv) for hsv in lhsv]
    lrgb = (np.array(lrgb) * (CBLACK + 1)).astype(np.uint8) # Convert to colour uints.
    lrgb = lrgb.reshape(-1, 3)
    if lcol is not None:
        lrgb = np.concatenate([lcol, lrgb])
    return lrgb

def detect_polygons(img,num):
    global LCOLOUR
    global COLREG
    global VARIANT
    
    LCOLOUR.add(num)
    
    # I dunno why, but mask has a 4th colour channel, which contains nothing. Alpha?
    if VARIANT != 0:
        out = img["image"][:-VARIANT,:-VARIANT,:3]
        img = img["mask"][:-VARIANT,:-VARIANT,:3]
    else:
        out = img["image"][:,:,:3]
        img = img["mask"][:,:,:3]
    
    # Convert the binary image to grayscale
    if img is None:
        img = np.zeros([IDIM,IDIM,3],dtype = np.uint8) + CBLACK # Stupid cv.
    if out is None:
        out = np.zeros_like(img) + CBLACK # Stupid cv.
    bimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Find contours in the image
    # Must reverse colours, otherwise draws an outer box (0->255). Dunno why gradio uses 255 for white anyway. 
    contours, hierarchy = cv2.findContours(bimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #img2 = np.zeros_like(img) + 255 # Fresh image.
    img2 = out # Update current image.

    COLREG = deterministic_colours(int(num) + 1, COLREG)
    color = COLREG[int(num),:]
    # Loop through each contour and detect polygons
    for cnt in contours:
        # Approximate the contour to a polygon
        approx = cv2.approxPolyDP(cnt, 0.0001 * cv2.arcLength(cnt, True), True)

        # If the polygon has 3 or more sides and is fully enclosed, fill it with a random color
        # if len(approx) >= 3: # BAD test.
        if cv2.contourArea(cnt) > cv2.arcLength(cnt, True) * POLYFACTOR: # Better, still messes up on large brush.
            #SBM BUGGY, prevents contours from . cv2.pointPolygonTest(approx, (approx[0][0][0], approx[0][0][1]), False) >= 0:
                                                          
                                                    
            
            # Draw the polygon on the image with a new random color
            color = [int(v) for v in color] # Opencv is dumb / C based and can't handle an int64 array.
            #cv2.drawContours(img2, [approx], 0, color = color) # Only outer sketch.
            cv2.fillPoly(img2,[approx],color = color)
                

                                                               
                                        

    # Convert the grayscale image back to RGB
    #img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB) # Converting to grayscale is dumb.
    
    skimg = create_canvas(img2.shape[0], img2.shape[1], indnew = False)
    if VARIANT != 0:
        skimg[:-VARIANT,:-VARIANT,:] = img2
    else:
        skimg[:,:,:] = img2
    
    print("Region sketch size", skimg.shape)
    return skimg, num + 1 if num + 1 <= CBLACK else num

def detect_mask(img, num, mult = CBLACK):
    """Extract specific colour and return mask.
    
    Multiplier for correct display.
    """
    color = deterministic_colours(int(num) + 1)[-1]
    color = color.reshape([1,1,3])
    mask = ((img["image"] == color).all(-1)) * mult
    return mask

def create_canvas(h, w, indnew = True):
    """New region sketch area.
    
    Small variant value is added (and ignored later) due to gradio refresh bug.
    Meant to be used only to start over or when the image dims change.
    """
    global VARIANT
    global LCOLOUR
    VARIANT = 1 - VARIANT
    if indnew:
        LCOLOUR = set()
    vret =  np.zeros(shape = (h + VARIANT, w + VARIANT, 3), dtype = np.uint8) + CBLACK
    return vret

class Script(modules.scripts.Script):
    def __init__(self):
        self.mode = ""
        self.calcmode = ""
        self.indmaskmode = False
        self.indexperiment = False
        self.w = 0
        self.h = 0
        self.usebase = False
        self.usecom = False
        self.usencom = False
        self.aratios = []
        self.bratios = []
        self.regmasks = None
        self.regbase = None
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
        self.anded = False
        self.lora_applied = False

    def title(self):
        return "Regional Prompter"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    infotext_fields = None
    """if set in ui(), this is a list of pairs of gradio component + text; the text will be used when
    parsing infotext to set the value for the component; see ui.py's txt2img_paste_fields for an example
    """

    paste_field_names = []
    """if set in ui(), this is a list of names of infotext fields; the fields will be sent through the
    various "Send to <X>" buttons when clicked
    """

    def ui(self, is_img2img):
        path_root = scripts.basedir()
        filepath = os.path.join(path_root,"scripts", "regional_prompter_presets.json")

        presets = []

        presets = loadpresets(filepath)

        with gr.Accordion("Regional Prompter", open=False):
            with gr.Row():
                active = gr.Checkbox(value=False, label="Active",interactive=True,elem_id="RP_active")
            with gr.Row():
                mode = gr.Radio(label="Divide mode", choices=["Horizontal", "Vertical", "Mask"], value="Horizontal",  type="value", interactive=True)
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
            settings = [mode, ratios, baseratios, usebase, usecom, usencom, calcmode, nchangeand, lnter, lnur]
        
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
        
        def makeimgtmp(aratios,mode,usecom,usebase):
            indflip = (mode == "Vertical")
            if DELIMROW not in aratios: # Commas only - interpret as 1d.
                aratios2 = split_l2(aratios, DELIMROW, DELIMCOL, fmap = ffloatd(1), indflip = False)
                aratios2r = [1]
            else:
                (aratios2r,aratios2) = split_l2(aratios, DELIMROW, DELIMCOL, 
                                                indsingles = True, fmap = ffloatd(1), indflip = indflip)
            # Change all splitters to breaks.
            aratios2 = list_percentify(aratios2)
            aratios2 = list_cumsum(aratios2)
            aratios2 = list_rangify(aratios2)
            aratios2r = list_percentify(aratios2r)
            aratios2r = list_cumsum(aratios2r)
            aratios2r = list_rangify(aratios2r)
            
            h = w = 128
            fx = np.zeros((h,w, 3), np.uint8)
            # Base image is coloured according to region divisions, roughly.
            for (i,ocell) in enumerate(aratios2r):
                for icell in aratios2[i]:
                    # SBM Creep: Colour by delta so that distinction is more reliable.
                    if not indflip:
                        fx[int(h*ocell[0]):int(h*ocell[1]),int(w*icell[0]):int(w*icell[1]),:] = fcolourise()
                    else:
                        fx[int(h*icell[0]):int(h*icell[1]),int(w*ocell[0]):int(w*ocell[1]),:] = fcolourise()
            img = PIL.Image.fromarray(fx)
            draw = PIL.ImageDraw.Draw(img)
            c = 0
            def coldealer(col):
                if sum(col) > 380:return "black"
                else:return "white"
            # Add region counters at the top left corner, coloured according to hue.
            for (i,ocell) in enumerate(aratios2r):
                for icell in aratios2[i]: 
                    if not indflip:
                        draw.text((int(w*icell[0]),int(h*ocell[0])),f"{c}",coldealer(fx[int(h*ocell[0]),int(w*icell[0])]))
                    else: 
                        draw.text((int(w*ocell[0]),int(h*icell[0])),f"{c}",coldealer(fx[int(h*icell[0]),int(w*ocell[0])]))
                    c += 1
            
            # Create ROW+COL template from regions.
            txtkey = fspace(DKEYINOUT[("in", indflip)]) + NLN  
            lkeys = [txtkey.join([""] * len(cell)) for cell in aratios2]
            txtkey = fspace(DKEYINOUT[("out", indflip)]) + NLN
            template = txtkey.join(lkeys) 
            if usebase:
                template = fspace(KEYBASE) + NLN + template
            if usecom:
                template = fspace(KEYCOMM) + NLN + template
            return img,gr.update(value = template)

        maketemp.click(fn=makeimgtmp, inputs =[ratios,mode,usecom,usebase],outputs = [areasimg,template])
        applypresets.click(fn=setpreset, inputs = availablepresets, outputs=settings)
        savesets.click(fn=savepresets, inputs = [presetname,*settings],outputs=availablepresets)
                
        return [active, debug, mode, ratios, baseratios, usebase, usecom, usencom, calcmode, nchangeand, lnter, lnur, polymask]

    def process(self, p, active, debug, mode, aratios, bratios, usebase, usecom, usencom, calcmode, nchangeand, lnter, lnur, polymask):
        if active:
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
                    })

            savepresets("lastrun",mode, aratios,bratios, usebase, usecom, usencom, calcmode, nchangeand, lnter, lnur)
            self.__init__()
            self.active = True
            self.mode = mode
            comprompt = comnegprompt = None
            # SBM ddim / plms detection.
            self.isvanilla = p.sampler_name in ["DDIM", "PLMS", "UniPC"]

            self.orig_all_prompts = p.all_prompts[:]
            self.orig_all_negative_prompts = p.all_negative_prompts[:]

            if not nchangeand and "AND" in p.prompt.upper():
                p.prompt = p.prompt.replace("AND",KEYBRK)
                for i in lange(p.all_prompts):
                    p.all_prompts[i] = p.all_prompts[i].replace("AND",KEYBRK)
                self.anded = True

            if (KEYROW in p.prompt.upper() or KEYCOL in p.prompt.upper() or DELIMROW in aratios):
                self.indexperiment = True
            elif KEYBRK not in p.prompt.upper():
                self.active = False
                unloader(self,p)
                return
            if mode == "Mask":
                self.indmaskmode = True
            self.w = p.width
            self.h = p.height
            if self.h % ATTNSCALE != 0 or self.w % ATTNSCALE != 0:
                # Testing shows a round down occurs in model.
                print("Warning: Nonstandard height / width.")
                self.h = self.h - self.h % ATTNSCALE
                self.w = self.w - self.w % ATTNSCALE
                
            self.batch_size = p.batch_size
            
            self.calcmode = calcmode

            self.debug = debug
            self.usebase = usebase
            self.usecom = usecom
            if KEYCOMM in p.prompt: # Automatic common toggle.
                self.usecom = True
            self.usencom = usencom
            if KEYCOMM in p.negative_prompt: # Automatic common toggle.
                self.usencom = True

            if hasattr(p,"enable_hr"): # Img2img doesn't have it.
                self.hr = p.enable_hr
                self.hr_w = (p.hr_resize_x if p.hr_resize_x > p.width else p.width * p.hr_scale)
                self.hr_h = (p.hr_resize_y if p.hr_resize_y > p.height else p.height * p.hr_scale)

            # SBM In mask mode, grabs each mask from coloured mask image.
            # If there's no base, remainder goes to first mask.
            # If there's a base, it will receive its own remainder mask, applied at 100%.
            if self.indmaskmode:
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
                    
                if (KEYBASE in p.prompt.upper()): # Designated base.
                    self.usebase = True
                    baseprompt = p.prompt.split(KEYBASE,1)[0]
                    mainprompt = p.prompt.split(KEYBASE,1)[1] 
                    #self.basebreak = fcountbrk(baseprompt) # No support for inner breaks currently.
                elif usebase: # Get base by first break as usual.
                    baseprompt = p.prompt.split(KEYBRK,1)[0]
                    mainprompt = p.prompt.split(KEYBRK,1)[1]
                else:
                    baseprompt = ""
                    mainprompt = p.prompt
                    
                # Prep masks.
                self.regmasks = []
                tm = None
                for c in sorted(LCOLOUR):
                    m = detect_mask(polymask, c, 1)
                    if VARIANT != 0:
                        m = m[:-VARIANT,:-VARIANT]
                    if m.any():
                        if tm is None:
                            tm = np.zeros_like(m) # First mask is ignored deliberately.
                            if self.usebase: # In base mode, base gets the outer regions.
                                tm = tm + m
                        else:
                            tm = tm + m
                        m = m.reshape([1, *m.shape]).astype(np.float16)
                        t = torch.from_numpy(m).to(devices.device) 
                        self.regmasks.append(t)
                # First mask applies to all unmasked regions.
                m = 1 - tm
                m = m.reshape([1, *m.shape]).astype(np.float16)
                t = torch.from_numpy(m).to(devices.device)
                if self.usebase:
                    self.regbase = t
                else:
                    self.regbase = None
                    self.regmasks[0] = t
                # t = torch.from_numpy(np.zeros([1,512,512], dtype = np.float16)).to(devices.device)
                # self.regmasks.append(t)
                # t = torch.from_numpy(np.ones([1,512,512], dtype = np.float16)).to(devices.device)
                # self.regmasks.append(t)
                
                breaks = mainprompt.count(KEYBRK) + int(self.usebase)
                self.bratios = split_l2(bratios, DELIMROW, DELIMCOL, fmap = ffloatd(0),
                                        basestruct = [[0] * (breaks + 1)], indflip = False)
                # Convert all keys to breaks, and expand neg to fit.
                mainprompt = mainprompt.replace(KEYROW,KEYBRK) # Cont: Should be case insensitive.
                mainprompt = mainprompt.replace(KEYCOL,KEYBRK)
                p.prompt = mainprompt
                if self.usebase:
                    p.prompt = baseprompt + fspace(KEYBRK) + p.prompt
                p.all_prompts = [p.prompt] * len(p.all_prompts)
                npr = p.negative_prompt
                npr.replace(KEYROW,KEYBRK)
                npr.replace(KEYCOL,KEYBRK)
                npr = npr.split(KEYBRK)
                nbreaks = len(npr) - 1
                if breaks >= nbreaks: # Repeating the first neg as in orig code.
                    npr.extend([npr[0]] * (breaks - nbreaks))
                else: # Cut off the excess negs.
                    npr = npr[0:breaks + 1]
                for i ,n in enumerate(npr):
                    if n.isspace() or n =="":
                        npr[i] = ","
                # p.negative_prompt = fspace(KEYBRK).join(npr)
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
            
            # SBM In matrix mode, the ratios are broken up 
            elif self.indexperiment:
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
                    if (DELIMROW not in aratios
                    and (KEYROW in mainprompt.upper()) != (KEYCOL in mainprompt.upper())):
                        # By popular demand, 1d integrated into 2d.
                        # This works by either adding a single row value (inner),
                        # or setting flip to the reverse (outer).
                        # Only applies when using just ADDROW / ADDCOL keys, and commas in ratio.
                        indflip2 = False
                        if (KEYROW in mainprompt.upper()) == indflip:
                            aratios = "1" + DELIMCOL + aratios
                        else:
                            indflip2 = True
                        (aratios2r,aratios2) = split_l2(aratios, DELIMROW, DELIMCOL, indsingles = True,
                                            fmap = ffloatd(1), basestruct = lbreaks,
                                            indflip = indflip2)
                    else: # Standard ratios, split to rows and cols.
                        (aratios2r,aratios2) = split_l2(aratios, DELIMROW, DELIMCOL, indsingles = True,
                                                        fmap = ffloatd(1), basestruct = lbreaks, indflip = indflip)
                    # More like "bweights", applied per cell only.
                    bratios2 = split_l2(bratios, DELIMROW, DELIMCOL, fmap = ffloatd(0), basestruct = lbreaks, indflip = indflip)
                else:
                    breaks = mainprompt.count(KEYBRK) + int(self.usebase)
                    (aratios2r,aratios2) = split_l2(aratios, DELIMROW, DELIMCOL, indsingles = True, fmap = ffloatd(1), indflip = indflip)
                    # Cannot determine which breaks matter.
                    lbreaks = split_l2("0", KEYROW, KEYCOL, fmap = fint, basestruct = aratios2, indflip = indflip)
                    bratios2 = split_l2(bratios, DELIMROW, DELIMCOL, fmap = ffloatd(0), basestruct = lbreaks, indflip = indflip)
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
                npr = p.negative_prompt
                npr.replace(KEYROW,KEYBRK)
                npr.replace(KEYCOL,KEYBRK)
                npr = np.split(KEYBRK)
                nbreaks = len(npr) - 1
                if breaks >= nbreaks: # Repeating the first neg as in orig code.
                    npr.extend([npr[0]] * (breaks - nbreaks))
                else: # Cut off the excess negs.
                    npr = npr[0:breaks + 1]
                for i ,n in enumerate(npr):
                    if n.isspace() or n =="":
                        npr[i] = ","
                # p.negative_prompt = fspace(KEYBRK).join(npr)
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
                shared.batch_cond_uncond = orig_batch_cond_uncond 
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
                print(f"mode : {self.calcmode}\ndivide : {mode}\nusebase : {self.usebase}")
                print(f"base ratios : {self.bratios}\nusecommon : {self.usecom}\nusenegcom : {self.usencom}\nuse 2D : {self.indexperiment}")
                print(f"divide : {self.divide}\neq : {self.eq}\n")
                print(f"ratios : {self.aratios}\n")
        else:
            unloader(self,p)
        return p

    def process_batch(self, p, active, debug, mode, aratios, bratios, usebase, usecom, usencom, calcmode,nchangeand, lnter, lnur, polymask, **kwargs):
        global lactive,labug
        if self.lora_applied: # SBM Don't override orig twice on batch calls.
            pass
        elif self.active and calcmode =="Latent":
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
            self.lora_applied = True
            lora_namer(self, p, lnter, lnur)
        else:
            lactive = False


    # TODO: Should remove usebase, usecom, usencom - grabbed from self value.
    def postprocess_image(self, p, pp, active, debug, mode, aratios, bratios, usebase, usecom, usencom, calcmode, nchangeand, lnter, lnur, polymask):
        if not self.active:
            return p
        if self.usecom or self.indexperiment or self.anded:
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
            # SBM Stale version workaround.
            if hasattr(params,"text_cond"):
                ct =  params.text_cond.clone()
            areas = xt.shape[0] // self.batch_size -1

            for a in range(areas):
                for b in range(self.batch_size):
                    params.x[b+a*self.batch_size] = xt[a + b * areas]
                    params.image_cond[b+a*self.batch_size] = ict[a + b * areas]
                    params.sigma[b+a*self.batch_size] = st[a + b * areas]
                    # SBM Stale version workaround.
                    if hasattr(params,"text_cond"):
                        params.text_cond[b+a*self.batch_size] = ct[a + b * areas]

    def denoised_callback(self, params: CFGDenoisedParams):
        if lactive:
            x = params.x
            batch = self.batch_size
            # x.shape = [batch_size, C, H // 8, W // 8]
            indrebuild = False
            if self.filters == [] :
                indrebuild = True
            elif self.filters[0].size() != x[0].size():
                indrebuild = True
            if indrebuild:
                if self.indmaskmode:
                    masks = (self.regmasks,self.regbase)
                else:
                    masks = self.aratios
                self.filters = makefilters(x.shape[1], x.shape[2], x.shape[3],masks,
                                           self.mode,self.usebase,self.bratios,self.indexperiment,self.indmaskmode)
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
            xs = x.size()[1]
            (dsh,dsw) = split_dims(xs, height, width, debug = self.debug)
            
            if "Horizontal" in self.mode: # Map columns / rows first to outer / inner.
                dsout = dsw
                dsin = dsh
            elif "Vertical" in self.mode:
                dsout = dsh
                dsin = dsw

            tll = self.pt if pn else self.nt
            
            # Base forward.
            cad = 0 if self.usebase else 1 # 1 * self.usebase is shorter.
            i = 0
            outb = None
            if self.usebase:
                context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
                # SBM Controlnet sends extra conds at the end of context, apply it to all regions.
                cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                if cnet_ext > 0:
                    context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)
                    
                i = i + 1 + self.basebreak
                out = main_forward(module, x, context, mask, divide, self.isvanilla)

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
                    # SBM Controlnet sends extra conds at the end of context, apply it to all regions.
                    cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                    if cnet_ext > 0:
                        context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)
                        
                    if self.debug : print(f"tokens : {tll[i][0]*TOKENSCON}-{tll[i][1]*TOKENSCON}")
                    i = i + 1 + dcell.breaks
                    # if i >= contexts.size()[1]: 
                    #     indlast = True
                    out = main_forward(module, x, context, mask, divide, self.isvanilla)
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
                    if self.debug : print(f"sumin:{sumin},sumout:{sumout},dsh:{dsh},dsw:{dsw}")
            
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
                # SBM Controlnet sends extra conds at the end of context, apply it to all regions.
                cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                if cnet_ext > 0:
                    context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)
                
                if self.debug : print(f"tokens : {tl[0]*TOKENSCON}-{tl[1]*TOKENSCON}")

                if self.usebase:
                    if i != 0:
                        area = self.aratios[i - 1]
                        bweight = self.bratios[i - 1]
                else:
                    area = self.aratios[i]

                out = main_forward(module, x, context, mask, divide, self.isvanilla)

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
        
        def masksepcalc(x,contexts,mask,pn,divide):
            xs = x.size()[1]
            (dsh,dsw) = split_dims(xs, height, width, debug = self.debug)

            tll = self.pt if pn else self.nt
            
            # Base forward.
            cad = 0 if self.usebase else 1 # 1 * self.usebase is shorter.
            i = 0
            outb = None
            if self.usebase:
                context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
                # SBM Controlnet sends extra conds at the end of context, apply it to all regions.
                cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                if cnet_ext > 0:
                    context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)
                    
                i = i + 1
                out = main_forward(module, x, context, mask, divide, self.isvanilla)

                if len(self.nt) == 1 and not pn:
                    if self.debug : print("return out for NP")
                    return out
                # if self.usebase:
                outb = out.clone()
                outb = outb.reshape(outb.size()[0], dsh, dsw, outb.size()[2]) 

            if self.debug : print(f"tokens : {tll},pn : {pn}")
            
            ox = torch.zeros_like(x)
            ox = ox.reshape(ox.shape[0], dsh, dsw, ox.shape[2])
            ftrans = Resize((dsh, dsw), interpolation = InterpolationMode("nearest"))
            for rmask in self.regmasks:
                # Need to delay mask tensoring so it's on the correct gpu.
                # Dunno if caching masks would be an improvement.
                if self.usebase:
                    bweight = self.bratios[0][i - 1]
                # Resize mask to current dims.
                # Since it's a mask, we prefer a binary value, nearest is the only option.
                rmask2 = ftrans(rmask.reshape([1, *rmask.shape])) # Requires dimensions N,C,{d}.
                rmask2 = rmask2.reshape(1, dsh, dsw, 1)
                
                # Grabs a set of tokens depending on number of unrelated breaks.
                context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
                # SBM Controlnet sends extra conds at the end of context, apply it to all regions.
                cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                if cnet_ext > 0:
                    context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)
                    
                if self.debug : print(f"tokens : {tll[i][0]*TOKENSCON}-{tll[i][1]*TOKENSCON}")
                i = i + 1
                # if i >= contexts.size()[1]: 
                #     indlast = True
                out = main_forward(module, x, context, mask, divide, self.isvanilla)
                if len(self.nt) == 1 and not pn:
                    if self.debug : print("return out for NP")
                    return out
                    
                out = out.reshape(out.size()[0], dsh, dsw, out.size()[2]) # convert to main shape.
                if self.usebase:
                    out = out * (1 - bweight) + outb * bweight
                ox = ox + out * rmask2

            if self.usebase:
                rmask = self.regbase
                rmask2 = ftrans(rmask.reshape([1, *rmask.shape])) # Requires dimensions N,C,{d}.
                rmask2 = rmask2.reshape(1, dsh, dsw, 1)
                ox = ox + outb * rmask2
            ox = ox.reshape(x.size()[0],x.size()[1],x.size()[2]) # Restore to 3d source.  
            return ox

        if self.eq:
            if self.debug : print("same token size and divisions")
            if self.indmaskmode:
                ox = masksepcalc(x, contexts, mask, True, 1)
            elif self.indexperiment:
                ox = matsepcalc(x, contexts, mask, True, 1)
            else:
                ox = regsepcalc(x, contexts, mask, True, 1)
        elif x.size()[0] == 1 * self.batch_size:
            if self.debug : print("different tokens size")
            if self.indmaskmode:
                ox = masksepcalc(x, contexts, mask, self.pn, 1)
            elif self.indexperiment:
                ox = matsepcalc(x, contexts, mask, self.pn, 1)
            else:
                ox = regsepcalc(x, contexts, mask, self.pn, 1)
        else:
            if self.debug : print("same token size and different divisions")
            # SBM You get 2 layers of x, context for pos/neg.
            # Each should be forwarded separately, pairing them up together.
            if self.isvanilla: # SBM Ddim reverses cond/uncond.
                nx, px = x.chunk(2)
                conn,conp = contexts.chunk(2)
            else:
                px, nx = x.chunk(2)
                conp,conn = contexts.chunk(2)
            if self.indmaskmode:
                opx = masksepcalc(px, conp, mask, True, 2)
                onx = masksepcalc(nx, conn, mask, False, 2)
            elif self.indexperiment:
                # SBM I think division may have been an incorrect patch.
                # But I'm not sure, haven't tested beyond DDIM / PLMS.
                opx = matsepcalc(px, conp, mask, True, 2)
                onx = matsepcalc(nx, conn, mask, False, 2)
                # opx = matsepcalc(px, contexts, mask, True, 2)
                # onx = matsepcalc(nx, contexts, mask, False, 2)
            else:
                opx = regsepcalc(px, conp, mask, True, 2)
                onx = regsepcalc(nx, conn, mask, False, 2)
                # opx = regsepcalc(px, contexts, mask, True, 2)
                # onx = regsepcalc(nx, contexts, mask, False, 2)
            if self.isvanilla: # SBM Ddim reverses cond/uncond.
                ox = torch.cat([onx, opx])
            else:
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
    aratios = [floatdef(a,1) for a in aratios.split(",")]
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
        self.bratios = [floatdef(b,0) for b in bratios.split(",")]
    except Exception:
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
        shared.batch_cond_uncond = orig_batch_cond_uncond 
    global lactive
    lactive = False
    self.active = False
    self.lora_applied = False

#############################################################
##### Preset save and load

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

def savepresets(*settings):
    # NAME must come first.
    name = settings[0]
    path_root = scripts.basedir()
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
    

######################################################
##### Latent Method

def lora_namer(self,p, lnter, lnur):
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
    u_llist = [d.copy() for d in llist[1:]]
    u_llist.append(llist[0].copy())
    regioner.te_llist = llist
    regioner.u_llist = u_llist
    regioner.ndeleter(lnter, lnur)
    if self.debug:
        print(regioner.te_llist)
        print(regioner.u_llist)


def makefilters(c,h,w,masks,mode,usebase,bratios,xy,indmask = None):
    if indmask is not None:
        (regmasks, regbase) = masks
        
    filters = []
    x =  torch.zeros(c, h, w).to(devices.device)
    if usebase:
        x0 = torch.zeros(c, h, w).to(devices.device)
    i=0
    if indmask:
        ftrans = Resize((h, w), interpolation = InterpolationMode("nearest"))
        for rmask, bratio in zip(regmasks,bratios[0]):
            # Resize mask to current dims.
            # Since it's a mask, we prefer a binary value, nearest is the only option.
            rmask2 = ftrans(rmask.reshape([1, *rmask.shape])) # Requires dimensions N,C,{d}.
            rmask2 = rmask2.reshape([1, h, w])
            fx = x.clone()
            if usebase:
                fx[:,:,:] = fx + rmask2 * (1 - bratio)
                x0[:,:,:] = x0 + rmask2 * bratio
            else:
                fx[:,:,:] = fx + rmask2 * 1
            filters.append(fx)
            
        if usebase: # Add base to x0.
            rmask = regbase
            rmask2 = ftrans(rmask.reshape([1, *rmask.shape])) # Requires dimensions N,C,{d}.
            rmask2 = rmask2.reshape([1, h, w])
            x0 = x0 + rmask2
    elif xy:
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
                        fx[:,int(h*dcell.st):int(h*dcell.ed),int(w*drow.st):int(w*drow.ed)] = 1  
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

    def ndeleter(self, lnter, lnur):
        for key in self.te_llist[0].keys():
            self.te_llist[0][key] = floatdef(lnter, 0)
        for key in self.u_llist[-1].keys():
            self.u_llist[-1][key] = floatdef(lnur, 0)

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
        if hasattr(module,"up_model") :
            module.up_model.weight = torch.nn.Parameter(module.up_model.weight.to(devices.device, dtype = torch.float))
            module.down_model.weight = torch.nn.Parameter(module.down_model.weight.to(devices.device, dtype=torch.float))
        else:
            module.up.weight = torch.nn.Parameter(module.up.weight.to(devices.device, dtype = torch.float))
            if hasattr(module.down, "weight"):
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
