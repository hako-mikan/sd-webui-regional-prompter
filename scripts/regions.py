import colorsys  # Polygon regions.
from pprint import pprint
import cv2  # Polygon regions.
import gradio as gr
import numpy as np
import PIL
import torch
from modules import devices


def lange(l):
    return range(len(l))

# SBM Keywords and delimiters for region breaks, following matlab rules.
# BREAK keyword is now passed through,  
KEYROW = "ADDROW"
KEYCOL = "ADDCOL"
KEYBASE = "ADDBASE"
KEYCOMM = "ADDCOMM"
KEYBRK = "BREAK"
KEYPROMPT = "ADDP"
DELIMROW = ";"
DELIMCOL = ","
DELIMROW = ";"
DELIMCOL = ","
MCOLOUR = 256
NLN = "\n"
DKEYINOUT = { # Out/in, horizontal/vertical or row/col first.
("out",False): KEYROW,
("in",False): KEYCOL,
("out",True): KEYCOL,
("in",True): KEYROW,
}

ALLKEYS = [KEYCOMM,KEYROW, KEYCOL, KEYBASE, KEYPROMPT]

fidentity = lambda x: x
ffloatd = lambda c: (lambda x: floatdef(x,c))
fcolourise = lambda: np.random.randint(0,MCOLOUR,size = 3)
fspace = lambda x: " {} ".format(x)

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


def isfloat(t):
    try:
        float(t)
        return True
    except Exception:
        return False

def ratiosdealer(aratios2,aratios2r):
    aratios2 = list_percentify(aratios2)
    aratios2 = list_cumsum(aratios2)
    aratios2 = list_rangify(aratios2)
    aratios2r = list_percentify(aratios2r)
    aratios2r = list_cumsum(aratios2r)
    aratios2r = list_rangify(aratios2r)
    return aratios2,aratios2r


def makeimgtmp(aratios,mode,usecom,usebase,inprocess = False):
    indflip = (mode == "Vertical")
    if DELIMROW not in aratios: # Commas only - interpret as 1d.
        aratios2 = split_l2(aratios, DELIMROW, DELIMCOL, fmap = ffloatd(1), indflip = False)
        aratios2r = [1]
    else:
        (aratios2r,aratios2) = split_l2(aratios, DELIMROW, DELIMCOL, 
                                        indsingles = True, fmap = ffloatd(1), indflip = indflip)
    # Change all splitters to breaks.
    (aratios2,aratios2r) = ratiosdealer(aratios2,aratios2r)
    
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

    if inprocess:
        changer = template.split(NLN)
        changer = [l.strip() for l in changer]
        return changer
    
    return img,gr.update(value = template)

################################################################
##### matrix
fcountbrk = lambda x: x.count(KEYBRK)
fint = lambda x: int(x)

def matrixdealer(self, p, aratios, bratios, mode, usebase, comprompt,comnegprompt):
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
    #p.all_prompts = [p.prompt] * len(p.all_prompts)

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
    # if comprompt is not None : 
    #     p.prompt = comprompt + fspace(KEYBRK) + p.prompt
    #     for i in lange(p.all_prompts):
    #         p.all_prompts[i] = comprompt + fspace(KEYBRK) + p.all_prompts[i]
    # if comnegprompt is not None :
    #     p.negative_prompt = comnegprompt + fspace(KEYBRK) + p.negative_prompt
    #     for i in lange(p.all_negative_prompts):
    #         p.all_negative_prompts[i] = comnegprompt + fspace(KEYBRK) + p.all_negative_prompts[i]
    p = keyreplacer(p)
    return self, p

################################################################
##### inpaint

"""
SBM mod: Mask polygon region.
- Basically a version of inpainting, where polygon outlines are drawn and added to a coloured image.
- Colours from the image are picked apart for masks corresponding to regions.
- In new mask mode, masks are stored instead of aratios, and applied to each region forward.
- Mask can be uploaded (alpha, no save), and standard colours are detected from it.
- Uncoloured regions default to the first colour detected;
  however, if base mode is used, instead base will be applied to the remainder at 100% strength.
  I think this makes it far more useful. At 0 strength, it will apply ONLY to said regions.
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

# SBM In mask mode, grabs each mask from coloured mask image.
# If there's no base, remainder goes to first mask.
# If there's a base, it will receive its own remainder mask, applied at 100%.
def inpaintmaskdealer(self, p, bratios, usebase, polymask, comprompt, comnegprompt):
    if self.usecom and KEYCOMM in p.prompt:
        p.prompt = p.prompt.split(KEYCOMM,1)[1]
    elif self.usecom and KEYBRK in p.prompt:
        p.prompt = p.prompt.split(KEYBRK,1)[1]
        
    if self.usencom and KEYCOMM in p.negative_prompt:
        p.negative_prompt = p.negative_prompt.split(KEYCOMM,1)[1]
    elif self.usencom and KEYBRK in p.negative_prompt:
        p.negative_prompt = p.negative_prompt.split(KEYBRK,1)[1]
        
    if (KEYBASE in p.prompt.upper()): # Designated base.
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
    
    # Convert all keys to breaks, and expand neg to fit.
    mainprompt = mainprompt.replace(KEYROW,KEYBRK) # Cont: Should be case insensitive.
    mainprompt = mainprompt.replace(KEYCOL,KEYBRK)
    
    # Simulated region anchroing for base weights.
    breaks = mainprompt.count(KEYBRK) + int(self.usebase)
    self.bratios = split_l2(bratios, DELIMROW, DELIMCOL, fmap = ffloatd(0),
                            basestruct = [[0] * (breaks + 1)], indflip = False)
    
    p.prompt = mainprompt
    if self.usebase:
        p.prompt = baseprompt + fspace(KEYBRK) + p.prompt
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
    return self, p


def keyreplacer(p):
    '''
    replace all separators to BREAK
    p.all_prompt and p.all_negative_prompt
    '''
    for key in ALLKEYS:
        for i in lange(p.all_prompts):
            p.all_prompts[i]= p.all_prompts[i].replace(key,KEYBRK)
        
        for i in lange(p.all_negative_prompts):
            p.all_negative_prompts[i] = p.all_negative_prompts[i].replace(key,KEYBRK)

    return p