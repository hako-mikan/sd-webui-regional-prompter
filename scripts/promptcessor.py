import os.path
import re
from pprint import pprint
import modules # SBM Apparently, basedir only works when accessed directly.
from modules import paths, scripts, shared, extra_networks

"""
The module handles all prompt related parsing and editing, preprocessing stage:
Keywords, commons, bases, pos, neg.
"""

# SBM Keywords and delimiters for region breaks, following matlab rules.
# BREAK keyword are passed through when ADDX are used.  
KEYROW = "ADDROW"
KEYCOL = "ADDCOL"
KEYBASE = "ADDBASE"
KEYCOMM = "ADDCOMM"
KEYBRK = "BREAK" # Generic key for attention context splitting.
KEYPROMPT = "ADDP"
KEYDUPE = "ADDUPE" # Add number after to select specific region, def is previous.
KEYAND = "AND" # Generic key for latent context splitting.
ALLKEYS = [KEYCOMM, KEYBASE, KEYROW, KEYCOL, KEYPROMPT]
REGDUPE = KEYDUPE + "(-?\d*)" # dupe + optional number.
TOKENS = 75
ENDTOKEN = 49407

fidentity = lambda x: x
fspace = lambda x: " {} ".format(x)

def lange(l):
    return range(len(l))
    
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

def replace_keys(prompt, tokey = KEYBRK):
    """Changes all regional keys to target generic key (break / and).
    
    Be careful that this doesn't erase comm/base, split them first.
    """
    for key in ALLKEYS:
        prompt = prompt.replace(key, tokey)
    return prompt

def chop_combase(prompt, usecom, usebase):
    """Retrieves base and common clauses from prompt.
    
    Base and common keys can now be placed in any order.
     If using breaks only, common should be placed before base.
     In any case both should be at the beginning of prompt.        
    """
    idxc = -1
    idxce = -1
    if usecom:
        try:
            idxc = prompt.index(KEYCOMM)
            idxce = idxc + len(KEYCOMM)
        except ValueError:
            try:
                idxc = prompt.index(KEYBRK)
                idxce = idxc + len(KEYBRK)
            except:
                print("Common not found.")
    
    idxb = -1
    idxbe = -1
    if usebase:
        try:
            idxb = prompt.index(KEYBASE)
            idxbe = idxb + len(KEYBASE) 
        except ValueError:
            try:
                idxb = prompt.index(KEYBRK, idxc + 1) + 1
                idxbe = idxb + len(KEYBRK)
            except:
                print("Base not found.")
    
    lidx = [(idxc, idxce), (idxb, idxbe)]
    bfirst = 0
    if idxc > idxb:
        lidx = [lidx[-1], lidx[0]]
        bfirst = 1
    
    lsegs = []
    prv = 0
    for (st, ed) in lidx:
        if st >= 0:
            lsegs.append(prompt[prv:st])
            prv = ed
        else:
            lsegs.append("")
    scomm = lsegs[bfirst]
    sbase = lsegs[1 - bfirst]
    smain = prompt[prv:]
    
    return smain, scomm, sbase 

def add_common(prompt, scomm, key = KEYBRK):
    """Add common clause to every clause in prompt delimited by key.
    
    Formerly comaddr.
    """
    if len(scomm) == 0: # No common, no change.
        return prompt
    
    # Split to list of clauses.
    lprompt = prompt.split(key)
    # Merge the list back with the common clauses and break.
    scmbrk = fspace(key) + scomm + ", "
    prompt = scmbrk.join(lprompt)
    # All that remains is prepending to the first clause.
    prompt = scomm + ", " + prompt
    return prompt

def dupe_index(prompt):
    """Returns index of dupe if exists, or null if not dupe.
    
    Currently negatives are relative region.
    """
    ldupe = re.findall(REGDUPE, prompt, re.IGNORECASE)
    if len(ldupe) == 0:
        return None
    if len(ldupe[0]) == 0:
        return -1
    else:
        return int(ldupe[0])

def ref_tokens(prompt, usebase, key = KEYBRK):
    """Count tokens per clause and create references to each region.
        
    Regions containing the dupe keyword shall copy the reference of other regions:
     by default the previous one, or a specific region number, or any relative prior region.
    Prompt must be cleaned ahead of all loras (pos at least).
    Returns all refs, but cannot remove dupes - missing loras.
    Creep: Support better dupes - any direction of relative reference, by mask number.
    """
    pt = [] # Token range.
    ppt = [] # Token count, info.
    padd = 0 # Total token counter.
    pdp = dict() # Contains reference type clauses.
    
    ppl = prompt.split(key) 
    for i,pp in enumerate(ppl):
        didx = dupe_index(pp)
        if didx is None:
            tokens, tokensnum = shared.sd_model.cond_stage_model.tokenize_line(pp)
            pt.append([padd, tokensnum // TOKENS + 1 + padd])
            ppt.append(tokensnum)
            padd = tokensnum // TOKENS + 1 + padd
        else:
            # Reference clauses are discarded from the prompt.
            # Base is always at index 0, this changes the indexing access.
            if didx >= 0:
                pdp[i + 1 - int(usebase)] = didx
            else:
                pdp[i + 1 - int(usebase)] = i + 1 - int(usebase) + didx
            pt.append([0,1]) # Placeholders.
            ppt.append([-1])
    
    # Copy all reference clauses' token positions.
    # Currently a simple left to right mapping, not recursive.
    for k in sorted(pdp.keys()):
        pt[k - 1 + int(usebase)] = pt[pdp[k] - 1 + int(usebase)]
        ppt[k - 1 + int(usebase)] = ppt[pdp[k] - 1 + int(usebase)]
        
    return padd, pt, ppt, pdp

def clean_dupes(prompt, pdp, usebase, key = KEYBRK):
    """Clean dupes from *original* prompt based on dupe detection.
    
    Beware, loras will be removed indiscriminately!
    This is partly intended since lora repetition is rather buggy.
    """
    ppl = prompt.split(key)
    ppl = [p for i,p in enumerate(ppl) if i + 1 - int(usebase) not in pdp]
    return key.join(ppl)

# SBM CONT: That is really bad, should not read the changes from the ratios. Prompt first!
def keyconverter(prompt, aratios, mode, usecom, usebase):
    """Converts BREAKS to ADDCOMM/ADDBASE/ADDCOL/ADDROW.
    
    Not necessary, when breaks and adds are mixed only the adds count.
    When only breaks are used, matrix mode will match them to ratios anyway.
    """
    keychanger = makeimgtmp(aratios,mode,usecom,usebase,inprocess = True)
    keychanger = keychanger[:-1]
    #print(keychanger,p.prompt)
    for change in keychanger:
        if change == KEYCOMM and KEYCOMM in prompt: continue
        if change == KEYBASE and KEYBASE in prompt: continue
        prompt= prompt.replace(KEYBRK,change,1)

    return prompt

def extend_clauses(prompt, cnt, idx = -1, key = KEYBRK):
    """Creates additional dupe clauses for pos / neg as necessary
    
    Pos will be created in break only matrix, when too many aratios values, 
    or in mask when there are too many masks.
    For neg, the anchor is always pos, with the exception of base usage.
    Rather than perform the calculation, which is mode specific,
     this receives the number as parm and merely dupes the clause.
    """
    if cnt <= 0:
        return prompt
    pextra = [prompt] + [KEYDUPE + str(idx)] * cnt
    return fspace(key).join(pextra)

class RegionPrompt():
    """Script extender. Assortment of prompt processing functions.
    
    Abstract class / interface.
    """
    def __init__(self):
        """Init.
        
        Usecom / ncom / base / nbase and other parameters should be given.
        """
        # self.usecom = False
        # self.usencom = False
        # self.usebase = False
        # self.usenbase = False
        self.comprompt = None
        self.comnegprompt = None
        self.baseprompt = None
        self.basenegprompt = None
        
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
    
        if KEYBASE in p.negative_prompt:
            self.usenbase = True
    
        if KEYPROMPT in p.prompt.upper():
            self.mode = "Prompt"
            # p.replace(KEYPROMPT,KEYBRK)
    
        # return self, p
    
    def separate_externals(self, p):
        """Remove common, base clauses everywhere, place them in props till later.
        
        Has flags set as well as cleaning up prompt breaks.
        Disables flags if no relevant clauses were found.
        """
        # self, p = self.flagfromkeys(p)
        self.flagfromkeys(p)
        
        (p.prompt, self.comprompt, self.baseprompt) = chop_combase(
            p.prompt, self.usecom, self.usebase)
        if len(self.comprompt) == 0:
            self.usecom = False
        if len(self.baseprompt) == 0:
            self.usebase = False
        (p.negative_prompt, self.comnegprompt, self.basenegprompt) = chop_combase(
            p.negative_prompt, self.usencom, self.usenbase)
        if len(self.comnegprompt) == 0:
            self.usencom = False
        if len(self.basenegprompt) == 0:
            self.usenbase = False
        
        # if self.usecom and KEYCOMM in p.prompt:
        #     self.comprompt = p.prompt.split(KEYCOMM,1)[0]
        #     p.prompt = p.prompt.split(KEYCOMM,1)[1]
        # elif self.usecom and KEYBRK in p.prompt:
        #     self.comprompt = p.prompt.split(KEYBRK,1)[0]
        #     p.prompt = p.prompt.split(KEYBRK,1)[1]
        # else:
        #     self.comprompt = ""
        # if self.usencom and KEYCOMM in p.negative_prompt:
        #     self.comnegprompt = p.negative_prompt.split(KEYCOMM,1)[0]
        #     p.negative_prompt = p.negative_prompt.split(KEYCOMM,1)[1]
        # elif self.usencom and KEYBRK in p.negative_prompt:
        #     self.comnegprompt = p.negative_prompt.split(KEYBRK,1)[0]
        #     p.negative_prompt = p.negative_prompt.split(KEYBRK,1)[1]
        # else:
        #     self.comnegprompt = ""

        # return self, p
    
    # def separate_externals_v2(self,p):
    #     if self.usecom and KEYCOMM in p.prompt:
    #         comprompt = p.prompt.split(KEYCOMM,1)[0]
    #         p.prompt = p.prompt.split(KEYCOMM,1)[1]
    #     elif self.usecom and KEYBRK in p.prompt:
    #         comprompt = p.prompt.split(KEYBRK,1)[0]
    #         p.prompt = p.prompt.split(KEYBRK,1)[1]
    #     if self.usencom and KEYCOMM in p.negative_prompt:
    #         comnegprompt = p.negative_prompt.split(KEYCOMM,1)[0]
    #         p.negative_prompt = p.negative_prompt.split(KEYCOMM,1)[1]
    #     elif self.usencom and KEYBRK in p.negative_prompt:
    #         comnegprompt = p.negative_prompt.split(KEYBRK,1)[0]
    #         p.negative_prompt = p.negative_prompt.split(KEYBRK,1)[1]
    #     # The addrow/addcol syntax is better, cannot detect regular breaks without it.
    #     # In any case, the preferred method will anchor the L2 structure. 
    #     if (KEYBASE in p.prompt.upper()): # Designated base.
    #         self.usebase = True
    #         baseprompt = p.prompt.split(KEYBASE,1)[0]
    #         mainprompt = p.prompt.split(KEYBASE,1)[1] 
    #         self.basebreak = fcountbrk(baseprompt)
    #     elif usebase: # Get base by first break as usual.
    #         baseprompt = p.prompt.split(KEYBRK,1)[0]
    #         mainprompt = p.prompt.split(KEYBRK,1)[1]
    #     else:
    #         baseprompt = ""
    #         mainprompt = p.prompt
    #     # Neg base.
    #     if (KEYBASE in p.negative_prompt.upper()):
    #         self.usenbase = True
    #         p.negative_prompt = p.negative_prompt.split(KEYBASE,1)[1]
    #     elif usenbase: # Get base by first break as usual.
    #         p.negative_prompt = p.negative_prompt.split(KEYBRK,1)[1]

    def replace_allp_keys(self, p, key = KEYBRK):
        """"Replace all separators to key.
        
        In p.all_prompt and p.all_negative_prompt.
        Formerly keyreplacer.
        """
        p.prompt = replace_keys(p.prompt, key)
        for i,_ in enumerate(p.all_prompts):
            p.all_prompts[i] = replace_keys(p.all_prompts[i], key)
        
        p.negative_prompt = replace_keys(p.negative_prompt, key)
        for i,_ in enumerate(p.all_negative_prompts):
            p.all_negative_prompts[i] = replace_keys(p.all_negative_prompts[i], key)
        
        # return p
    
    def apply_comms(self, p, key = KEYBRK):
        """Prepends common clauses to pos/neg prompts.
        
        To be used after replacement.
        Originally, commons were added per pr in all prompt;
        I find this unnecessary, don't know a use case where all differs from main prompt.
        Formerly commondealer + anddealer.
        """
        all_prompts = []
        all_negative_prompts = []
        
        if key != KEYBRK:
            self.change_key(p, KEYBRK, key)
        # if self.usecom:
        self.prompt = p.prompt = add_common(p.prompt, self.comprompt, key)
        for pr in p.all_prompts:
            # all_prompts.append(add_common(pr, self.comprompt))
            all_prompts.append(p.prompt)
        p.all_prompts = all_prompts
    
        # if self.usencom:
        self.negative_prompt = p.negative_prompt = add_common(p.negative_prompt, self.comnegprompt, key)
        for pr in p.all_negative_prompts:
            # all_negative_prompts.append(add_common(pr, self.comnegprompt))
            all_negative_prompts.append(p.negative_prompt)
        p.all_negative_prompts = all_negative_prompts
        
        self.divide = p.prompt.count(key) + 1 # SBM Dunno what it's used for. Info?
        
    def rejoin_bases(self, p, key = KEYBRK):
        """Add base clauses back to main prompts with break.
        
        """
        if self.usebase:
            p.prompt = self.baseprompt + fspace(key) + p.prompt
        if self.usenbase:
            p.negative_prompt = self.basenegprompt + fspace(key) + p.negative_prompt

    def tokenise(self, p, key = KEYBRK):
        """Count tokens per clause and create references to each region.
        
        Regions containing the dupe keyword shall copy the reference of other regions:
         by default the previous one, or a specific region number, or any relative prior region.
        Formerly tokendealer.
        Creep: Support better dupes - any direction of relative reference, by mask number.
        """
        ptext, _ = extra_networks.parse_prompt(p.all_prompts[0]) # SBM From update_token_counter.
        ntext = p.all_negative_prompts[0]
        ppl = ptext.split(key)
        targets = [p.rsplit(",", 1)[-1] for p in ppl[1:]] # Key is last comma section in all clauses but the first.
        tt = []
        
        paddp, pt, ppt, pdp = ref_tokens(ptext, self.usebase, key)
        for i, prompt in enumerate(p.all_prompts): # Remove dupes from all prompts.
            p.all_prompts[i] = clean_dupes(prompt, pdp, self.usebase, key)
        
        # Prompt mode detects usage of certain tokens 
        if self.modep:
            for target in targets:
                ptokens, tokensnum = shared.sd_model.cond_stage_model.tokenize_line(ppl[0])
                ttokens, _ = shared.sd_model.cond_stage_model.tokenize_line(target)
    
                i = 1
                tlist = []
                while ttokens[0].tokens[i] != ENDTOKEN:
                    for (j, maintok) in enumerate(ptokens): # SBM Long prompt.
                        if ttokens[0].tokens[i] in maintok.tokens:
                            tlist.append(maintok.tokens.index(ttokens[0].tokens[i]) + TOKENS * j)
                    i += 1
                if tlist != [] : tt.append(tlist)
        
        paddn, nt, pnt, pdn = ref_tokens(ntext, self.usenbase, key)
        for i, prompt in enumerate(p.all_negative_prompts): # Remove dupes from all negs.
            p.all_negative_prompts[i] = clean_dupes(prompt, pdn, self.usenbase, key)
    
        # SBM It might be best to do away with eq altogether.
        # Pos / neg are always separable, it's just slower but guarantees correct results.
        # For now, if usebases don't match it is disabled.
        self.eq = (paddp == paddn and self.usebase == self.usenbase)
        self.pt = pt
        self.nt = nt
        self.pe = tt
        self.ppt = ppt
        self.pnt = pnt
        self.pdp = pdp # SBM Add dupe mappings for latent mode, which doesn't handle context.
        self.pdn = pdn
        
        # return self

    def change_key(self, p, fromkey, tokey):
        """Simple key replacement in all prompts.
        
        """
        p.prompt = p.prompt.replace(fromkey, tokey)
        for i, prompt in enumerate(p.all_prompts):
            p.all_prompts[i] = prompt.replace(fromkey,tokey)
        
        p.negative_prompt = p.negative_prompt.replace(fromkey, tokey)
        for i, prompt in enumerate(p.all_negative_prompts):
            p.all_negative_prompts[i] = prompt.replace(fromkey, tokey)
            
    def merge_negs(self, p, key = KEYBRK):
        """Merge negative prompts into positive.
        
        This is part of a workaround for latent mode.
        The other part is matching clauses in denoising.
        """
        if self.calcmode == "Latent" and self.lateneg:
            self.prompt = p.prompt = p.prompt + fspace(key) + p.negative_prompt
            p.negative_prompt = "spam"
            for i,_ in enumerate(p.all_prompts):
                p.all_prompts[i] = p.all_prompts[i] + fspace(key) + p.all_negative_prompts[i]
                p.all_negative_prompts[i] = "spam"
