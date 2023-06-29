import math
from pprint import pprint
import ldm.modules.attention as atm
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode, Resize  # Mask.

TOKENSCON = 77
TOKENS = 75

pmasks = {}
pmaskshw =[]
pmasksf = {}
maskready = False

def main_forward(module,x,context,mask,divide,isvanilla = False,userpp = False,tokens=[],width = 64,height = 64,step = 0):
    
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

    ## for prompt mode make basemask from attention maps

    global pmaskshw,pmasks

    if userpp and step > 0:
        for b in range(attn.shape[0] // 8):
            if pmaskshw == []:
                pmaskshw = [(height,width)]
            elif height > pmaskshw[0][0]:
                del pmaskshw
                pmaskshw = [(height,width)]
                global maskready
                maskready = False
                pmasks = {}
            elif (height,width) not in pmaskshw:
                pmaskshw.append((height,width))

            for t in tokens: 
                add = attn[8*b:8*(b+1),:,t[0]:t[0]+len(t)]**1.2
                add = torch.sum(add,dim = 2)
                t = f"{t}-{b}"         
                if t not in pmasks:
                    pmasks[t] = add * 0
                else:
                    if pmasks[t].shape[1] != add.shape[1]:
                        add = add.view(8,height,width)
                        add = F.resize(add,pmaskshw[0])
                        add = add.reshape_as(pmasks[t])

                    pmasks[t] = pmasks[t] + add

    out = atm.einsum('b i j, b j d -> b i d', attn, v)
    out = atm.rearrange(out, '(b h) n d -> b n (h d)', h=h)
    out = module.to_out(out)

    return out

def hook_forwards(self, root_module: torch.nn.Module, remove=False):
    for name, module in root_module.named_modules():
        if "attn2" in name and module.__class__.__name__ == "CrossAttention":
            module.forward = hook_forward(self, module)
            if remove:
                del module.forward

################################################################################
##### Attention mode 

def hook_forward(self, module):
    def forward(x, context=None, mask=None):
        if self.debug :
            print("input : ", x.size())
            print("tokens : ", context.size())
            print("module : ", module.lora_layer_name)

        # if not self.calced and "diffusion_model_output_blocks_11_1_transformer_blocks_0_attn2" == module.lora_layer_name:
        #     self.calced = True

        height = self.h
        width = self.w

        def hr_cheker(n):
            return (n != 0) and (n & (n - 1) == 0)

        if not hr_cheker(height * width // x.size()[1]) and self.hr:
            height = self.hr_h
            width = self.hr_w

        xs = x.size()[1]
        scale = round(math.sqrt(height * width / xs))

        dsh = round(height / scale)
        dsw = round(width / scale)
        ha, wa = xs % dsh, xs % dsw
        if ha == 0:
            dsw = int(xs / dsh)
        elif wa == 0:
            dsh = int(xs / dsw)

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
            
            i = 0
            outb = None
            if self.usebase:
                context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
                # SBM Controlnet sends extra conds at the end of context, apply it to all regions.
                cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                if cnet_ext > 0:
                    context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)
                    
                i = i + 1 + self.basebreak
                out = main_forward(module, x, context, mask, divide, self.isvanilla,userpp =True,step = self.step)

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
                    out = main_forward(module, x, context, mask, divide, self.isvanilla,userpp = self.pn, step = self.step)
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
                        if self.debug : print(f"{int(dsh*drow.st) + addout}:{int(dsh*drow.ed)},{int(dsw*dcell.st) + addin}:{int(dsw*dcell.ed)}")
                        if self.usebase : 
                            # outb_t = outb[:,:,int(dsw*drow.st):int(dsw*drow.ed),:].clone()
                            outb_t = outb[:,int(dsh*drow.st) + addout:int(dsh*drow.ed),
                                            int(dsw*dcell.st) + addin:int(dsw*dcell.ed),:].clone()
                            out = out * (1 - dcell.base) + outb_t * dcell.base
                    elif "Vertical" in self.mode: # Cols are the outer list, rows are cells.
                        out = out[:,int(dsh*dcell.st) + addin:int(dsh*dcell.ed),
                                  int(dsw*drow.st) + addout:int(dsw*drow.ed),:]
                        if self.debug : print(f"{int(dsh*dcell.st) + addin}:{int(dsh*dcell.ed)}-{int(dsw*drow.st) + addout}:{int(dsw*drow.ed)}")
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

        def masksepcalc(x,contexts,mask,pn,divide):
            xs = x.size()[1]
            (dsh,dsw) = split_dims(xs, height, width, debug = self.debug)

            tll = self.pt if pn else self.nt
            
            # Base forward.
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

        def promptsepcalc(x, contexts, mask, pn,divide):
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

                userpp = self.pn and i == 0

                out = main_forward(module, x, context, mask, divide, self.isvanilla, userpp = userpp, width = dsw, height = dsh, tokens = self.pe, step = self.step)

                if (len(self.nt) == 1 and not pn) or (self.modep and self.calcmode =="Latent"):
                    if self.debug : print("return out for NP or Latent")
                    return out

                if self.debug : print(scale, dsh, dsw, dsh * dsw, x.size()[1])

                if i == 0:
                    outb = out.clone()
                    continue
                else:
                    h_states.append(out)

            if self.debug:
                for h in h_states :
                    print(f"divided : {h.size()}")
                print(pmaskshw)

            if pmaskshw == []:
                return outb

            ox = outb.clone() if self.ex else outb * 0

            if self.debug:
                print(pmaskshw,maskready,(dsh,dsw) in pmaskshw and maskready,len(pmasksf),len(h_states))

            if (dsh,dsw) in pmaskshw and maskready:
                depth = pmaskshw.index((dsh,dsw))
                maskb = None
                for masks , state in zip(pmasksf.values(),h_states):
                    mask = masks[depth]
                    masked = torch.multiply(state, mask)
                    if self.ex:
                        ox = torch.where(masked !=0 , masked, ox)
                    else:
                        ox = ox + masked
                    maskb = maskb + mask if maskb is not None else mask
                maskb = 1 - maskb
                if not self.ex : ox = ox + torch.multiply(outb, maskb)
                return ox
            else:
                return outb

        if self.eq:
            if self.debug : print("same token size and divisions")
            if self.indmaskmode:
                ox = masksepcalc(x, contexts, mask, True, 1)
            elif self.cells:
                ox = matsepcalc(x, contexts, mask, True, 1)
            else:
                ox = promptsepcalc(x, contexts, mask, True, 1)
        elif x.size()[0] == 1 * self.batch_size:
            if self.debug : print("different tokens size")
            if self.indmaskmode:
                ox = masksepcalc(x, contexts, mask, self.pn, 1)
            elif self.cells:
                ox = matsepcalc(x, contexts, mask, self.pn, 1)
            else:
                ox = promptsepcalc(x, contexts, mask, self.pn, 1)
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
            elif self.cells:
                # SBM I think division may have been an incorrect patch.
                # But I'm not sure, haven't tested beyond DDIM / PLMS.
                opx = matsepcalc(px, conp, mask, True, 2)
                onx = matsepcalc(nx, conn, mask, False, 2)
            else:
                opx = promptsepcalc(px, conp, mask, True, 2)
                onx = promptsepcalc(nx, conn, mask, False, 2)
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

def repeat_div(x,y):
    """Imitates dimension halving common in convolution operations.
    
    This is a pretty big assumption of the model,
    but then if some model doesn't work like that it will be easy to spot.
    """
    while y > 0:
        x = math.ceil(x / 2)
        y = y - 1
    return x

#################################################################################
##### for Prompt mode
def reset_pmasks(self): # init parameters in every batch
    global pmasks, pmaskshw, pmasksf, maskready
    self.step = 0
    pmasks = {}
    pmaskshw =[]
    pmasksf = {}
    maskready = False
    self.x = None
    return self


def savepmasks(self,processed):
    print(len(pmasks),len(self.pe),len(self.th))
    for mask ,th in zip(pmasks.values(),self.th):
        img, _ , _= makepmask(mask, self.h, self.w,th, self.step)
        processed.images.append(img)
    return processed


def makepmask(mask, h, w, th, step, bratio = 1): # make masks from attention cache return [for preview, for attention, for Latent]
    th = th - step * 0.005
    bratio = 1 - bratio
    mask = torch.mean(mask,dim=0)
    mask = mask / mask.max().item()
    mask = torch.where(mask > th ,1,0)
    mask = mask.float()
    mask = mask.view(1,pmaskshw[0][0],pmaskshw[0][1]) 
    img = torchvision.transforms.functional.to_pil_image(mask)
    img = img.resize((w,h))
    mask = F.resize(mask,(h,w),interpolation=F.InterpolationMode.NEAREST)
    lmask = mask
    mask = mask.reshape(h*w)
    mask = torch.where(mask > 0.1 ,1,0)
    return img,mask * bratio , lmask * bratio
