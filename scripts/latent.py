from difflib import restore
import random
from pprint import pprint
from typing import Union
import torch
from modules import devices, extra_networks, shared
from modules.script_callbacks import CFGDenoisedParams, CFGDenoiserParams
from torchvision.transforms import InterpolationMode, Resize  # Mask.
import scripts.attention as att
from scripts.regions import floatdef

orig_lora_forward = None
orig_lora_apply_weights = None
orig_lora_Linear_forward = None
orig_lora_Conv2d_forward = None
lactive = False
labug =False
pactive = False

def setloradevice(self):
    regioner.__init__()
    import lora
    if self.debug : print("change LoRA device for new lora")
    if hasattr(lora,"lora_apply_weights"): # for new LoRA applying
        for l in lora.loaded_loras:
            l.name = l.name + "added_by_regional_prompter" + str(random.random())
            for key in l.modules.keys():
                changethedevice(l.modules[key])

def setuploras(self,p):
    import lora
    global orig_lora_forward,orig_lora_apply_weights,lactive, orig_lora_Linear_forward, orig_lora_Conv2d_forward, lactive, labug
    lactive = True
    labug = self.debug

    if hasattr(lora,"lora_apply_weights"): # for new LoRA applying
        if self.debug : print("hijack lora_apply_weights")
        orig_lora_apply_weights = lora.lora_apply_weights
        orig_lora_Linear_forward = torch.nn.Linear.forward
        orig_lora_Conv2d_forward = torch.nn.Conv2d.forward
        lora.lora_apply_weights = lora_apply_weights
        torch.nn.Linear.forward = lora_Linear_forward
        torch.nn.Conv2d.forward = lora_Conv2d_forward

    elif hasattr(lora,"lora_forward"):
        if self.debug : print("hijack lora_forward")
        orig_lora_forward = lora.lora_forward
        lora.lora_forward = lora_forward

    return self

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

def denoiser_callback_s(self, params: CFGDenoiserParams):
    if self.modep:  # in Prompt mode, make masks from sum of attension maps
        self.step = params.sampling_step
        self.calced = False
        if self.pe == [] : return

        if self.calcmode == "Latent":
            self.filters  = []
            for b in range(self.batch_size):
                if len(att.pmaskshw) < 3: return
                allmask = []
                basemask = None
                for t, th, bratio in zip(self.pe, self.th, self.bratios):
                    key = f"{t}-{b}"
                    _, _, mask = att.makepmask(att.pmasks[key], params.x.shape[2], params.x.shape[3], th, self.step, bratio = bratio)
                    mask = mask.repeat(params.x.shape[1],1,1)
                    basemask = 1 - mask if basemask is None else basemask - mask
                    allmask.append(mask)
                    if self.ex:
                        for l in range(len(allmask) - 1):
                            mt = allmask[l] - mask
                            allmask[l] = torch.where(mt > 0, 1,0)
                basemask = torch.where(basemask > 0 , 1,0)
                allmask.insert(0,basemask)
                self.filters.extend(allmask)
                self.neg_filters.extend([1- f for f in allmask])

            att.maskready = True
        
        else:    
            for t, th, bratio in zip(self.pe, self.th, self.bratios):
                if len(att.pmaskshw) < 3: return
                allmask = []
                for hw in att.pmaskshw:
                    masks = None
                    for b in range(self.batch_size):
                        key = f"{t}-{b}"
                        _, mask, _ = att.makepmask(att.pmasks[key], hw[0], hw[1], th, self.step, bratio = bratio)
                        mask = mask.unsqueeze(0).unsqueeze(-1)
                        masks = mask if b ==0 else torch.cat((masks,mask),dim=0)
                    allmask.append(mask)     
                att.pmasksf[key] = allmask
        
            att.maskready = True

    if self.lactive or self.lpactive:
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

def denoised_callback_s(self, params: CFGDenoisedParams):
    if self.lactive or self.lpactive:
        x = params.x
        xt = params.x.clone()
        batch = self.batch_size
        areas = xt.shape[0] // batch -1

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

        if not self.lpactive:
            if indrebuild:
                if self.indmaskmode:
                    masks = (self.regmasks,self.regbase)
                else:
                    masks = self.aratios  #makefilters(c,h,w,masks,mode,usebase,bratios,indmask = None)
                self.filters = makefilters(x.shape[1], x.shape[2], x.shape[3],masks, self.mode, self.usebase, self.bratios, self.indmaskmode)
                self.filters = [f for f in self.filters]*batch
                self.neg_filters = [1- f for f in self.filters]
        else:
            if not att.maskready:
                self.filters = [1,*[0 for a in range(areas - 1)]] * batch
                self.neg_filters = [1 - f for f in self.filters]

        if self.debug : print("filterlength : ",len(self.filters))

        if labug : 
            for i in range(params.x.shape[0]):
                print(torch.max(params.x[i]))

        for b in range(batch):
            for a in range(areas):
                x[a + b * areas] = xt[b+a*batch]

        for b in range(batch):
            for a in range(areas) :
                if self.debug : print(f"x = {x.size()}i = {a + b*areas}, cond = {a + b*areas}, uncon = {x.size()[0]+(b-batch)}")
                x[a + b*areas, :, :, :] =  x[a + b*areas, :, :, :] * self.filters[a + b*areas] + x[x.size()[0]+(b-batch), :, :, :] * self.neg_filters[a + b*areas]

######################################################
##### Latent Method

def lora_namer(self, p, lnter, lnur):
    ldict = {}
    lorder = [] # Loras call order for matching with u/te lists.
    import lora as loraclass
    for lora in loraclass.loaded_loras:
        ldict[lora.name] = lora.multiplier
    
    subprompts = self.current_prompts[0].split("AND")
    llist =[ldict.copy() for i in range(len(subprompts)+1)]
    for i, prompt in enumerate(subprompts):
        _, extranets = extra_networks.parse_prompts([prompt])
        calledloras = extranets["lora"]

        names = ""
        tdict = {}

        for called in calledloras:
            if called.items[0] not in lorder:
                lorder.append(called.items[0])
            names = names + called.items[0]
            tdict[called.items[0]] = called.items[1]

        for key in llist[i].keys():
            shin_key = key.split("added_by_regional_prompter")[0]
            shin_key = shin_key.split("added_by_lora_block_weight")[0]
            if shin_key in names:
                llist[i+1][key] = float(tdict[shin_key])
            else:
                llist[i+1][key] = 0
                
    global regioner
    regioner.__init__()
    u_llist = [d.copy() for d in llist[1:]]
    u_llist.append(llist[0].copy())
    regioner.te_llist = llist
    regioner.u_llist = u_llist
    regioner.ndeleter(lnter, lnur, lorder)
    if self.debug:
        print("LoRA regioner : TE list",regioner.te_llist)
        print("LoRA regioner : U list",regioner.u_llist)

def makefilters(c,h,w,masks,mode,usebase,bratios,indmask):
    if indmask:
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
    else:
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

    def expand_del(self, val, lorder):
        """Broadcast single / comma separated val to lora list. 
        
        """
        lval = val.split(",")
        if len(lval) > len(lorder):
            lval = lval[:len(lorder)]
        lval = [floatdef(v, 0) for v in lval]
        if len(lval) < len(lorder): # Propagate difference.
            lval.extend([lval[-1]] * (len(lorder) - len(lval)))
        return lval

    def ndeleter(self, lnter, lnur, lorder = None):
        """Multiply global weights by 0:1 factor.
        
        Can be any value, negative too, but doesn't help much.
        """
        if lorder is None:
            lkeys = self.te_llist[0].keys()
        else:
            lkeys = lorder
        lnter = self.expand_del(lnter, lkeys)
        for (key, val) in zip(lkeys, lnter):
            self.te_llist[0][key] *= val
        if lorder is None:
            lkeys = self.u_llist[-1].keys()
        else:
            lkeys = lorder
        lnur = self.expand_del(lnur, lkeys)
        for (key, val) in zip(lkeys, lnur):
            self.u_llist[-1][key] *= val

    def te_start(self):
        self.mlist = self.te_llist[self.te_count % len(self.te_llist)]
        self.te_count += 1
        import lora
        for i in range(len(lora.loaded_loras)):
            lora.loaded_loras[i].multiplier = self.mlist[lora.loaded_loras[i].name]

    def u_start(self):
        if labug : print("u_count",self.u_count ,"u_count '%' divide",  self.u_count % len(self.u_llist))
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
            if "9" in lora_layer_name and ("_attn1_to_q" in lora_layer_name or "self_attn_q_proj" in lora_layer_name): print(lora_m.multiplier,lora_m.name,lora_layer_name,lora_m)
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


LORAANDSOON = {
    "IA3Module" : "w",
    "LoraKronModule" : "w1",
    "LycoKronModule" : "w1",
}


def changethedevice(module):
    ltype = type(module).__name__
    if ltype == "LoraUpDownModule" or ltype == "LycoUpDownModule" :
        if hasattr(module,"up_model") :
            module.up_model.weight = torch.nn.Parameter(module.up_model.weight.to(devices.device, dtype = torch.float))
            module.down_model.weight = torch.nn.Parameter(module.down_model.weight.to(devices.device, dtype=torch.float))
        else:
            module.up.weight = torch.nn.Parameter(module.up.weight.to(devices.device, dtype = torch.float))
            if hasattr(module.down, "weight"):
                module.down.weight = torch.nn.Parameter(module.down.weight.to(devices.device, dtype=torch.float))
        
    elif ltype == "LoraHadaModule" or ltype == "LycoHadaModule":
        module.w1a = torch.nn.Parameter(module.w1a.to(devices.device, dtype=torch.float))
        module.w1b = torch.nn.Parameter(module.w1b.to(devices.device, dtype=torch.float))
        module.w2a = torch.nn.Parameter(module.w2a.to(devices.device, dtype=torch.float))
        module.w2b = torch.nn.Parameter(module.w2b.to(devices.device, dtype=torch.float))
        
        if module.t1 is not None:
            module.t1 = torch.nn.Parameter(module.t1.to(devices.device, dtype=torch.float))

        if module.t2 is not None:
            module.t2 = torch.nn.Parameter(module.t2.to(devices.device, dtype=torch.float))
        
    elif ltype == "FullModule":
        module.weight = torch.nn.Parameter(module.weight.to(devices.device, dtype=torch.float))
    
    if hasattr(module, 'bias') and module.bias != None:
        module.bias = torch.nn.Parameter(module.bias.to(devices.device, dtype=torch.float))


def restoremodel(p):
    model = p.sd_model
    for name, module in model.named_modules():
        if hasattr(module, "lora_weights_backup"):
            if module.lora_weights_backup is not None:
                if isinstance(module, torch.nn.MultiheadAttention):
                    module.in_proj_weight.copy_(module.lora_weights_backup[0])
                    module.out_proj.weight.copy_(module.lora_weights_backup[1])
                else:
                    module.weight.copy_(module.lora_weights_backup)
                module.lora_weights_backup = None
                module.lora_current_names = None


def unloadlorafowards(p):
    global orig_lora_Linear_forward, orig_lora_Conv2d_forward, orig_lora_apply_weights, orig_lora_forward, lactive
    lactive = False
    import lora
    lora.loaded_loras.clear()
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
