import copy
from pprint import pprint
import torch
from modules import devices, shared, extra_networks, sd_hijack
from modules.script_callbacks import CFGDenoisedParams, CFGDenoiserParams
from torchvision.transforms import InterpolationMode, Resize  # Mask.
import scripts.attention as att
from scripts.regions import floatdef
from scripts.attention import makerrandman

try:
    from modules.ui import versions_html
    forge = "forge" in versions_html()
    reforge = "reForge" in versions_html()
except:
    forge = reforge = False

if forge:
    from modules.script_callbacks import AfterCFGCallbackParams, on_cfg_after_cfg

denoised_params = AfterCFGCallbackParams if forge else CFGDenoisedParams

islora = True
in_hr = False
layer_name = "lora_layer_name"
orig_Linear_forward = None

orig_lora_functional = False

lactive = False
labug =False
MINID = 1000
MAXID = 10000
LORAID = MINID # Discriminator for repeated lora usage / across gens, presumably.

def setuploras(self):
    global lactive, labug, islora, orig_Linear_forward, orig_lora_functional, layer_name
    lactive = True
    labug = self.debug
    islora = self.isbefore15
    layer_name = self.layer_name
    orig_lora_functional = orig_lora_functional = shared.opts.lora_functional if hasattr(shared.opts,"lora_functional") else False

    try:
        if 150 <= self.ui_version <= 159 or self.slowlora:
            shared.opts.lora_functional = False
        else:
            shared.opts.lora_functional = True
    except:
        pass

    is15 = 150 <= self.ui_version <= 159
    orig_Linear_forward = torch.nn.Linear.forward
    torch.nn.Linear.forward = h15_Linear_forward if is15 else h_Linear_forward

    if forge:
        shared.sd_model.forge_objects.unet.set_model_unet_function_wrapper(lambda apply, params: denoised_callback_s(apply, params, p3=self))
        from backend.args import dynamic_args
        dynamic_args["online_lora"] = True
        import networks as net
        net.load_networks = load_networks

        for name, module in shared.sd_model.forge_objects.clip.cond_stage_model.clip_l.named_modules():
            if name == "transformer.text_model.encoder.layers.0.self_attn.q_proj":
                module.forward = forge_linear_forward.__get__(module)
    if reforge:
        shared.sd_model.forge_objects.unet.set_model_unet_function_wrapper(lambda apply, params: denoised_callback_s(apply, params, p3=self))

def cloneparams(orig,target):
    target.x = orig.x.clone()
    target.image_cond  = orig.image_cond.clone()
    target.sigma  = orig.sigma.clone()

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
    self.step = params.sampling_step
    self.total_step = params.total_sampling_steps

    if "Pro" in self.mode:  # in Prompt mode, make masks from sum of attension maps
        if self.x == None : cloneparams(params,self) # return to step 0 if mask is ready
        self.pfirst = True

        lim = 1 if self.is_sdxl else 3

        if len(att.pmaskshw) > lim:
            self.filters = []
            for b in range(self.batch_size):

                allmask = []
                basemask = None
                for t, th, bratio in zip(self.pe, self.th, self.bratios):
                    key = f"{t}-{b}"
                    _, _, mask = att.makepmask(att.pmasks[key], params.x.shape[2], params.x.shape[3], th, self.step,self.total_step, self.is_sdxl,bratio = bratio)
                    mask = mask.repeat(params.x.shape[1],1,1)
                    basemask = 1 - mask if basemask is None else basemask - mask
                    if self.ex:
                        for l in range(len(allmask)):
                            mt = allmask[l] - mask
                            allmask[l] = torch.where(mt > 0, 1,0)
                    allmask.append(mask)
                if not self.ex:
                    sum = torch.stack(allmask, dim=0).sum(dim=0)
                    sum = torch.where(sum == 0, 1 , sum)
                    allmask = [mask  / sum for mask in allmask]
                basemask = torch.where(basemask > 0, 1, 0)
                allmask.insert(0,basemask)
                self.filters.extend(allmask)
            att.maskready = True

            for t, th, bratio in zip(self.pe, self.th, self.bratios):
                allmask = []
                for hw in att.pmaskshw:
                    masks = None
                    for b in range(self.batch_size):
                        key = f"{t}-{b}"
                        _, mask, _ = att.makepmask(att.pmasks[key], hw[0], hw[1], th, self.step,self.total_step, self.is_sdxl,bratio = bratio)
                        mask = mask.unsqueeze(0).unsqueeze(-1)
                        masks = mask if b ==0 else torch.cat((masks,mask),dim=0)
                    allmask.append(mask)     
                att.pmasksf[key] = allmask
                att.maskready = True

            if not self.rebacked: 
                cloneparams(self,params)
                params.sampling_step = 0
                self.rebacked = True

    if "La" in self.calc:
        self.condi = 0
        global in_hr, regioner
        regioner.step = params.sampling_step
        in_hr = self.in_hr
        regioner.u_count = 0
        if "u_list" not in self.log.keys() and hasattr(regioner,"u_llist"):
            self.log["u_list"] = regioner.u_llist.copy()
        if "u_list_hr" not in self.log.keys() and hasattr(regioner,"u_llist") and in_hr:
            self.log["u_list_hr"] = regioner.u_llist.copy()
        xt = params.x.clone()
        ict = params.image_cond.clone()
        st =  params.sigma.clone()
        batch = self.batch_size
        areas = xt.shape[0] // batch -1
        if forge: return
        # SBM Stale version workaround.
        if hasattr(params,"text_cond"):
            if "DictWithShape" in params.text_cond.__class__.__name__:
                ct = {}
                for key in params.text_cond.keys():
                    ct[key] = params.text_cond[key].clone()
            else:
                ct =  params.text_cond.clone()

        for a in range(areas):
            for b in range(batch):
                params.x[b+a*batch] = xt[a + b * areas]
                params.image_cond[b+a*batch] = ict[a + b * areas]
                params.sigma[b+a*batch] = st[a + b * areas]
                # SBM Stale version workaround.
                if hasattr(params,"text_cond"):
                    if "DictWithShape" in params.text_cond.__class__.__name__:
                        for key in params.text_cond.keys():
                            params.text_cond[key][b+a*batch] = ct[key][a + b * areas]
                    else:
                        params.text_cond[b+a*batch] = ct[a + b * areas]

def denoised_callback_s(p1, p2 = None, p3 = None):
    # if forge p1: model.apply(), p2: params, p3: script
    # if A1111 p1: script, p2: DenoisedParams, p3: None
    if p3 is not None:
        self = p3      
        input_x = p2["input"]
        timestep = p2["timestep"]
        cond_or_uncond = p2["cond_or_uncond"]
        c = p2["c"]
        conds = c["c_crossattn"]
        y = c["y"] if "y" in c else None

        if not lactive:
            return p1(input_x, timestep, **c)
        
        length = len(cond_or_uncond)
        batch = input_x.shape[0] // length

        cond_or_uncond = cond_or_uncond * batch

        region_list, orig_list = forge_make_chenge_list(batch, length)

        outs = []
        for i in range(length):
            regioner.set_region(length - i - 1)
            c["c_crossattn"] = conds[i*batch:i*batch+batch]
            if y is not None:
                c["y"] = y[i*batch:i*batch+batch]
            outs.append(p1(input_x[i*batch:i*batch+batch], torch.cat([timestep[i:i+1]]*batch), **c))

        output = torch.cat(outs)

        x = output[region_list]
        xt = x.clone()
        areas = length - 1
    else:
        self = p1
        params = p2

        batch = self.batch_size
        x = params.x
        xt = params.x.clone()
        areas = xt.shape[0] // batch - 1

    if "La" in self.calc:
        # x.shape = [batch_size, C, H // 8, W // 8]

        if not "Pro" in self.mode:
            indrebuild = self.filters == [] or self.filters[0].size() != x[0].size()

            if indrebuild:
                if "Ran" in self.mode:
                    if self.filters == []:
                        self.filters = [self.ranbase] + self.ransors if self.usebase else self.ransors
                    elif self.filters[0][:,:].size() != x[0,0,:,:].size():
                        self.filters = hrchange(self.ransors,x.shape[2], x.shape[3])
                else:
                    if "Mask" in self.mode:
                        masks = (self.regmasks,self.regbase)
                    else:
                        masks = self.aratios  #makefilters(c,h,w,masks,mode,usebase,bratios,indmask = None)
                    self.filters = makefilters(x.shape[1], x.shape[2], x.shape[3],masks, self.mode, self.usebase, self.bratios, "Mas" in self.mode)
                self.filters = [f for f in self.filters]*batch
        else:
            if not att.maskready:
                self.filters = [1,*[0 for a in range(areas - 1)]] * batch

        if self.debug: 
            print("filterlength : ",len(self.filters))
            print("x : ",x.shape)
            print("areas : ",areas)

        for b in range(batch):
            for a in range(areas) :
                fil = self.filters[a + b*areas]
                if self.debug : print(f"x = {x.size()}i = {a + b*areas}, j = {b + a*batch}, cond = {a + b*areas},filsum = {fil if type(fil) is int else torch.sum(fil)}, uncon = {x.size()[0]+(b-batch)}")
                x[a + b * areas, :, :, :] =  xt[b + a*batch, :, :, :] * fil + x[x.size()[0]+(b-batch), :, :, :] * (1 - fil)

    if self.total_step == self.step + 2:
        if self.rps is not None and self.diff:
            if self.rps.latent is None:
                self.rps.latent = x.clone()
                return
            elif self.rps.latent.shape[2:] != x.shape[2:] and self.rps.latent_hr is None:
                self.rps.latent_hr = x.clone()
                return
            else:
                for b in range(batch):
                    for a in range(areas) :
                        fil = self.filters[a+1]
                        orig = self.rps.latent if self.rps.latent.shape[2:] == x.shape[2:] else self.rps.latent_hr
                        if self.debug : print(f"x = {x.size()}i = {a + b*areas}, j = {b + a*batch}, cond = {a + b*areas},filsum = {fil if type(fil) is int else torch.sum(fil)}, uncon = {x.size()[0]+(b-batch)}")
                        #print("1",type(self.rps.latent),type(fil))
                        x[:,:,:,:] =  orig[:,:,:,:] * (1 - fil) + x[:,:,:,:] * fil

    #if params.total_sampling_steps - 7 == params.sampling_step + 2:
    if att.maskready:
        if self.rps is not None and self.diff:
            if self.rps.latent is not None:
                if self.rps.latent.shape[2:] != x.shape[2:]:
                    if self.rps.latent_hr is None: return
                for b in range(batch):
                    for a in range(areas) :
                        fil = self.filters[a+1]
                        orig = self.rps.latent if self.rps.latent.shape[2:] == x.shape[2:] else self.rps.latent_hr
                        if self.debug : print(f"x = {x.size()}i = {a + b*areas}, j = {b + a*batch}, cond = {a + b*areas},filsum = {fil if type(fil) is int else torch.sum(fil)}, uncon = {x.size()[0]+(b-batch)}")
                        #print("2",type(self.rps.latent),type(fil))
                        x[:,:,:,:] =  orig[:,:,:,:] * (1 - fil) + x[:,:,:,:] * fil

    if self.step == 0 and self.in_hr:
        if self.rps is not None and self.diff:
            if self.rps.latent is not None:
                if self.rps.latent.shape[2:] != x.shape[2:] and self.rps.latent_hr is None: return
                for b in range(batch):
                    for a in range(areas) :
                        fil = self.filters[a+1]
                        orig = self.rps.latent if self.rps.latent.shape[2:] == x.shape[2:] else self.rps.latent_hr
                        if self.debug : print(f"x = {x.size()}i = {a + b*areas}, j = {b + a*batch}, cond = {a + b*areas},filsum = {fil if type(fil) is int else torch.sum(fil)}, uncon = {x.size()[0]+(b-batch)}")
                        #print("3",type(self.rps.latent),type(fil))
                        x[:,:,:,:] =  orig[:,:,:,:] * (1 - fil) + x[:,:,:,:] * fil

    if p3 is not None: #forge
        out = x[orig_list]
        return out

def forge_make_chenge_list(batch, length):
    orig = [x for x in range(batch*length)]

    chunks_1 = [orig[i:i + batch] for i in range(0, len(orig), batch)]
    chunks_2 = [[(i) + (length - 1) * x for x in range(batch)] for i in range(length)]
    
    out1, out2, = [], []

    for c1 in chunks_1[::-1]:
        out1.extend(c1)
    
    out2.extend(chunks_1[-1])
    
    for c2 in chunks_2[:-1][::-1]:
        out2.extend(c2)

    return out1, out2


######################################################
##### Latent Method

def hrchange(filters,h, w):
    out = []
    for filter in filters:
        out.append(makerrandman(filter,h,w,True))
    return out

# Remove tags from called lora names.
flokey = lambda x: (x.split("added_by_regional_prompter")[0]
                    .split("added_by_lora_block_weight")[0].split("_in_LBW")[0].split("_in_RP")[0])

def lora_namer(self, p, lnter, lnur):
    ldict_u = {}
    ldict_te = {}
    lorder = [] # Loras call order for matching with u/te lists.
    import lora as loraclass
    name_to_hash = {}
    for lora in loraclass.loaded_loras:
        ldict_u[lora.network_on_disk.filename if forge else lora.name] =lora.multiplier if self.isbefore15 else lora.unet_multiplier
        ldict_te[lora.network_on_disk.filename if forge else lora.name] =lora.multiplier  if self.isbefore15 else lora.te_multiplier
        name_to_hash[lora.network_on_disk.alias] = lora.network_on_disk.filename
        name_to_hash[lora.network_on_disk.name] = lora.network_on_disk.filename
    
    subprompts = self.current_prompts[0].split("AND")
    ldictlist_u =[ldict_u.copy() for i in range(len(subprompts)+1)]
    ldictlist_te =[ldict_te.copy() for i in range(len(subprompts)+1)]

    for i, prompt in enumerate(subprompts):
        _, extranets = extra_networks.parse_prompts([prompt])
        calledloras = extranets["lora"]

        names = ""
        tdict = {}

        for called in calledloras:
            names = names + name_to_hash[called.items[0]] if forge else names + called.items[0]
            tdict[name_to_hash[called.items[0]] if forge else called.items[0]] = syntaxdealer(called.items,"unet=",1)

        for key in ldictlist_u[i].keys():
            shin_key = flokey(key) 
            if shin_key in names:
                ldictlist_u[i+1][key] = float(tdict[shin_key])
                ldictlist_te[i+1][key] = float(tdict[shin_key])
                if key not in lorder:
                    lorder.append(key)
            else:
                ldictlist_u[i+1][key] = 0
                ldictlist_te[i+1][key] = 0
                
    if self.debug: print("Regioner lorder: ",lorder)
    global regioner
    regioner.__init__(self.lstop,self.lstop_hr)
    u_llist = [d.copy() for d in ldictlist_u[1:]]
    u_llist.append(ldictlist_u[0].copy())
    regioner.te_llist = ldictlist_te
    regioner.u_llist = u_llist
    regioner.ndeleter(lnter, lnur, lorder)
    if self.debug:
        print("LoRA regioner : TE list",regioner.te_llist)
        print("LoRA regioner : U list",regioner.u_llist)

def syntaxdealer(items,type,index): #type "unet=", "x=", "lwbe=" 
    for item in items:
        if type in item:
            if "@" in item:return 1 #for loractl
            return item.replace(type,"")
    return items[index] if "@" not in items[index] else 1

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

TE_START_NAME_XL = "0_transformer_text_model_encoder_layers_0_self_attn_q_proj"

class LoRARegioner:

    def __init__(self,stop=0,stop_hr=0):
        self.te_count = 0
        self.u_count = 0
        self.te_llist = [{}]
        self.u_llist = [{}]
        self.mlist = {}
        self.ctl = False
        self.step = 0
        self.stop = stop
        self.stop_hr = stop_hr
        self.stopped = False
        self.stopped_hr = False
        self.orig_weight = {}

        try:
            import lora_ctl_network as ctl
            self.ctlweight = copy.deepcopy(ctl.lora_weights)
            for set in self.ctlweight.values():
                for weight in set.values():
                    if type(weight) == list:
                        self.ctl = True        
        except:
            pass

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

    def search_key(self,lora,i,xlist):
        lorakey = lora.loaded_loras[i].name
        if lorakey not in xlist.keys():
            shin_key = flokey(lorakey)
            picked = False
            for mlkey in xlist.keys():
                if mlkey.startswith(shin_key):
                    lorakey = mlkey
                    picked = True
            if not picked:
                print(f"key is not found in:{xlist.keys()}")
        return lorakey

    def te_start(self):
        self.mlist = self.te_llist[self.te_count % len(self.te_llist)]
        if self.mlist == {}: return
        self.te_count += 1
        import lora
        for i in range(len(lora.loaded_loras)):
            lorakey = self.search_key(lora,i,self.mlist)
            lora.loaded_loras[i].multiplier = self.mlist[lorakey]
            lora.loaded_loras[i].te_multiplier = self.mlist[lorakey]

    def u_start(self):
        if labug : print("u_count",self.u_count ,"u_count '%' divide",  self.u_count % len(self.u_llist))
        self.mlist = self.u_llist[self.u_count % len(self.u_llist)]
        if self.mlist == {}: return
        self.u_count  += 1

        stopstep = self.stop_hr if in_hr else self.stop

        import lora
        for i in range(len(lora.loaded_loras)):
            lorakey = self.search_key(lora,i,self.mlist)
            lora.loaded_loras[i].multiplier = 0 if self.step + 2 > stopstep and stopstep else self.mlist[lorakey]
            lora.loaded_loras[i].unet_multiplier = 0 if self.step + 2 > stopstep and stopstep else self.mlist[lorakey]
            if labug :print(lorakey,lora.loaded_loras[i].multiplier,lora.loaded_loras[i].multiplier ) 
            if self.ctl:
                import lora_ctl_network as ctl
                key = "hrunet" if in_hr else "unet"
                if self.mlist[lorakey] == 0 or (self.step + 2 > stopstep and stopstep):
                    ctl.lora_weights[lorakey][key] = [[0],[0]]
                    if labug :print(ctl.lora_weights[lorakey])
                else:
                    if key in self.ctlweight[lorakey].keys():
                        ctl.lora_weights[lorakey][key] = self.ctlweight[lorakey][key]
                    else:
                        ctl.lora_weights[lorakey][key] = self.ctlweight[lorakey]["unet"]
                    if labug :print(ctl.lora_weights[lorakey])

    def reset(self):
        self.te_count = 0
        self.u_count = 0
        self.stopped = False
        self.stopped_hr = False
        self.orig_weight = {}

    def te_start_f(self):
        self.mlist = self.te_llist[self.te_count % len(self.te_llist)]
        if self.mlist == {}: return
        self.te_count += 1
        
        if labug:
            print(f"Set LoRA for Region {self.te_count % len(self.te_llist)}, u_count",self.u_count ,"u_count '%' divide",  self.u_count % len(self.u_llist))
            print(self.mlist)

        lora_lorader = shared.sd_model.forge_objects.clip.patcher.lora_loader
        lora_patches = shared.sd_model.forge_objects.clip.patcher.lora_patches
        offload_device = shared.sd_model.forge_objects.clip.patcher.offload_device

        for lora_key, patch in lora_patches.items():
            for list_key in self.mlist:
                if list_key in lora_key[0]:
                    if labug: 
                        print(f"LoRA {lora_key} detected in {self.mlist}")
                    for patch_key in patch:
                        if patch_key + list_key not in self.orig_weight:
                            self.orig_weight[patch_key + list_key] = patch[patch_key][0][0]
                        patch[patch_key][0][0] = self.orig_weight[patch_key + list_key] * self.mlist[list_key]

        refresh(lora_lorader, lora_patches=lora_patches, offload_device=offload_device)

    def set_region(self, region):
        self.mlist = self.u_llist[region]
        if labug:
            print(f"Set LoRA for Region {region}, u_count",self.u_count ,"u_count '%' divide",  self.u_count % len(self.u_llist))
            print(self.mlist)
        if self.mlist == {}: return
 
        strengths = list(self.mlist.values())

        def set_strengths(strengths):
            for name, module in shared.sd_model.forge_objects.unet.model.named_modules():
                patches = getattr(module, 'forge_online_loras', None)
                weight_patches, bias_patches = None, None
                if patches is not None:
                    weight_patches = patches.get('weight', None)
                    if weight_patches:
                        if len(weight_patches) != len(strengths) :
                            continue
                        for i in range(len(strengths)):
                            if name not in self.orig_weight:
                                self.orig_weight[name] = [x[0] for x in weight_patches]
                            weight_patches[i][0] = strengths[i] * self.orig_weight[name][i]

        stopstep = self.stop_hr if in_hr else self.stop
        if self.step >= stopstep:
            if (self.stopped_hr if in_hr else self.stopped):
                return
            else:
                set_strengths(0)
                if in_hr:
                    self.stopped_hr = True
                else:
                    self.stopped = True

        set_strengths(strengths)

regioner = LoRARegioner()

############################################################
##### for new lora apply method in web-ui

def h_Linear_forward(self, input):
    changethelora(getattr(self, layer_name, None))
    if islora:
        import lora
        return lora.lora_forward(self, input, torch.nn.Linear_forward_before_lora)
    elif forge or reforge:
        return orig_Linear_forward(self, input)
    else:
        import networks
        if shared.opts.lora_functional:
            return networks.network_forward(self, input, networks.originals.Linear_forward)
        networks.network_apply_weights(self)
        return networks.originals.Linear_forward(self, input)

def h15_Linear_forward(self, input):
    changethelora(getattr(self, layer_name, None))
    if islora:
        import lora
        return lora.lora_forward(self, input, torch.nn.Linear_forward_before_lora)
    else:
        import networks
        if shared.opts.lora_functional:
            return networks.network_forward(self, input, networks.network_Linear_forward)
        networks.network_apply_weights(self)
        return torch.nn.Linear_forward_before_network(self, input)

def forge_linear_forward(self, x):
    regioner.te_start_f()
    from backend import operations as op
    if self.parameters_manual_cast:
        weight, bias, signal = op.weights_manual_cast(self, x)
        with op.main_stream_worker(weight, bias, signal):
            return torch.nn.functional.linear(x, weight, bias)
    else:
        weight, bias = op.get_weight_and_bias(self)
        return torch.nn.functional.linear(x, weight, bias)

def changethelora(name):
    if lactive:
        global regioner
        if name == TE_START_NAME or name == TE_START_NAME_XL:
            regioner.te_start()
        elif name == UNET_START_NAME:
            regioner.u_start()

LORAANDSOON = {
    "LoraHadaModule" : "w1a",
    "LycoHadaModule" : "w1a",
    "NetworkModuleHada": "w1a",
    "FullModule" : "weight",
    "NetworkModuleFull": "weight",
    "IA3Module" : "w",
    "NetworkModuleIa3" : "w",
    "LoraKronModule" : "w1",
    "LycoKronModule" : "w1",
    "NetworkModuleLokr": "w1",
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
        
    elif ltype == "LoraHadaModule" or ltype == "LycoHadaModule" or ltype == "NetworkModuleHada":
        module.w1a = torch.nn.Parameter(module.w1a.to(devices.device, dtype=torch.float))
        module.w1b = torch.nn.Parameter(module.w1b.to(devices.device, dtype=torch.float))
        module.w2a = torch.nn.Parameter(module.w2a.to(devices.device, dtype=torch.float))
        module.w2b = torch.nn.Parameter(module.w2b.to(devices.device, dtype=torch.float))
        
        if module.t1 is not None:
            module.t1 = torch.nn.Parameter(module.t1.to(devices.device, dtype=torch.float))

        if module.t2 is not None:
            module.t2 = torch.nn.Parameter(module.t2.to(devices.device, dtype=torch.float))
        
    elif ltype == "FullModule" or ltype == "NetworkModuleFull":
        module.weight = torch.nn.Parameter(module.weight.to(devices.device, dtype=torch.float))
    
    if hasattr(module, 'bias') and module.bias != None:
        module.bias = torch.nn.Parameter(module.bias.to(devices.device, dtype=torch.float))

def unloadlorafowards(p):
    global orig_Linear_forward, lactive, labug
    lactive = labug = False
        
    try:
        shared.opts.lora_functional =  orig_lora_functional
    except:
        pass

    import lora
    if forge:
        from backend.args import dynamic_args
        dynamic_args["online_lora"] = False
    else:
        emb_db = sd_hijack.model_hijack.embedding_db
        for net in lora.loaded_loras:
            if hasattr(net,"bundle_embeddings"):
                for emb_name, embedding in net.bundle_embeddings.items():
                    if embedding.loaded:
                        emb_db.register_embedding_by_name(None, shared.sd_model, emb_name)

    lora.loaded_loras.clear()
    if orig_Linear_forward != None :
        torch.nn.Linear.forward = orig_Linear_forward
        orig_Linear_forward = None

def refresh(self, lora_patches, offload_device=torch.device('cpu')):
    from backend.patcher import lora
    from backend import utils, memory_management, operations
    hashes = str(list(lora_patches.keys()))

    # Merge Patches

    all_patches = {}

    for (_, _, _, online_mode), patches in lora_patches.items():
        for key, current_patches in patches.items():
            all_patches[(key, online_mode)] = all_patches.get((key, online_mode), []) + current_patches

    # Initialize

    memory_management.signal_empty_cache = True

    parameter_devices = lora.get_parameter_devices(self.model)

    # Restore

    for m in set(self.online_backup):
        del m.forge_online_loras

    self.online_backup = []

    for k, w in self.backup.items():
        if not isinstance(w, torch.nn.Parameter):
            # In very few cases
            w = torch.nn.Parameter(w, requires_grad=False)

        utils.set_attr_raw(self.model, k, w)

    self.backup = {}

    lora.set_parameter_devices(self.model, parameter_devices=parameter_devices)

    # Patch

    for (key, online_mode), current_patches in all_patches.items():
        try:
            parent_layer, child_key, weight = utils.get_attr_with_parent(self.model, key)
            assert isinstance(weight, torch.nn.Parameter)
        except:
            raise ValueError(f"Wrong LoRA Key: {key}")

        if online_mode:
            if not hasattr(parent_layer, 'forge_online_loras'):
                parent_layer.forge_online_loras = {}

            parent_layer.forge_online_loras[child_key] = current_patches
            self.online_backup.append(parent_layer)
            continue

        if key not in self.backup:
            self.backup[key] = weight.to(device=offload_device)

        bnb_layer = None

        if hasattr(weight, 'bnb_quantized') and operations.bnb_avaliable:
            bnb_layer = parent_layer
            from backend.operations_bnb import functional_dequantize_4bit
            weight = functional_dequantize_4bit(weight)

        gguf_cls = getattr(weight, 'gguf_cls', None)
        gguf_parameter = None

        if gguf_cls is not None:
            gguf_parameter = weight
            from backend.operations_gguf import dequantize_tensor
            weight = dequantize_tensor(weight)

        try:
            weight = lora.merge_lora_to_weight(current_patches, weight, key, computation_dtype=torch.float32)
        except:
            print('Patching LoRA weights out of memory. Retrying by offloading models.')
            lora.set_parameter_devices(self.model, parameter_devices={k: offload_device for k in parameter_devices.keys()})
            memory_management.soft_empty_cache()
            weight = lora.merge_lora_to_weight(current_patches, weight, key, computation_dtype=torch.float32)

        if bnb_layer is not None:
            bnb_layer.reload_weight(weight)
            continue

        if gguf_cls is not None:
            gguf_cls.quantize_pytorch(weight, gguf_parameter)
            continue

        utils.set_attr_raw(self.model, key, torch.nn.Parameter(weight, requires_grad=False))

    # End

    lora.set_parameter_devices(self.model, parameter_devices=parameter_devices)
    self.loaded_hash = hashes
    return


def load_networks(names, te_multipliers=None, unet_multipliers=None, dyn_dims=None):
    from modules import sd_models
    import networks as nets
    from backend.args import dynamic_args
    from modules import sd_models, errors

    global lora_state_dict_cache

    current_sd = sd_models.model_data.get_sd_model()
    if current_sd is None:
        return

    nets.loaded_networks.clear()

    unavailable_networks = []
    for name in names:
        if name.lower() in nets.forbidden_network_aliases and nets.available_networks.get(name) is None:
            unavailable_networks.append(name)
        elif nets.available_network_aliases.get(name) is None:
            unavailable_networks.append(name)

    if unavailable_networks:
        nets.update_available_networks_by_names(unavailable_networks)

    networks_on_disk = [nets.available_networks.get(name, None) if name.lower() in nets.forbidden_network_aliases else nets.available_network_aliases.get(name, None) for name in names]
    if any(x is None for x in networks_on_disk):
        nets.list_available_networks()
        networks_on_disk = [nets.available_networks.get(name, None) if name.lower() in nets.forbidden_network_aliases else nets.available_network_aliases.get(name, None) for name in names]

    for i, (network_on_disk, name) in enumerate(zip(networks_on_disk, names)):
        try:
            net = nets.load_network(name, network_on_disk)
        except Exception as e:
            errors.display(e, f"loading network {network_on_disk.filename}")
            continue
        net.mentioned_name = name
        network_on_disk.read_hash()
        nets.loaded_networks.append(net)

    online_mode = dynamic_args.get('online_lora', False)

    if not current_sd.forge_objects.unet.model.storage_dtype in [torch.float32, torch.float16, torch.bfloat16]:
        online_mode = False

    compiled_lora_targets = []
    for a, b, c in zip(networks_on_disk, unet_multipliers, te_multipliers):
        compiled_lora_targets.append([a.filename, b, c, online_mode])

    compiled_lora_targets_hash = str(compiled_lora_targets)

    if current_sd.current_lora_hash == compiled_lora_targets_hash:
        return

    current_sd.current_lora_hash = compiled_lora_targets_hash
    current_sd.forge_objects.unet = current_sd.forge_objects_original.unet
    current_sd.forge_objects.clip = current_sd.forge_objects_original.clip

    for filename, strength_model, strength_clip, online_mode in compiled_lora_targets:
        lora_sd = nets.load_lora_state_dict(filename)
        current_sd.forge_objects.unet, current_sd.forge_objects.clip = nets.load_lora_for_models(
            current_sd.forge_objects.unet, current_sd.forge_objects.clip, lora_sd, strength_model, strength_clip,
            filename=filename, online_mode=online_mode)

    current_sd.forge_objects_after_applying_lora = current_sd.forge_objects.shallow_copy()
    return