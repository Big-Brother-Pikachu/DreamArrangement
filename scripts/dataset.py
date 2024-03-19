import json, glob, random
import numpy as np
import torch
from scripts.utils import *

class TableDataset:
    '''Table Dataset'''

    def __init__(self, path=["data/YCB_kitchen_data"], split="folder", num_class=8):
        """ Currently all scenes_tv (test+validation) are used in training, and validation uses test.
                vertical:                       len(self.scenes_tv)=5668,  len(self.scenes_test)=224
        """
        maxnobj = 0
        maxnmask = 0
        self.path = path
        glo = [0, 1, 2, 3, 4]
        loc = [0, 1, 2, 3]

        self.nfpbpn = 250
        self.maxnfpoc = 25
        
        self.pos_dim = 2 # coord in x, y (disregard z); normalized to [-1,1]
        self.ang_dim = 2 # cos(theta), sin(theta), where theta is in [-pi, pi]
        self.siz_dim = 2 # length of bounding box in x, y; normalized to [-1, 1]
        self.cla_dim = num_class
        self.sha_dim = self.siz_dim+self.cla_dim

        self.room_size = [2, 2]

        self.data_tv = []
        self.mask_tv = []
        self.cond_tv = []
        self.word_tv = []
        self.file_tv = []
        self.data_test = []
        self.mask_test = []
        self.cond_test = []
        self.word_test = []
        self.file_test = []

        for p in path:
            with open(f"{p}/split.json") as f:
                split_j = json.load(f)
            ## shape: square, circle, horizon line, vertical line

            for f in split_j[0]:
                maxnobj = max(maxnobj, split_j[0][f]['maxnobj'])
            for f in split_j[1]:
                maxnobj = max(maxnobj, split_j[1][f]['maxnobj'])

            for f in split_j[0]:
                maxnmask = max(maxnmask, split_j[0][f]['maxnmask'])
            for f in split_j[1]:
                maxnmask = max(maxnmask, split_j[1][f]['maxnmask'])
        self.maxnobj = maxnobj
        self.maxnmask = maxnmask

        for p in path:
            with open(f"{p}/split.json") as f:
                split_j = json.load(f)
            ## shape: square, circle, horizon line, vertical line

            if split == "folder":
                train_split = split_j[0]
                val_split = split_j[1]
                for s in train_split:
                    desc = train_split[s]['description']
                    shape = train_split[s]['shape']
                    surround = train_split[s]['surround']
                    if shape not in glo:
                        continue
                    if surround not in loc:
                        continue
                    for folder in train_split[s]['folder']:
                        for file in glob.glob(f"{p}/{s}/{folder}/*.pt", recursive=True):
                            input, mask = self.loadfile(file)
                            if input is not None:
                                self.data_tv.append(input)
                                self.mask_tv.append(mask)
                                self.cond_tv.append(desc)
                                self.word_tv.append(torch.LongTensor([[shape, surround]]))
                                self.file_tv.append(file)
                for s in val_split:
                    desc = val_split[s]['description']
                    shape = val_split[s]['shape']
                    surround = val_split[s]['surround']
                    if shape not in glo:
                        continue
                    if surround not in loc:
                        continue
                    for folder in val_split[s]['folder']:
                        for file in glob.glob(f"{p}/{s}/{folder}/*.pt", recursive=True):
                            input, mask = self.loadfile(file)
                            if input is not None:
                                self.data_test.append(input)
                                self.mask_test.append(mask)
                                self.cond_test.append(desc)
                                self.word_test.append(torch.LongTensor([[shape, surround]]))
                                self.file_test.append(file)
            elif split == "instance":
                for sp in split_j[0]:
                    desc = split_j[0][sp]['description']
                    shape = split_j[0][sp]['shape']
                    surround = split_j[0][sp]['surround']
                    if shape not in glo:
                        continue
                    if surround not in loc:
                        continue
                    with open(f"{p}/{sp}/instance_train_split.txt", "r") as f:
                        train_split = list(p.rstrip("\n").replace("png", "pt") for p in f.readlines())
                    for s in train_split:
                        file = f"{p}/{sp}/{s}"
                        input, mask = self.loadfile(file)
                        if input is not None:
                            self.data_tv.append(input)
                            self.mask_tv.append(mask)
                            self.cond_tv.append(desc)
                            self.word_tv.append(torch.LongTensor([[shape, surround]]))
                            self.file_tv.append(file)
                for sp in split_j[1]:
                    desc = split_j[1][sp]['description']
                    shape = split_j[1][sp]['shape']
                    surround = split_j[1][sp]['surround']
                    if shape not in glo:
                        continue
                    if surround not in loc:
                        continue
                    with open(f"{p}/{sp}/instance_test_split.txt", "r") as f:
                        val_split = list(p.rstrip("\n").replace("png", "pt") for p in f.readlines())
                    for s in val_split:
                        file = f"{p}/{sp}/{s}"
                        input, mask = self.loadfile(file)
                        if input is not None:
                            self.data_test.append(input)
                            self.mask_test.append(mask)
                            self.cond_test.append(desc)
                            self.word_test.append(torch.LongTensor([[shape, surround]]))
                            self.file_test.append(file)
        self.data_tv = torch.cat(self.data_tv).numpy()
        self.mask_tv = torch.cat(self.mask_tv).numpy()
        self.word_tv = torch.cat(self.word_tv).numpy()
        self.data_test = torch.cat(self.data_test).numpy()
        self.mask_test = torch.cat(self.mask_test).numpy()
        self.word_test = torch.cat(self.word_test).numpy()

    def loadfile(self, file):
        """ file: string

            obj: tensor. 
        """ 
        scene = torch.load(file)
        input = []
        mask = []
        for c in scene:
            for instance in scene[c]:
                bbox = instance["bbox"].float()
                bbox_size = instance["bbox_size"].float()
                bbox_mean = torch.mean(bbox, dim=0)
                if "orientation" in instance:
                    orientation = instance["orientation"].float()
                    class_emb = torch.zeros((self.cla_dim, ))
                    class_emb[c - 1] = 1
                    obj = torch.cat([bbox_mean, orientation, bbox_size, class_emb]).unsqueeze(0)
                    input.append(obj)
                else:
                    mask_type = instance["type"]
                    orientation = torch.tensor([1, 0]).float()
                    mask_obj = torch.cat([bbox_mean, orientation, bbox_size, mask_type]).unsqueeze(0)
                    mask.append(mask_obj)
        if len(input) > self.maxnobj:
            return None, None
        if len(mask) > self.maxnmask:
            return None, None
        while len(input) < self.maxnobj:
            input.append(torch.zeros_like(obj))
        input = torch.cat(input).unsqueeze(0)
        while len(mask) < self.maxnmask:
            mask.append(torch.zeros_like(mask_obj))
        mask = torch.cat(mask).unsqueeze(0)
        return input, mask

    @staticmethod
    def parse_cla(cla):
        """ cla: [nobj, cla_dim]

            nobj: scalar, number of objects in the scene
            cla_idx: [nobj,], each object's class type index. 
        """ 
        nobj = cla.shape[0]
        for o_i in range(cla.shape[0]):
            if np.sum(cla[o_i]) == 0: 
                nobj = o_i
                break
        cla_idx = np.argmax(cla[:nobj,:], axis=1) #[nobj,cla_dim] -> [nobj,] (each obj's class index)
        return nobj, cla_idx

    @staticmethod
    def reset_padding(nobjs, toreset):
        """ nobjs: [batch_size]
            toreset(2): [batch_size, maxnumobj, 2]
        """
        for scene_idx in range(toreset.shape[0]):
            toreset[scene_idx, nobjs[scene_idx]:,:]=0
        return toreset

    @staticmethod
    def get_objbbox_corneredge(pos, ang_rad, siz):
        """ pos: [pos_dim,]
            ang_rad: [1,1], rotation from (1,0) in radians
            siz: [siz_dim,], full bbox length

            corners: corner points (4x2 numpy array) of the rotated bounding box centered at pos and with bbox len siz,
            bboxedge: array of 2-tuples of numpy arrays [x,y]
        """
        siz = (siz.astype(np.float32))/2
        corners = np.array([[pos[0]-siz[0], pos[1]+siz[1]],  # top left (origin: bottom left)
                            [pos[0]+siz[0], pos[1]+siz[1]],  # top right
                            [pos[0]-siz[0], pos[1]-siz[1]],  # bottom left 
                            [pos[0]+siz[0], pos[1]-siz[1]]]) # bottom right #(4, 2)
        corners =  np_rotate_center(corners, np.repeat((ang_rad), repeats=4, axis=0), pos) # (4, 2/1/2) , # +np.pi/2, because our 0 degree means 90
            # NOTE: no need to add pi/2: obj already starts facing pos y, we directly rotate from that
        bboxedge = [(corners[2], corners[0]), (corners[0], corners[1]), (corners[1], corners[3]), (corners[3], corners[2])]
                    # starting from bottom left corner
        return corners, bboxedge

    @staticmethod
    def get_xyminmax(ptxy):
        """ptxy: [numpt, 2] numpy array"""
        return np.amin(ptxy[:,0]), np.amax(ptxy[:,0]), np.amin(ptxy[:,1]), np.amax(ptxy[:,1])

    ## HELPER FUNCTION: not agnostic of specific dataset configs
    def emd_by_class(self, noisy_pos, clean_pos, clean_ang, clean_sha, nobjs, same_class):
        """ For each scene, for each object, assign it a target object of the same class based on its position.
            Performs earthmover distance assignment based on pos from noisy to target, for instances of one class, 
            and assign ang correspondingly.

            pos/ang/sha: [batch_size, maxnumobj, ang/pos/sha(siz+cla)_dim]
            nobjs: [batch_size]
            clean_sha: noisy and target (clean) should share same shape data (size and class don't change)
        """
        numscene = noisy_pos.shape[0]
        noisy_labels = np.zeros((numscene, self.maxnobj, self.pos_dim+self.ang_dim))
        for scene_i in range(numscene):
            nobj = nobjs[scene_i]
            cla_idx = np.argmax(clean_sha[scene_i, :nobj, self.siz_dim:], axis=1) # (nobj, cla_dim) -> (nobj,) # example: array([ 9, 11, 15,  7,  2, 18, 15])
            for group in same_class:
                cla_idx = np.where(np.isin(cla_idx + 1, group), group[0] - 1, cla_idx)
            for c in np.unique(cla_idx) : # example unique out: array([ 2,  7,  9, 11, 15, 18]) (indices of the 1 in one-hot encoding)
                objs = np.where(cla_idx==c)[0] # 1d array of obj indices whose class is c # example: array([2, 6]) for c=15
                p1 = [tuple(pt) for pt in noisy_pos[scene_i, objs, :]] # array of len(objs) tuples
                p2 = [tuple(pt) for pt in clean_pos[scene_i, objs, :]] # array of len(objs) tuples
                chair_assignment, chair_assign_ang = earthmover_assignment(p1, p2, clean_ang[scene_i, objs, :]) # len(objs)x2: assigned pos for each pt in p1 (in that order)
                noisy_labels[scene_i, objs, 0:self.pos_dim] = np.array(chair_assignment)
                noisy_labels[scene_i, objs, self.pos_dim:self.pos_dim+self.ang_dim] = np.array(chair_assign_ang)
        # NOTE: noisy_labels[scene_i, nobj:, :] left as 0; each obj assigned exactly once
        return noisy_labels
    
    ## HELPER FUNCTION: not agnostic of specific dataset configs
    def move_less(self, noisy_pos, clean_pos, nobjs, clean_word):
        """ For each scene, we want to find the closest configuration for each noisy sample. Thus we can move less.

            pos: [batch_size, maxnumobj, pos_dim]
            nobjs: [batch_size]
            clean_sha: noisy and target (clean) should share same shape data (size and class don't change)
        """
        numscene = noisy_pos.shape[0]
        for scene_i in range(numscene):
            word = clean_word[scene_i]
            if word[0] == 2 or word[0] == 3:
                nobj = nobjs[scene_i]
                p1 = clean_pos[scene_i, :nobj, :]
                p2 = noisy_pos[scene_i, :nobj, :]
                delta = np.mean(p2 - p1, axis=0)
                clean_pos[scene_i, :nobj, [word[0] - 2]] += delta[[word[0] - 2], ]
        return clean_pos

    def is_valid(self, o_i, f_i, o_pos, o_ang, scene_pos, scene_ang, scene_sha, scene_fpoc, within_floorplan=True, no_penetration=True, pen_siz_scale=0.92):
        """ A object's pos + ang is valid if the object's bounding box does not intersect with any floor plan wall or other object's bounding box edge.
            Note this function modifies the input arguments in place.

            o_i: scalar, the index of the object of interest in the scene
            o_{pos, ang}: [1, dim], information about the object being repositioned
            scene_{pos, ang, sha}: [nobj, dim], info about the rest of the scene, without padding, the obj_ith entry of pos and ang is skipped
            scene_fpoc: [nfpoc, pos_dim], without padding, ordered (consecutive points form lines).
            pen_siz_scale: to allow for some minor intersection (respecting ground truth dataset)
        """
        # Dnormalize data to the same scale: [-3, 3] (meters) for both x and y(z) axes.
        room_size = np.array(self.room_size) #[x, y, z]
        o_pos = o_pos*room_size[[0,1]]/2  #[1, dim]
        scene_fpoc = scene_fpoc* room_size[[0,1]]/2 # from [-1,1] to [-3,3]
        scene_pos = scene_pos*room_size[[0,1]]/2 # [nobj, pos_dim]
        scene_ang = trig2ang(scene_ang) #[nobj, 1], in [-pi, pi]
        scene_siz = (scene_sha[:,:self.siz_dim] +1) * (room_size[[0,1]]/2)  # [nobj, siz_dim], bbox len 
    
        # check intersection with floor plan
        if within_floorplan:
            # check if obj o's pos and corners is outside floor plan's convex bounds
            fp_x_min, fp_x_max, fp_y_min, fp_y_max = TableDataset.get_xyminmax(scene_fpoc)
            if ((o_pos[0,0]<fp_x_min) or (o_pos[0,0]>fp_x_max)or (o_pos[0,1]<fp_y_min) or (o_pos[0,1]>fp_y_max)): return False
            o_corners, o_bboxedge = TableDataset.get_objbbox_corneredge(o_pos[0], trig2ang(o_ang), scene_siz[o_i]) # cor=(4,2)
            o_cor_x_min, o_cor_x_max, o_cor_y_min, o_cor_y_max = TableDataset.get_xyminmax(o_corners)
            if ((o_cor_x_min<fp_x_min) or (o_cor_x_max>fp_x_max)or(o_cor_y_min)<fp_y_min) or (o_cor_y_max>fp_y_max): return False

        # check intersection with each of the other objects
        if no_penetration:
            o_scale_corners, o_scale_bboxedge = TableDataset.get_objbbox_corneredge(o_pos[0], trig2ang(o_ang), scene_siz[o_i]*pen_siz_scale)
            o_scale_cor_x_min, o_scale_cor_x_max, o_scale_cor_y_min, o_scale_cor_y_max = TableDataset.get_xyminmax(o_scale_corners)

            for other_o_i in range (scene_pos.shape[0]):
                if other_o_i == o_i: continue # do not compare against itself
                if other_o_i == f_i: continue # do not compare against floor
                other_scale_cor, other_scale_edg = TableDataset.get_objbbox_corneredge(scene_pos[other_o_i], scene_ang[other_o_i:other_o_i+1,:], scene_siz[other_o_i]*pen_siz_scale)
                other_scale_cor_x_min, other_scale_cor_x_max, other_scale_cor_y_min, other_scale_cor_y_max = TableDataset.get_xyminmax(other_scale_cor)
                
                # check entire outside one another
                if ((o_scale_cor_x_max<=other_scale_cor_x_min) or (o_scale_cor_x_min>=other_scale_cor_x_max) or
                    (o_scale_cor_y_max<=other_scale_cor_y_min) or (o_scale_cor_y_min>=other_scale_cor_y_max)):
                   continue # go check next obj

                # check if one is inside the other:
                if ((other_scale_cor_x_min <= o_scale_cor_x_min <= other_scale_cor_x_max) and (other_scale_cor_x_min <= o_scale_cor_x_max <= other_scale_cor_x_max) and
                    (other_scale_cor_y_min <= o_scale_cor_y_min <= other_scale_cor_y_max) and (other_scale_cor_y_min <= o_scale_cor_y_max <= other_scale_cor_y_max)):
                    return False
                if ((o_scale_cor_x_min <= other_scale_cor_x_min <= o_scale_cor_x_max) and (o_scale_cor_x_min <= other_scale_cor_x_max <= o_scale_cor_x_max) and
                    (o_scale_cor_y_min <= other_scale_cor_y_min <= o_scale_cor_y_max) and (o_scale_cor_y_min <= other_scale_cor_y_max <= o_scale_cor_y_max)):
                    return False
                # check if edges intersect
                for edge_i in range(4):
                    for other_edge_i in range(4):
                        if do_intersect(o_scale_bboxedge[edge_i][0], o_scale_bboxedge[edge_i][1], 
                                        other_scale_edg[other_edge_i][0], other_scale_edg[other_edge_i][1]):
                            return False

        return True

    def clever_add_noise(self, noisy_orig_pos, noisy_orig_ang, noisy_orig_sha, noisy_orig_nobj, floor_mask, no_translate_mask, no_rotate_mask, noisy_orig_vol, 
                         noise_level_stddev, angle_noise_level_stddev, t, noise_schedule, weigh_by_class=False, within_floorplan=False, no_penetration=False, max_try=None, pen_siz_scale=0.92):
        """ noisy_orig_pos/ang/sha: [batch_size, maxnobj, pos_dim/ang_dim/sha_dim]
            noisy_orig_fpoc:        [batch_size, maxnfpoc, pos_dim]
            noisy_orig_vol:         used only if weigh_by_class
        """
        if not weigh_by_class and not within_floorplan and not no_penetration:
            # NOTE: each scene has zero-mean gaussian distributions for noise
            # noisy_pos = add_gaussian_gaussian_noise_by_class("chair', noisy_orig_pos, noisy_orig_sha, noise_level_stddev=noise_level_stddev)
            noise_level = noise_schedule.sqrt_one_minus_alphas_cumprod.gather(-1, t.cpu())
            scene_noise_level = noise_level_stddev * noise_level
            scene_angle_noise_level = angle_noise_level_stddev * noise_level
            noisy_pos = np.zeros((noisy_orig_pos.shape))
            for scene_idx in range(noisy_orig_pos.shape[0]):
                noisy_pos[scene_idx] = np_add_gaussian_noise_scale(noisy_orig_pos[scene_idx], scene_noise_level[scene_idx])
            noise_rads = np.zeros((noisy_orig_ang.shape[0], noisy_orig_ang.shape[1], 1))
            for scene_idx in range(noisy_orig_ang.shape[0]):
                noise_rads[scene_idx] = np.random.normal(size=(noisy_orig_ang.shape[1], 1), loc=0.0, scale=scene_angle_noise_level[scene_idx])
            noisy_ang = np_rotate_wrapper(noisy_orig_ang, noise_rads)
            # noisy_pos = np_add_gaussian_gaussian_noise(noisy_orig_pos, noise_level_stddev=noise_level_stddev)
            # noisy_ang = np_add_gaussian_gaussian_angle_noise(noisy_orig_ang, noise_level_stddev=angle_noise_level_stddev) # [batch_size, maxnumobj, 2], stay normalized
            
            static_mask_pos = np.tile(np.expand_dims(no_translate_mask, axis=-1), (1, 1, self.pos_dim))
            static_mask_ang = np.tile(np.expand_dims(no_rotate_mask, axis=-1), (1, 1, self.ang_dim))
            noisy_pos = np.where(static_mask_pos, noisy_orig_pos, noisy_pos)
            noisy_ang = np.where(static_mask_ang, noisy_orig_ang, noisy_ang)
            noisy_pos, noisy_ang = TableDataset.reset_padding(noisy_orig_nobj, noisy_pos), TableDataset.reset_padding(noisy_orig_nobj, noisy_ang) # [batch_size, maxnumobj, dim]
            return noisy_pos, noisy_ang
                
        if max_try==None: 
            max_try=1 # weigh_by_class only, exactly 1 iteration through while loop (always reach break in first iter)
            if within_floorplan: max_try+= 1000
            if no_penetration: max_try+= 2000 # very few in range 300-1500 for bedroom

        noisy_pos = np.copy(noisy_orig_pos) # the most up to date arrangement, with 0 padded
        noisy_ang = np.copy(noisy_orig_ang)
        obj_noise_factor = 1 # overriden if weighing by class
        
        if weigh_by_class: obj_noise_factors = 1/np.sqrt((noisy_orig_vol+0.00001)*2) # [batch_size, maxnobj, 1], intuition: <1 for large objects, >1 for small objects
            # Purpose of *2: so not as extreme: vol<2, factor < 1/vol (not too large); > 2, factor > 1/vol (not too small)
        # Ending value are inf, but not used as we only consider up until nobj
        
        for scene_i in range(noisy_orig_pos.shape[0]): # each scene has its own noise level
            # NOTE: each scene has zero-mean gaussian distributions for noise and noise level
            scene_noise_level = abs(np.random.normal(loc=0.0, scale=noise_level_stddev)) # 68% in one stddev
            scene_angle_noise_level = abs(np.random.normal(loc=0.0, scale=angle_noise_level_stddev))

            # noise_level = noise_schedule.sqrt_one_minus_alphas_cumprod[t[scene_i]].item()
            # scene_noise_level = noise_level_stddev * noise_level
            # scene_angle_noise_level = angle_noise_level_stddev * noise_level

            # floor_idx = floor_mask[scene_i]
            floor_idx = -1
            # noisy_orig_fpoc, _ = TableDataset.get_objbbox_corneredge(noisy_orig_pos[scene_i, floor_idx, :], trig2ang(noisy_orig_ang[scene_i, [floor_idx]]), noisy_orig_sha[scene_i, floor_idx,:self.siz_dim] +1)
            noisy_orig_fpoc, _ = TableDataset.get_objbbox_corneredge(floor_mask[scene_i, 0, :self.pos_dim], trig2ang(floor_mask[scene_i, [0], self.pos_dim:self.pos_dim+self.ang_dim]), floor_mask[scene_i, 0, self.pos_dim+self.ang_dim:self.pos_dim+self.ang_dim+self.siz_dim] +1)

            # parse_nobj, cla_idx = TableDataset.parse_cla(noisy_orig_sha[scene_i, :, self.siz_dim:]) # generated input, never perturbed
            obj_indices = list(range(noisy_orig_nobj[scene_i]))
            random.shuffle(obj_indices) # shuffle in place
            for obj_i in obj_indices: # 0 padding unchanged
                # if "chair" not in self.object_types[cla_idx[obj_i]]: continue # only add noise for chairs
                if no_translate_mask[scene_i, obj_i] and no_rotate_mask[scene_i, obj_i]:
                    continue

                if weigh_by_class: obj_noise_factor = obj_noise_factors[scene_i, obj_i, 0] # larger objects have smaller noise
                # print(f"\n--- obj_i={obj_i}: nf={nf}")
                try_count = -1
                while True:
                    try_count += 1
                    if try_count >= max_try: 
                        # print(f"while loop counter={try_count}")
                        break
                    obj_noise = obj_noise_factor * np.random.normal(size=(noisy_orig_pos[scene_i, obj_i:obj_i+1, :]).shape, loc=0.0, scale=scene_noise_level) # [1, pos_dim]
                    new_o_pos = noisy_orig_pos[scene_i, obj_i:obj_i+1, :] + obj_noise # [1, pos_dim]

                    if no_rotate_mask[scene_i, obj_i]:
                        new_o_ang = np.copy(noisy_orig_ang[scene_i, obj_i:obj_i+1, :])
                    else:
                        obj_angle_noise = obj_noise_factor * np.random.normal(size=(1,1), loc=0.0, scale=scene_angle_noise_level) 
                        new_o_ang = np_rotate(noisy_orig_ang[scene_i, obj_i:obj_i+1, :], obj_angle_noise) # [1, ang_dim=2]

                    if within_floorplan or no_penetration: # if can skip, will always break
                        if not self.is_valid( obj_i, floor_idx, np.copy(new_o_pos), np.copy(new_o_ang),
                                            np.copy(noisy_pos[scene_i, :noisy_orig_nobj[scene_i], :]), np.copy(noisy_ang[scene_i, :noisy_orig_nobj[scene_i], :]),  # latest state
                                            np.copy(noisy_orig_sha[scene_i, :noisy_orig_nobj[scene_i], :]), np.copy(noisy_orig_fpoc), 
                                            within_floorplan=within_floorplan, no_penetration=no_penetration, pen_siz_scale=pen_siz_scale):
                            continue # try regenerating noise

                    # reached here means passed checks
                    noisy_pos[scene_i, obj_i:obj_i+1, :] = new_o_pos
                    noisy_ang[scene_i, obj_i:obj_i+1, :] = new_o_ang

                    # print(f"break @ try_count={try_count}")
                    break # continue to next object
      
        return noisy_pos, noisy_ang
    
    def augment_data(self, orig_data, orig_mask, orig_word, data_augment=1, floorplan_class=1, within_floorplan=False, max_try=None):
        """ noisy_orig_pos/ang/sha: [batch_size, maxnobj, pos_dim/ang_dim/sha_dim]
            noisy_orig_fpoc:        [batch_size, maxnfpoc, pos_dim]
            noisy_orig_vol:         used only if weigh_by_class
        """
        clean_nobj = np.sum(np.sum(orig_data, axis=-1) != 0, axis=1)
        clean_nmask = np.sum(np.sum(orig_mask, axis=-1) != 0, axis=1)
        clean_pos, clean_ang, clean_sha = orig_data[:, :, :self.pos_dim], orig_data[:, :, self.pos_dim:self.pos_dim+self.ang_dim], orig_data[:, :, self.pos_dim+self.ang_dim:]
        mask_pos, mask_ang, mask_sha = orig_mask[:, :, :self.pos_dim], orig_mask[:, :, self.pos_dim:self.pos_dim+self.ang_dim], orig_mask[:, :, self.pos_dim+self.ang_dim:]
        assert floorplan_class > 0
        floor_index = np.argmax(orig_mask[:, :, self.pos_dim+self.ang_dim+self.siz_dim] == floorplan_class, axis=-1)
        floor_index = floor_index[:, np.newaxis, np.newaxis]
        floor_mask = np.take_along_axis(orig_mask, floor_index, axis=1)
                
        if max_try==None: 
            max_try=1 # weigh_by_class only, exactly 1 iteration through while loop (always reach break in first iter)
            if within_floorplan: max_try+= 1000

        aug_data, aug_mask = [], []
        for _ in range(data_augment):
            noisy_pos = np.copy(clean_pos) # the most up to date arrangement, with 0 padded
            noisy_mask = np.copy(mask_pos)
            for scene_i in range(orig_data.shape[0]): # each scene has its own noise level
                # NOTE: each scene has zero-mean gaussian distributions for noise and noise level
                floor_idx = -1
                noisy_orig_fpoc, _ = TableDataset.get_objbbox_corneredge(floor_mask[scene_i, 0, :self.pos_dim], trig2ang(floor_mask[scene_i, [0], self.pos_dim:self.pos_dim+self.ang_dim]), floor_mask[scene_i, 0, self.pos_dim+self.ang_dim:self.pos_dim+self.ang_dim+self.siz_dim] +1)
                try_count = -1

                word = orig_word[scene_i]
                if word[0] < 4:
                    try_count = max_try
                
                while True:
                    try_count += 1
                    if try_count >= max_try: 
                        # print(f"while loop counter={try_count}")
                        aug_data.append(np.concatenate([noisy_pos[[scene_i]], clean_ang[[scene_i]], clean_sha[[scene_i]]], axis=-1))
                        aug_mask.append(np.concatenate([noisy_mask[[scene_i]], mask_ang[[scene_i]], mask_sha[[scene_i]]], axis=-1))
                        break
                    obj_noise = np.random.normal(size=(noisy_pos[scene_i, [1], :]).shape, loc=0.0, scale=0.5) # [1, pos_dim]

                    # parse_nobj, cla_idx = TableDataset.parse_cla(noisy_orig_sha[scene_i, :, self.siz_dim:]) # generated input, never perturbed
                    obj_indices = list(range(clean_nobj[scene_i]))
                    random.shuffle(obj_indices) # shuffle in place
                    for obj_i in obj_indices: # 0 padding unchanged
                        # print(f"\n--- obj_i={obj_i}: nf={nf}")
                        new_o_pos = clean_pos[scene_i, obj_i:obj_i+1, :] + obj_noise # [1, pos_dim]
                        # reached here means passed checks
                        noisy_pos[scene_i, obj_i:obj_i+1, :] = new_o_pos

                    mask_index = floor_index[scene_i, 0, 0]
                    mask_indices = list(range(clean_nmask[scene_i]))
                    random.shuffle(mask_indices) # shuffle in place
                    for obj_i in mask_indices: # 0 padding unchanged
                        if obj_i == mask_index: continue
                        # print(f"\n--- obj_i={obj_i}: nf={nf}")
                        new_o_pos = mask_pos[scene_i, obj_i:obj_i+1, :] + obj_noise # [1, pos_dim]
                        # reached here means passed checks
                        noisy_mask[scene_i, obj_i:obj_i+1, :] = new_o_pos

                    flag = True
                    if within_floorplan: # if can skip, will always break
                        for obj_i in obj_indices:
                            if not self.is_valid( obj_i, floor_idx, np.copy(noisy_pos[scene_i, obj_i:obj_i+1, :]), np.copy(clean_ang[scene_i, obj_i:obj_i+1, :]),
                                                np.copy(noisy_pos[scene_i, :clean_nobj[scene_i], :]), np.copy(clean_ang[scene_i, :clean_nobj[scene_i], :]),  # latest state
                                                np.copy(clean_sha[scene_i, :clean_nobj[scene_i], :]), np.copy(noisy_orig_fpoc), 
                                                within_floorplan=within_floorplan, no_penetration=False):
                                flag = False
                                break
                        for obj_i in mask_indices:
                            if obj_i == mask_index: continue
                            if not self.is_valid( obj_i, mask_index, np.copy(noisy_mask[scene_i, obj_i:obj_i+1, :]), np.copy(mask_ang[scene_i, obj_i:obj_i+1, :]),
                                                np.copy(noisy_mask[scene_i, :clean_nobj[scene_i], :]), np.copy(mask_ang[scene_i, :clean_nobj[scene_i], :]),  # latest state
                                                np.copy(mask_sha[scene_i, :clean_nobj[scene_i], :]), np.copy(noisy_orig_fpoc), 
                                                within_floorplan=within_floorplan, no_penetration=False):
                                flag = False
                                break
                    if flag:
                        aug_data.append(np.concatenate([noisy_pos[[scene_i]], clean_ang[[scene_i]], clean_sha[[scene_i]]], axis=-1))
                        aug_mask.append(np.concatenate([noisy_mask[[scene_i]], mask_ang[[scene_i]], mask_sha[[scene_i]]], axis=-1))
                        break # continue to next object

        return np.concatenate(aug_data, axis=0), np.concatenate(aug_mask, axis=0)

    def _gen_kitchen_batch_preload(self, batch_size, data_partition='train', random_idx=None):
        """ Reads from preprocessed data npz files (already normalized) to return data for batch_size number of scenes.
            Variable data length is dealt with through padding with 0s at the end.

            random_idx: if given, selects these designated scenes from the set of all trainval or test data.
            
            Returns:
            batch_scenepaths: [batch_size], contains full path to the directory named as the scenepath (example
                             scenepath = '<scenedir>/004f900c-468a-4f70-83cc-aa2c98875264_SecondBedroom-27399')
            batch_nbj: [batch_size], contains numbers of objects for each scene/room.

            batch_pos: position has size [batch_size, maxnumobj, pos_dim]=[x, y], where [:,0:2,:] are the 2 tables,
                       and the rest are the chairs.
            batch_ang: [batch_size, maxnumobj, ang_dim=[cos(th), sin(th)] ]
            batch_sha: [batch_size, maxnumobj, siz_dim+cla_dim], represents bounding box lengths and
                       class of object/furniture with one hot encoding.
            
            batch_vol: [batch_size, maxnumobj], volume of each object's bounding box (in global absolute scale).              
        """
        if data_partition == "train":
            data = self.data_tv
            mask = self.mask_tv
            cond = self.cond_tv
            word = self.word_tv
            file = self.file_tv
        elif data_partition == "test":
            data = self.data_test
            mask = self.mask_test
            cond = self.cond_test
            word = self.word_test
            file = self.file_test
        random_idx = np.random.choice(len(data), size=batch_size, replace=False) if random_idx is None else random_idx
        assert random_idx.shape[0] == batch_size

        return data[random_idx], mask[random_idx], [cond[i] for i in random_idx], word[random_idx], [file[i] for i in random_idx]

    def gen_kitchen(self, batch_size, t, noise_schedule, no_rotate_class=[4, 5, 7], static_class=[], same_class=[], floorplan_class=1, random_idx=None, data_partition='train', use_emd=True, use_move_less=True, 
                        abs_pos=True, abs_ang=True, noise_level_stddev=0.1, angle_noise_level_stddev=np.pi/12,
                        weigh_by_class = False, within_floorplan = False, no_penetration = False, pen_siz_scale=0.92, 
                        is_classification=False, data_augment = 1):
            """ Main entry point for generating data form the YCB_kitchen dataset.

                batch_size: number of scenes
                abs_pos/ang: if True, the returned labels are (final ordered assigned pos/ang); otherwise, use relative pos/ang
                            = (final ordered assigned pos - initial pos, angle between final and initial angles)
                noise_level_stddev, angle_noise_level_stddev: for adding position and angle noise. 
                                                            input: {clean scenes + gaussian gaussian noise}, labels: {clean scene without noise + emd}
                pen_siz_scale: value between 0 and 1, the lower the value, the more intersection among objects it allows when adding noise
                
                Intermediary: {clean, noisy}_{pos, ang, sha} have shape [batch_size, maxnumobj, 2]

                Returns: 
                    input:  [batch_size, maxnumobj, pos+ang+siz+cla] = [x, y, cos(th), sin(th), bbox_lenx, bbox_leny, one-hot-encoding for class]
                    mask:  [batch_size, maxnmask, pos+ang+siz+1] = [x, y, cos(th), sin(th), bbox_lenx, bbox_leny, type]
                    label:  [batch_size, maxnumobj, pos+ang] = [x, y, cos(th), sin(th)]
                    padding_mask: [batch_size, maxnumobj], for Transformer (nn.TransformerEncoder) to ignore padded zeros (value=False for not masked)
                    scenepaths: [batch_size,] <class 'numpy.ndarray'> of  <class 'numpy.str_'>

                Clean input's labels are identical to the input. Noisy input's labels are ordered assignment to clean position for each 
                point (through earthmover distance assignment), using the original clean sample (noisy_orig_pos) as label, and angle labels
                are the corresponding angles from the assignment.
            
                Note that this implementation takes a non-hierarhical, non-scene-graph, free-for-all approach: emd assignment is done freely 
                among all objects of a class. 

                Also note that in generating the noisy positions, chairs are given greater noise while tables are given less. This is to
                teach the network to move certain types less.
            """
            clean_data, clean_mask, clean_cond, clean_word, clean_scenepaths = self._gen_kitchen_batch_preload(batch_size // 1, data_partition=data_partition, random_idx=random_idx if random_idx is None else random_idx[:batch_size // 1])
            if data_augment > 1:
                clean_data, clean_mask = self.augment_data(clean_data, clean_mask, clean_word, data_augment=1, floorplan_class=floorplan_class, within_floorplan=within_floorplan)
                clean_cond = clean_cond * 1
                clean_word = np.tile(clean_word, (1, 1))
                clean_scenepaths = clean_scenepaths * 1
            clean_nobj = np.sum(np.sum(clean_data, axis=-1) != 0, axis=1)
            clean_nmask = np.sum(np.sum(clean_mask, axis=-1) != 0, axis=1)
            clean_pos, clean_ang, clean_sha = clean_data[:, :, :self.pos_dim], clean_data[:, :, self.pos_dim:self.pos_dim+self.ang_dim], clean_data[:, :, self.pos_dim+self.ang_dim:]
            clean_vol = (clean_sha[:, :, 0] + 1) * (clean_sha[:, :, 1] + 1)
            clean_cla = np.argmax(clean_data[:, :, self.pos_dim+self.ang_dim+self.siz_dim:], axis=-1)
            clean_cla[np.sum(clean_data, axis=-1) == 0] = -1
            if len(static_class) > 0:
                clean_static = clean_cla == static_class[0] - 1
                for sc in static_class[1:]:
                    clean_static = np.bitwise_or(clean_static, clean_cla == sc - 1)
            else:
                clean_static = clean_cla == -2

            # special_scene = np.bitwise_and(clean_word[:, 0] == 4, clean_nmask == 1)
            # clean_static[special_scene] = np.bitwise_or(clean_static[special_scene], clean_cla[special_scene] == 3) # plate

            clean_translate = clean_static
            clean_rotate = clean_static
            for sc in no_rotate_class:
                clean_rotate = np.bitwise_or(clean_rotate, clean_cla == sc - 1)
            # clean_fpoc = np.argmax(clean_cla == floorplan_class - 1, axis=-1)
            assert floorplan_class > 0
            floor_index = np.argmax(clean_mask[:, :, self.pos_dim+self.ang_dim+self.siz_dim] == floorplan_class, axis=-1)
            floor_index = floor_index[:, np.newaxis, np.newaxis]
            clean_fpoc = np.take_along_axis(clean_mask, floor_index, axis=1)
            
            # input: pos, ang, siz, cla
            perturbed_pos, perturbed_ang = self.clever_add_noise(clean_pos, clean_ang, clean_sha, clean_nobj, clean_fpoc, clean_translate, clean_rotate, clean_vol, noise_level_stddev, angle_noise_level_stddev, 
                                                                 t, noise_schedule, weigh_by_class=weigh_by_class, within_floorplan=within_floorplan, no_penetration=no_penetration, pen_siz_scale=pen_siz_scale)

            input = np.concatenate([perturbed_pos, perturbed_ang, clean_sha], axis=2) # [batch_size, maxnumobj, dims]

            # label: pos, ang
            if is_classification: # for kitchen model evaluation
                classification_token = np.concatenate([np.array([[[0,0, 1,0, 1,1]]]), np.zeros((1,1,self.cla_dim))], axis=2) # [1, 1, 2+2+2+19=25] -> [batch_size, 1, dims=25]
                input = np.concatenate([input, np.repeat(classification_token, batch_size, axis=0)], axis=1)  # [batch_size, maxnumobj+1, pos+ang+siz+cla=dims=25]
                # classification labels: [batch_size, 1]; clean is 1, noisy is 0 = probability(clean)
                labels = np.concatenate([(np.zeros((batch_size//2, 1))+1), np.zeros((batch_size//2, 1))], axis=0)
            else:
                # The assignment does not affect the moving direction
                if use_move_less:
                    clean_pos = self.move_less(np.copy(perturbed_pos), np.copy(clean_pos), np.copy(clean_nobj), np.copy(clean_word))
                # Calculate absolute labels for noisy
                if use_emd:
                    labels = self.emd_by_class(np.copy(perturbed_pos), np.copy(clean_pos), np.copy(clean_ang), np.copy(clean_sha), np.copy(clean_nobj), same_class)
                else:
                    labels = np.copy(np.concatenate([clean_pos, clean_ang], axis=2) ) # directly use original scene
                # If needed, overwrite it with relative
                if not abs_pos:
                    labels[:,:,:self.pos_dim] = labels[:,:,:self.pos_dim] - perturbed_pos
                if not abs_ang: # relative: change noisy_labels from desired pos and ang to displacement
                    ang_diff = np_angle_between(perturbed_ang, labels[:,:,self.pos_dim:self.pos_dim+self.ang_dim]) # [batch_size, nobj, 1].
                        # both input to np angle between are normalized, from noisy_ang(noisy_input) to noisy_labels(noisy_orig_ang), in [-pi, pi], 
                    labels[:,:,self.pos_dim:self.pos_dim+1]              = np.cos(ang_diff) # [batch_size, nobj, 1]
                    labels[:,:,self.pos_dim+1:self.pos_dim+self.ang_dim] = np.sin(ang_diff) # [batch_size, nobj, 1]


            # padding mask: for Transformer to ignore padded zeros
            padding_mask = np.repeat([np.arange(self.maxnobj)], batch_size, axis=0) #[batch_size,maxnobj]
            padding_mask = (padding_mask>=clean_nobj.reshape(-1,1)) # [batch_size,maxnobj] >= [batch_size, 1]: [batch_size,maxnobj] 
                # False=not masked (unchanged), True=masked, not attended to (isPadding)
            mask_padding_mask = np.repeat([np.arange(self.maxnmask)], batch_size, axis=0) #[batch_size,maxnmask]
            mask_padding_mask = (mask_padding_mask>=clean_nmask.reshape(-1,1)) # [batch_size,maxnmask] >= [batch_size, 1]: [batch_size,maxnmask] 
                # False=not masked (unchanged), True=masked, not attended to (isPadding)
            if is_classification: # add to padding for classifcation token at the end (not masked: false==0)
                padding_mask = np.concatenate([padding_mask, np.zeros((batch_size, 1))], axis=1) # [batch_size, maxnobj+1]

            return input, clean_mask, clean_cond, clean_word, labels, padding_mask, mask_padding_mask, clean_scenepaths