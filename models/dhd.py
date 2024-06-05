

import os
import torch
import torch.distributed as dist

import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, List
from torchvision.ops.boxes import batched_nms, box_iou

from models.ops import binary_focal_loss_with_logits
from models.interaction_head import InteractionHead
from CLIP import clip
import numpy as np
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict

import sys
sys.path.append('GroundingDINO')
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util import box_ops, get_tokenlizer
from GroundingDINO.groundingdino.util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    interpolate,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from hicodet.static_hico import HICO_INTERACTIONS, ACT_TO_ING, TRAIN_VERB_NUMS, HOI_IDX_TO_OBJ_IDX, TRAIN_HOI_NUMS, OBJ_IDX_TO_OBJ_NAME,\
                            ACT_IDX_TO_ACT_NAME, UA_HOI_IDX, UC_HOI_IDX
from vcoco.static_vcoco import VCOCO_INTERACTIONS, vcoco_hoi_text_label
from vg.meta import VG_INTERACTIONS, VG_VERBS


NO_INTERACTION = [9, 23, 30, 45, 53, 64, 75, 85, 91, 95, 106, 110, 128, 145, 159, 169, 
                  173, 185, 193, 197, 207, 213, 223, 231, 234, 238, 242, 246, 251, 256, 
                  263, 272, 282, 289, 294, 304, 312, 324, 329, 335, 341, 347, 351, 355, 
                  362, 367, 375, 382, 388, 392, 396, 406, 413, 417, 428, 433, 437, 444, 
                  448, 452, 462, 473, 482, 487, 501, 505, 515, 527, 532, 537, 545, 549, 
                  557, 561, 566, 575, 583, 587, 594, 599]

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, model, targets, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        OBJECTS = OBJ_IDX_TO_OBJ_NAME
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        logits = out_logits.sigmoid()  # (bs, nq, 256)
        boxes = out_bbox  # (bs, nq, 4)
        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :] # (4, 900, 4)

        # filter output
        box_threshold = 0.2
        text_threshold = 0.2


        # get phrase
        tokenlizer = model.tokenizer 
        tokenized = tokenlizer(targets[0]['caption'])

        # build pred phrase
        labels_b = []
        scores_b = []
        bboxes_b = []
        
        for b in range(len(logits)):
            logits_filt = logits.clone()
            boxes_filt = boxes.clone()
            filt_mask = logits_filt[b].max(dim=-1)[0] > box_threshold
            # print(logits_filt.max(dim=-1)[0])
            logits_filt = logits_filt[b][filt_mask]  # bs, num_filt, 256
            boxes_filt = boxes_filt[b][filt_mask]  # bs, num_filt, 4
            labels = []
            scores = []
            bboxes = []

            for logit, box in zip(logits_filt, boxes_filt):
                pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
                if pred_phrase in OBJECTS:
                    labels.append(OBJECTS.index(pred_phrase))
                    scores.append(logit.max().item())
                    bboxes.append(box)

            labels_b.append(labels)
            scores_b.append(scores)
            if bboxes == []:
                bboxes_b.append(torch.zeros((0, 4)))   
            else:
                bboxes_b.append(torch.stack(bboxes))

        results = [{'scores': torch.Tensor(s).to(boxes.device), 'labels': torch.Tensor(l).to(boxes.device).type(torch.int64), 'boxes': b.to(boxes.device)} for s, l, b in zip(scores_b, labels_b, bboxes_b)]

        return results

class DHD(nn.Module):
    """
    Unary-pairwise transformer

    Parameters:
    -----------
    detector: nn.Module
        Object detector (DETR)
    postprocessor: nn.Module
        Postprocessor for the object detector
    interaction_head: nn.Module
        Interaction head of the network
    human_idx: int
        Index of the human class
    num_classes: int
        Number of action classes
    alpha: float
        Hyper-parameter in the focal loss
    gamma: float
        Hyper-parameter in the focal loss
    box_score_thresh: float
        Threshold used to eliminate low-confidence objects
    fg_iou_thresh: float
        Threshold used to associate detections with ground truth
    min_instances: float
        Minimum number of instances (human or object) to sample
    max_instances: float
        Maximum number of instances (human or object) to sample
    """
    def __init__(self,
        detector: nn.Module,
        postprocessor: nn.Module,
        interaction_head: nn.Module,
        human_idx: int, num_classes: int,
        alpha: float = 0.5, gamma: float = 2.0,
        box_score_thresh: float = 0.2, fg_iou_thresh: float = 0.5,
        min_instances: int = 3, max_instances: int = 15, 
        binary=False, binary_thres: float=0.5, teacher_model: nn.Module=None, eval_knn: bool=False, roi_size=7, args=None
    ) -> None:
        super().__init__()
        self.detector = detector
        self.postprocessor = postprocessor
        self.interaction_head = interaction_head

        self.human_idx = human_idx
        self.num_classes = num_classes

        self.alpha = alpha
        self.gamma = gamma

        self.box_score_thresh = box_score_thresh
        self.fg_iou_thresh = fg_iou_thresh

        self.min_instances = min_instances
        self.max_instances = max_instances

        self.binary = binary
        self.binary_thres = binary_thres
        self.teacher_model=teacher_model

        # self.hoi_weight = self.cal_weights(TRAIN_HOI_NUMS)
        self.eval_knn = eval_knn
        # if eval_knn: self.init_knn_features()
        sampling_ratio = 2
        self.box_roi_pool = MultiScaleRoIAlign(featmap_names=['0'], output_size=roi_size, sampling_ratio=sampling_ratio)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.init_text_features(args.dataset)
        # self.init_swig_features()
        self.training_type = args.training_type
        self.neg_pair_weight = args.neg_pair_wieght
        self.use_cache_box = args.use_cache_box

    def recover_boxes(self, boxes, size):
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        h, w = size
        scale_fct = torch.stack([w, h, w, h])
        boxes = boxes * scale_fct
        return boxes

    def associate_with_ground_truth(self, boxes_h, boxes_o, targets):
        n = boxes_h.shape[0]
        labels = torch.zeros(n, self.num_classes, device=boxes_h.device)
        
        gt_bx_h = self.recover_boxes(targets['boxes_h'], targets['size'])
        gt_bx_o = self.recover_boxes(targets['boxes_o'], targets['size'])

        x, y = torch.nonzero(torch.min(
            box_iou(boxes_h, gt_bx_h),
            box_iou(boxes_o, gt_bx_o)
        ) >= self.fg_iou_thresh).unbind(1)

        invalid_nx, _ = torch.nonzero(torch.max(
            box_iou(boxes_h, gt_bx_h),
            box_iou(boxes_o, gt_bx_o)
        ) < self.fg_iou_thresh).unbind(1)
        
        # labels[x, targets['labels'][y]] = 1
        labels[x, targets['verb'][y]] = 1
        # labels[: , NO_INTERACTION] = 0
        masks = torch.zeros_like(labels)
        if targets['dataset'] == 0:
            masks[:, :len(HICO_INTERACTIONS)] = 1
        elif targets['dataset'] == 1:
            masks[:, len(HICO_INTERACTIONS): len(HICO_INTERACTIONS) + len(VCOCO_INTERACTIONS)] = 1
        elif targets['dataset'] == 2:
            masks[:, len(HICO_INTERACTIONS) + len(VCOCO_INTERACTIONS):] = 1


        weights = torch.ones_like(labels)
        weights[invalid_nx] = self.neg_pair_weight
  
        # if self.training_type == 'ua':
        #     dele = torch.mean(labels[:, UA_HOI_IDX], dim=-1) > 0
        #     masks[dele, :] = 0
        # elif self.training_type == 'nfuc':
        #     dele = torch.mean(labels[:, UC_HOI_IDX['non_rare_first']], dim=-1) > 0
        #     masks[dele, :] = 0
        
        return labels, masks, weights

    def associate_with_ground_truth_bh(self, boxes_h, boxes_o, targets):
        n = boxes_h.shape[0]
        
        labels = torch.zeros(n, len(self.verb_text_features), device=boxes_h.device)

        gt_bx_h = self.recover_boxes(targets['boxes_h'], targets['size'])
        # gt_bx_o = self.recover_boxes(targets['boxes_o'], targets['size'])

        x, y = torch.nonzero(
            box_iou(boxes_h, gt_bx_h) >= self.fg_iou_thresh).unbind(1)
        
        invalid_nx, _ = torch.nonzero(
            box_iou(boxes_h, gt_bx_h) < self.fg_iou_thresh).unbind(1)

        labels[x, targets['verb'][y]] = 1
        # labels[:, 57] = 0
        masks = torch.zeros_like(labels)
        if targets['dataset'] == 0:
            masks[:, :117] = 1
        elif targets['dataset'] == 1:
            masks[:, 117: 117 + 24] = 1
        elif targets['dataset'] == 2:
            masks[:, 117 + 24:] = 1

        weights = torch.ones_like(labels)
        weights[invalid_nx] = self.neg_pair_weight
        # tmp_labels = torch.zeros(n, self.num_classes, device=boxes_h.device)
        # tmp_labels[x, targets['labels'][y]] = 1
        # if self.training_type == 'ua':
        #     dele = torch.mean(tmp_labels[:, UA_HOI_IDX], dim=-1) > 0
        #     masks[dele, :] = 0
        # elif self.training_type == 'nfuc':
        #     dele = torch.mean(tmp_labels[:, UC_HOI_IDX['non_rare_first']], dim=-1) > 0
        #     masks[dele, :] = 0

        return labels, masks, weights
    
    def associate_with_ground_truth_binary(self, boxes_h, boxes_o, targets):
        n = boxes_h.shape[0]
        labels = torch.zeros(n, self.num_classes, device=boxes_h.device)

        gt_bx_h = self.recover_boxes(targets['boxes_h'], targets['size'])
        gt_bx_o = self.recover_boxes(targets['boxes_o'], targets['size'])

        x, y = torch.nonzero(torch.min(
            box_iou(boxes_h, gt_bx_h),
            box_iou(boxes_o, gt_bx_o)
        ) >= self.fg_iou_thresh).unbind(1)

        labels[x, targets['hoi'][y]] = 1

        # hico no_interaction cls
        labels[: , NO_INTERACTION] = 0
        # if targets['dataset'] == 0: 
        #     labels[:, 57] = 0
            
        labels = torch.tensor(torch.mean(labels, dim=-1) > 0, dtype=torch.int64)

        return labels

    def compute_interaction_binary_loss(self, boxes, bh, bo, logits, prior, targets):
        labels = torch.cat([
            self.associate_with_ground_truth_binary(bx[h], bx[o], target)
            for bx, h, o, target in zip(boxes, bh, bo, targets)
        ])

        n_p = len(torch.nonzero(labels))

        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

        ce_loss = F.cross_entropy(logits, labels, reduction='none')

        # Calculate the modulating factor
        pt = torch.exp(-ce_loss)
        alpha = self.alpha
        if alpha is not None:
            alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
            pt = alpha_t * pt

        # Calculate the focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
    def compute_interaction_cls_loss(self, boxes, bh, bo, logits, prior, targets):
        gts_list= [
            self.associate_with_ground_truth(bx[h], bx[o], target)
            for bx, h, o, target in zip(boxes, bh, bo, targets)
        ]
        labels_list, masks_list, weights_list = zip(*gts_list)
        labels = torch.cat(labels_list)
        masks = torch.cat(masks_list)
        weights = torch.cat(weights_list)
        
        prior = torch.cat(prior, dim=1).prod(0)
        x, y = torch.nonzero(prior).unbind(1)


        logits = logits[x, y]; prior = prior[x, y]; labels = labels[x, y]; masks = masks[x,y]; weights=weights[x,y]
        
        n_p = len(torch.nonzero(labels))
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

        loss = (weights * binary_focal_loss_with_logits(
            torch.log(
                prior / (1 + torch.exp(-logits) - prior) + 1e-8
            ), labels, reduction='none',
            alpha=self.alpha, gamma=self.gamma
        ))[masks.bool()]
        loss = loss.sum()

        if n_p == 0:return loss
        return loss / n_p

    def compute_verb_loss(self, boxes, bh, bo, logits, prior, targets):
        gts_list= [
            self.associate_with_ground_truth_bh(bx[h], bx[o], target)
            for bx, h, o, target in zip(boxes, bh, bo, targets)
        ]
        labels_list, masks_list, weights_list = zip(*gts_list)
        labels = torch.cat(labels_list)
        masks = torch.cat(masks_list)
        weights = torch.cat(weights_list)

        # prior = torch.cat(prior, dim=1).prod(0)
        # x, y = torch.nonzero(prior).unbind(1)
        #
        # logits = logits[x, y]; prior = prior[x, y]; labels = labels[x, y]

        n_p = len(torch.nonzero(labels))
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

        loss = (weights * binary_focal_loss_with_logits(
            logits, labels, reduction='none',
            alpha=self.alpha, gamma=self.gamma
        ))[masks.bool()]
        loss = loss.sum()


        if n_p == 0: return loss
        return loss / n_p
    
    def prepare_region_proposals(self, results, hidden_states):
        region_props = []
        for res, hs in zip(results, hidden_states):
            sc, lb, bx = res.values()

            keep = batched_nms(bx, sc, lb, 0.5)
            sc = sc[keep].view(-1)  # object_socre
            lb = lb[keep].view(-1)  # object_class
            bx = bx[keep].view(-1, 4)   # bbox
            hs = hs[keep].view(-1, 256) # object_hidden_state

            keep = torch.nonzero(sc >= self.box_score_thresh).squeeze(1)

            is_human = lb == self.human_idx
            hum = torch.nonzero(is_human).squeeze(1)
            obj = torch.nonzero(is_human == 0).squeeze(1)
            n_human = is_human[keep].sum(); n_object = len(keep) - n_human
            # Keep the number of human and object instances in a specified interval
            if n_human < self.min_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.min_instances]
                keep_h = hum[keep_h]
            elif n_human > self.max_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.max_instances]
                keep_h = hum[keep_h]
            else:
                keep_h = torch.nonzero(is_human[keep]).squeeze(1)
                keep_h = keep[keep_h]

            if n_object < self.min_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.min_instances]
                keep_o = obj[keep_o]
            elif n_object > self.max_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.max_instances]
                keep_o = obj[keep_o]
            else:
                keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
                keep_o = keep[keep_o]

            keep = torch.cat([keep_h, keep_o])

            region_props.append(dict(
                boxes=bx[keep],
                scores=sc[keep],
                labels=lb[keep],
                hidden_states=hs[keep]
            ))

        return region_props

    def prepare_region_proposals_dino(self, results, targets):
        region_props = []

        for res, target in zip(results, targets):
            sc, lb, bx = res.values()
            
            if self.use_cache_box:
                bx = self.recover_boxes(bx, target['size'])

            is_human = lb == self.human_idx
            hum = torch.nonzero(is_human).squeeze(1)
            obj = torch.nonzero(is_human == 0).squeeze(1)

            n_human = is_human.sum(); n_object = len(is_human) - n_human

            if n_human > self.max_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.max_instances]
                keep_h = hum[keep_h]
            else:
                keep_h = hum

            if n_object > self.max_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.max_instances]
                keep_o = obj[keep_o]
            else:
                keep_o = obj

            keep = torch.cat([keep_h, keep_o])

            region_props.append(dict(
                boxes=bx[keep].view(-1, 4),
                scores=sc[keep].view(-1),
                labels=lb[keep].view(-1)
            ))
        
        return region_props

    def extract_interactive_pair(self, binary_thres, binary_logits, prior, bh, bo, objects, pair_tokens):
        n = [len(b) for b in bh]
        binary_logits = binary_logits.split(n)
        pair_tokens = pair_tokens.split(n)

        res_binary_logits, res_prior, res_bh, res_bo, res_objects, res_pair_tokens = [], [], [], [], [], []
        for h, o, bi, pr, obj, pt in zip(
            bh, bo, binary_logits, prior, objects, pair_tokens
        ):
            binary = F.softmax(bi, -1)[..., 1]
            keep = binary > binary_thres
            res_binary_logits.append(bi[keep])
            res_prior.append(pr[:, keep, :])
            res_bh.append(h[keep])
            res_bo.append(o[keep])
            res_objects.append(obj[keep])
            res_pair_tokens.append(pt[keep])
        return res_binary_logits, res_prior, res_bh, res_bo, res_objects, res_pair_tokens

    def postprocessing(self, boxes, bh, bo, binary_logits, cls_logits, verb_logits, prior, objects, attn_maps, image_sizes):
        n = [len(b) for b in bh]
        # print(n)
        detections = []
        
        binary_logits = binary_logits.split(n)
        cls_logits = cls_logits.split(n)
        verb_logits = verb_logits.split(n)
        for bx, h, o, bi, lg, vl, pr, obj, attn, size in zip(
            boxes, bh, bo, binary_logits, cls_logits, verb_logits, prior, objects, attn_maps, image_sizes
        ):
            pr = pr.prod(0)
            x, y = torch.nonzero(pr).unbind(1)  # (pair_idx, verb)
            scores = torch.sigmoid(lg[x, y])
            verb_scores = torch.sigmoid(vl[x])
            
            detections.append(dict(
                boxes=bx, pairing=torch.stack([h[x], o[x]]),
                labels=y,
                objects=obj[x], attn_maps=attn, size=size,
                scores=scores * pr[x, y],
                verb_scores=verb_scores
            ))
            if self.binary:
                binary = F.softmax(bi, -1)[..., 1]
                detections[-1]['scores'] = detections[-1]['scores'] * binary[x]
                detections[-1].update(dict(bi_pairing=torch.stack([h, o]),
                                            bi_scores=binary,
                                            bi_objects=obj, 
                                            bi_labels=(binary>self.binary_thres).float(),))

        return detections
    
    def init_text_features(self, dataset):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = 'cpu'
        ao_pair = [(d['action'], d['object']) for d in HICO_INTERACTIONS]
        
        if 'hicodet' in dataset:
            prompt = ["a picture of person {} {}".format(ACT_TO_ING[a], o.replace('_', ' ')) for a, o in ao_pair]
            obj_prompt = ["a picture of {}".format(o) for o in(OBJ_IDX_TO_OBJ_NAME)]
            verb_prompt = ["a picture of person {} something".format(ACT_TO_ING[a]) for a in (ACT_IDX_TO_ACT_NAME)]
            if 'hicodet+vcoco' in dataset:
                prompt = prompt + VCOCO_INTERACTIONS
                vcoco_verb_map = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 24, 25, 27, 28]
                verb_prompt = verb_prompt + [vcoco_hoi_text_label[(a, 80)] for a in vcoco_verb_map]
                if dataset =='hicodet+vcoco+vg':
                    vg_prompt = ["a picture of person {} {}".format(a, o) for (a, o) in VG_INTERACTIONS]
                    # obj_prompt = ["a picture of {}".format(o) for o in(OBJ_IDX_TO_OBJ_NAME)]
                    vg_verb_prompt = ["a picture of person {} something".format(a) for a in VG_VERBS]
                    prompt = prompt + vg_prompt
                    verb_prompt = verb_prompt + vg_verb_prompt
        
        if dataset == 'vg':
            vg_prompt = ["a picture of person {} {}".format(a, o) for (a, o) in VG_INTERACTIONS]
            # obj_prompt = ["a picture of {}".format(o) for o in(OBJ_IDX_TO_OBJ_NAME)]
            vg_verb_prompt = ["a picture of person {} something".format(a) for a in VG_VERBS]
            prompt = vg_prompt
            verb_prompt = vg_verb_prompt
        
        if dataset == 'swig':
            prompt = ["a picture of person {}".format(inter['name']) for inter in SWIG_INTERACTIONS]
            verb_prompt = ["a picture of person {} something".format(a['name']) for a in SWIG_ACTIONS]

        
        text_inputs = torch.cat(
            [clip.tokenize(p) for p in prompt]).to(device)
        print(len(text_inputs))
        with torch.no_grad():
            if 'vg' in dataset:
                text_features = self.teacher_model.encode_text(text_inputs[:10000])
                text_features1 = self.teacher_model.encode_text(text_inputs[10000:])
                text_features = torch.cat([text_features, text_features1])
            else:
                text_features = self.teacher_model.encode_text(text_inputs)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features.float().to(device)

        
        # obj_text_inputs = torch.cat(
        #     [clip.tokenize(p) for p in obj_prompt]).to(device)
        # with torch.no_grad():
        #     obj_text_features = self.teacher_model.encode_text(obj_text_inputs)
        # obj_text_features = obj_text_features / obj_text_features.norm(dim=-1, keepdim=True)
        # self.obj_text_features = obj_text_features.float().to(device)
        self.obj_text_features = None

        
        verb_text_inputs = torch.cat(
            [clip.tokenize(p) for p in verb_prompt]).to(device)
        with torch.no_grad():
            verb_text_features = self.teacher_model.encode_text(verb_text_inputs)
        verb_text_features = verb_text_features / verb_text_features.norm(dim=-1, keepdim=True)
        self.verb_text_features = verb_text_features.float().to(device)


    
    def init_swig_features(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = 'cpu'
        # ao_pair = [(d['action'], d['object']) for d in HICO_INTERACTIONS]
        # fix_prompt = ["a picture of person {} {}".format(ACT_TO_ING[a], o.replace('_', ' ')) for a, o in ao_pair]
        fix_prompt = ["a picture of person {}".format(inter['name']) for inter in SWIG_INTERACTIONS]
        prompt = fix_promptx
        text_inputs = torch.cat(
            [clip.tokenize(p) for p in prompt]).to(device)
        with torch.no_grad():
            text_features = self.teacher_model.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features.float().to(device)

        # obj_prompt = ["a picture of {}".format(o) for o in(OBJ_IDX_TO_OBJ_NAME)]
        # obj_text_inputs = torch.cat(
        #     [clip.tokenize(p) for p in obj_prompt]).to(device)
        # with torch.no_grad():
        #     obj_text_features = self.teacher_model.encode_text(obj_text_inputs)
        # obj_text_features = obj_text_features / obj_text_features.norm(dim=-1, keepdim=True)
        # self.obj_text_features = obj_text_features.float().to(device)

    def cal_weights(self, label_nums, p=0.5):
        num_fgs = len(label_nums)
        weight = [0] * (num_fgs)
        num_all = sum(label_nums)

        for index in range(num_fgs):
            if label_nums[index] == 0: continue
            weight[index] = np.power(num_all/label_nums[index], p)

        weight = np.array(weight)
        weight = weight / np.mean(weight[weight>0])

        # weight[-1] = np.power(num_all/label_nums[-1], p) if label_nums[-1] != 0 else 0

        weight = torch.FloatTensor(weight).cuda()
        return weight
    
    def forward(self,
        images: List[Tensor],
        targets: Optional[List[dict]] = None
    ) -> List[dict]:
        """
        Parameters:
        -----------
        images: List[Tensor]
            Input images in format (C, H, W)
        targets: List[dict], optional
            Human-object interaction targets

        Returns:
        --------
        results: List[dict]
            Detected human-object interactions. Each dict has the following keys:
            `boxes`: torch.Tensor
                (N, 4) Bounding boxes for detected human and object instances
            `pairing`: torch.Tensor
                (2, M) Pairing indices, with human instance preceding the object instance
            `scores`: torch.Tensor
                (M,) Interaction score for each pair
            `labels`: torch.Tensor
                (M,) Predicted action class for each pair
            `objects`: torch.Tensor
                (M,) Predicted object class for each pair
            `attn_maps`: list
                Attention weights in the cooperative and competitive layers
            `size`: torch.Tensor
                (2,) Image height and width
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        image_sizes = torch.as_tensor([
            im.size()[-2:] for im in images
        ], device=images[0].device)

        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # gdino
        if self.use_cache_box:
            results = [{k: t[k] for k in ['scores_g', 'labels_g', 'boxes_g']} for t in targets]
        else:
            for t in targets:
                object_text = OBJ_IDX_TO_OBJ_NAME
                caption = '.'.join(object_text)
                caption = caption.lower()
                caption = caption.strip()
                if not caption.endswith("."):
                    caption = caption + "."
                t['caption'] = caption
            results = self.detector(images, targets)
            hs = results['hs']
            features = results['features']
            results = self.postprocessor(self.detector, targets, results, image_sizes)

        region_props = self.prepare_region_proposals_dino(results, targets)

        boxes = [r['boxes'] for r in region_props]

        ### clip backbone
        target_clip_inputs = torch.cat([t['clip_input'].unsqueeze(0) for t in targets])
        with torch.no_grad():
            ## scale box to new size
            bs_ratios = [(float(s[0]) / float(s_orig[0]), float(s[1]) / float(s_orig[1])) for s, s_orig in zip([img.shape[-2:] for img in target_clip_inputs], image_sizes)]
            resized_boxes = [box * torch.as_tensor([ratios[0], ratios[1], ratios[0], ratios[1]], device=box.device) for box, ratios in zip(boxes, bs_ratios)]
            ## clip from VIPLO
            global_features, patch_features = self.teacher_model.encode_image(target_clip_inputs)
            global_features = global_features @ self.teacher_model.visual.proj
            patch_features = patch_features @ self.teacher_model.visual.proj

            clip_features = OrderedDict()
            B, L, C = patch_features.shape
            clip_features['0'] = patch_features.permute(0,2,1).reshape(B, C, int(L**0.5), int(L**0.5)).to(dtype=torch.float32)
            clip_features['3'] = global_features.to(dtype=torch.float32)
            image_shapes = [img.shape[-2:] for img in target_clip_inputs]
            box_features = self.box_roi_pool(clip_features, resized_boxes, image_shapes)
            box_features = box_features.split([len(boxes[i]) for i in range(len(boxes))])
      
        binary_logits, cls_logits, verb_logits, prior, bh, bo, objects, attn_maps, new_boxes = self.interaction_head(
            image_sizes, region_props, box_features, clip_features['3'], self.obj_text_features, targets, self.text_features, self.verb_text_features
        )

        if self.training:
            interaction_cls_loss = self.compute_interaction_cls_loss(new_boxes, bh, bo, cls_logits, prior, targets)
            # verb_loss = self.compute_verb_loss(new_boxes, bh, bo, verb_logits, prior, targets)
            cls_weight = 1.
            verb_weight = 1.
            binary_weight = 1.

            loss_dict = dict(
                interaction_loss=cls_weight * interaction_cls_loss,
            )

            if self.binary:
                interaction_binary_loss = self.compute_interaction_binary_loss(boxes, bh, bo, binary_logits, prior,
                                                                            targets)
                loss_dict.update(dict(binary_loss=binary_weight * interaction_binary_loss + verb_weight * verb_loss))

            return loss_dict

        if self.eval_knn:
            detections = self.knn_postprocessing(boxes, bh, bo, binary_logits, cls_logits, prior, objects, attn_maps, image_sizes, pair_features)
        else:
            detections = self.postprocessing(boxes, bh, bo, binary_logits, cls_logits, verb_logits, prior, objects, attn_maps, image_sizes)
        
        return detections

def build_detector(args, class_corr):
    if args.use_cache_box:
        detr = None
        postprocessors = None
    else:
        dino_args = SLConfig.fromfile(args.dino_config)
        detr = build_model(dino_args)
        postprocessors = PostProcess()
        if os.path.exists(args.pretrained):
            if dist.get_rank() == 0:
                print(f"Load weights for the object detector from {args.pretrained}")
                detr.load_state_dict(clean_state_dict(torch.load(args.pretrained, map_location='cpu')['model']), strict=False)
    
    clip_model, _ = clip.load(args.clip_model, device=args.device, jit=False)
    roi_pool_size = 7
    # output_dim = clip_model.visual.width #* roi_pool_size ** 2
    output_dim = clip_model.visual.output_dim
    teacher_model = clip_model
    
    box_head = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(output_dim * roi_pool_size ** 2, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, args.hidden_dim),
                nn.ReLU()
            )

    binary_predictor = torch.nn.Linear(args.repr_dim * 2, 2)
    # predictor = torch.nn.Linear(args.repr_dim * 2, args.num_classes)
    predictor = None
    # predictor = torch.nn.Linear(output_dim, args.num_classes)
    interaction_head = InteractionHead(
        predictor, binary_predictor, args.hidden_dim, args.repr_dim, output_dim,
        # detr.backbone[0].num_channels,
        output_dim,
        args.num_classes, args.human_idx, class_corr,
        box_head,
        args=args
    )
        
    detector = DHD(
        detr, postprocessors, interaction_head,
        human_idx=args.human_idx, num_classes=args.num_classes,
        alpha=args.alpha, gamma=args.gamma,
        box_score_thresh=args.box_score_thresh,
        fg_iou_thresh=args.fg_iou_thresh,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        binary=args.binary,
        binary_thres=args.binary_thres,
        teacher_model=teacher_model,
        eval_knn=args.eval_knn,
        roi_size=roi_pool_size,
        args=args
    )
    
    return detector
