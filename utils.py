
import os
import torch
import pickle
import numpy as np
import scipy.io as sio
import time

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, ConcatDataset

from vcoco.vcoco import VCOCO
from hicodet.hicodet import HICODet
from vg.vg import VG

import pocket.pocket as pocket
from pocket.pocket.core import DistributedLearningEngine
from pocket.pocket.utils import DetectionAPMeter, BoxPairAssociation
import clip



import sys
sys.path.append('detr')
# import detr.datasets.transforms as T
import detr.datasets.transforms_g as T
from detr.util import box_ops
import itertools
import detr.util.misc as utils

from hicodet.static_hico import RARE_HOI_IDX, NEW_VERB_OBJ_TO_HOI, HOI_IDX_TO_ACT_IDX, HICO_INTERACTIONS
from vcoco.static_vcoco import VCOCO_INTERACTIONS


def custom_collate(batch):
    images = []
    targets = []
    for im, tar in batch:
        images.append(im)
        targets.append(tar)
    return images, targets

    
class DataFactory(Dataset):
    def __init__(self, name, partition, data_root, clip_model=None, training_type='full'):
        if name not in ['hicodet', 'vcoco', 'vg', 'swig', 'hicodet+vcoco', 'hicodet+vcoco+vg']:  # (0, 1, 2, 3, 4)
            raise ValueError("Unknown dataset ", name)

        if name == 'hicodet':
            assert partition in ['train2015', 'test2015'], \
                "Unknown HICO-DET partition " + partition
            self.dataset = HICODet(
                root=os.path.join(data_root, 'hico_20160224_det/images', partition),
                # anno_file=os.path.join(data_root, 'instances_{}.json'.format(partition)),
                anno_file=os.path.join(data_root, 'instances_{}_gdino.json'.format(partition)),
                target_transform=pocket.ops.ToTensor(input_format='dict'), training=partition=='train2015', training_type=training_type
            )
        elif name == 'vcoco':
            assert partition in ['train', 'val', 'trainval', 'test'], \
                "Unknown V-COCO partition " + partition
            image_dir = dict(
                train='mscoco2014/train2014',
                val='mscoco2014/train2014',
                trainval='mscoco2014/train2014',
                test='mscoco2014/val2014'
            )
            self.dataset = VCOCO(
                root=os.path.join(data_root, image_dir[partition]),
                anno_file=os.path.join(data_root, 'instances_vcoco_{}.json'.format(partition)
                ), target_transform=pocket.ops.ToTensor(input_format='dict')
            )
        elif name == 'vg':
            self.dataset = VG(
                root=os.path.join('./vg', 'vg/VG_100K'),
                anno_file=os.path.join('./vg', 'vg/vg_hicodet_anno_xyxy_v1.json'),
                keep_names_file=os.path.join('./vg', 'vg/meta.json'),
                target_transform=pocket.ops.ToTensor(input_format='dict')
            )
        elif name == 'swig':
            assert partition in ['train', 'test', 'dev'], \
                "Unknown Swig partition " + partition
            self.dataset = Swig(
                root=os.path.join(data_root, 'swig_hoi/images_512'),
                anno_file=os.path.join(data_root, 'swig_hoi/swig_{}_1000.json'.format(partition)),
                # anno_file=os.path.join(data_root, 'swig_hoi/swig_train_1000.json'),
                target_transform=pocket.ops.ToTensor(input_format='dict'),
                training=partition=='train',
            )
        elif name == 'hicodet+vcoco':
            hicodet = HICODet(
                root=os.path.join('./hicodet', 'hico_20160224_det/images', partition),
                # anno_file=os.path.join(data_root, 'instances_{}.json'.format(partition)),
                anno_file=os.path.join('./hicodet', 'instances_{}_gdino.json'.format(partition)),
                target_transform=pocket.ops.ToTensor(input_format='dict'), training=partition=='train2015', training_type=training_type
            )
            vcoco = VCOCO(
                root=os.path.join('./vcoco', 'mscoco2014/train2014'),
                anno_file=os.path.join('./vcoco', 'instances_vcoco_{}.json'.format('trainval')
                ), target_transform=pocket.ops.ToTensor(input_format='dict')
            )
            if partition == 'train2015':
                self.dataset = ConcatDataset([hicodet, vcoco])
            else:
                self.dataset = hicodet
        elif name == 'hicodet+vcoco+vg':
            hicodet = HICODet(
                root=os.path.join('./hicodet', 'hico_20160224_det/images', partition),
                # anno_file=os.path.join(data_root, 'instances_{}.json'.format(partition)),
                anno_file=os.path.join('./hicodet', 'instances_{}_gdino.json'.format(partition)),
                target_transform=pocket.ops.ToTensor(input_format='dict'), training=partition=='train2015', training_type=training_type
            )
            vcoco = VCOCO(
                root=os.path.join('./vcoco', 'mscoco2014/train2014'),
                anno_file=os.path.join('./vcoco', 'instances_vcoco_{}.json'.format('trainval')
                ), target_transform=pocket.ops.ToTensor(input_format='dict')
            )
            vg = VG(
                root=os.path.join('./vg', 'vg/VG_100K'),
                anno_file=os.path.join('./vg', 'vg/vg_hicodet_anno_xyxy_v1.json'),
                keep_names_file=os.path.join('./vg', 'vg/meta.json'),
                target_transform=pocket.ops.ToTensor(input_format='dict')
            )
            if partition == 'train2015':
                self.dataset = ConcatDataset([hicodet, vcoco, vg])
            else:
                self.dataset = hicodet

                # Prepare dataset transforms
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        if partition.startswith('train'):
            self.transforms = [T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(.4, .4, .4),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ])
                )
        ]),normalize]
        else:
            # self.transforms = T.Compose([
            #     T.RandomResize([800], max_size=1333),
            #     normalize,
            # ])
            self.transforms = [T.Compose([
                T.RandomResize([800], max_size=1333),
                # normalize,
            ]), normalize]

        self.name = name

        self.training = partition.startswith('train')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _, self.clip_preprocess = clip.load(clip_model, device)
        del _

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, target = self.dataset[i]
        if self.name == 'hicodet':
            target['dataset'] = torch.tensor(0)
            # target['labels'] = target['verb']
            target['labels'] = target['hoi']
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            target['boxes_h'][:, :2] -= 1
            target['boxes_o'][:, :2] -= 1
            if 'boxes_g' in target.keys() and len(target['boxes_g']) > 0:
                target['boxes_g'][:, :2] -= 1
        elif self.name == 'vcoco':
            target['dataset'] = torch.tensor(1)
            target['labels'] = target['actions']
            target['object'] = target.pop('objects') - 1
        elif self.name == 'vg':
            target['dataset'] = torch.tensor(2)
            target['labels'] = target['verbs']
        elif self.name == 'swig':
            target['dataset'] = torch.tensor(3)
            # target['labels'] = target['verb']
            target['labels'] = target['hoi']
            if 'boxes_g' in target.keys() and len(target['boxes_g']) == 0:
                target['boxes_g'] = torch.tensor([[0.,0.,0.,0.]])
                target['labels_g'] = torch.tensor([[0]])
                target['scores_g'] = torch.tensor([[0.1]])
        
        elif self.name == 'hicodet+vcoco':
            if 'verb' in target.keys():
                target['dataset'] = torch.tensor(0)
                target['labels'] = target['hoi']
                # Convert ground truth boxes to zero-based index and the
                # representation from pixel indices to coordinates
                target['boxes_h'][:, :2] -= 1
                target['boxes_o'][:, :2] -= 1
                if 'boxes_g' in target.keys() and len(target['boxes_g']) > 0:
                    target['boxes_g'][:, :2] -= 1
            elif 'actions' in target.keys():
                target['dataset'] = torch.tensor(1)
                target['labels'] = target['hoi'] + 600
                target['verb'] = target['actions'] + 117
                target['object'] = target.pop('objects') - 1
            else:
                raise

        elif self.name == 'hicodet+vcoco+vg':
            if 'verb' in target.keys():
                target['dataset'] = torch.tensor(0)
                target['labels'] = target['hoi']
                # Convert ground truth boxes to zero-based index and the
                # representation from pixel indices to coordinates
                target['boxes_h'][:, :2] -= 1
                target['boxes_o'][:, :2] -= 1
                if 'boxes_g' in target.keys() and len(target['boxes_g']) > 0:
                    target['boxes_g'][:, :2] -= 1
            elif 'actions' in target.keys():
                target['dataset'] = torch.tensor(1)
                target['labels'] = target['hoi'] + len(HICO_INTERACTIONS)
                target['verb'] = target['actions'] + 117
                target['object'] = target['objects'] - 1
            elif 'verbs' in target.keys():
                target['dataset'] = torch.tensor(2)
                target['labels'] = target['hoi'] + len(HICO_INTERACTIONS) + len(VCOCO_INTERACTIONS)
                target['verb'] = target['verbs'] + 117 + 24
                target['object'] = target['objects']
                target['object'][target['object']>0] += 80
                target['labels_g'][target['labels_g']>0] += 80
                # if 'boxes_g' in target.keys() and len(target['boxes_g']) == 0:
                #     target['boxes_g'] = torch.tensor([[0.,0.,0.,0.]])
                #     target['labels_g'] = torch.tensor([[0]])
                #     target['scores_g'] = torch.tensor([[0.1]])
            else:
                raise


        img_0, target_0 = self.transforms[0](image, target)
        image, target = self.transforms[1](img_0, target_0)
        clip_input = self.clip_preprocess(img_0)
        target['clip_input'] = clip_input

        return image, target
        

class CacheTemplate(defaultdict):
    """A template for VCOCO cached results """
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self[k] = v
    def __missing__(self, k):
        seg = k.split('_')
        # Assign zero score to missing actions
        if seg[-1] == 'agent':
            return 0.
        # Assign zero score and a tiny box to missing <action,role> pairs
        else:
            return [0., 0., .1, .1, 0.]

class CustomisedDLE(DistributedLearningEngine):
    def __init__(self, net, dataloader, max_norm=0, num_classes=117, start_epoch=0, **kwargs):
        super().__init__(net, None, dataloader, **kwargs)
        self.max_norm = max_norm
        self.num_classes = num_classes
        self._state.epoch = start_epoch
        self._state.iteration = len(self._train_loader) * self._state.epoch
    
    def __call__(self, start: int, n: int) -> None:
        self.epochs = n
        # Train for a specified number of epochs
        self._on_start()
        for _ in range(start, n):
            self._on_start_epoch()
            timestamp = time.time()
            for batch in self._train_loader:
                self._state.inputs = batch[:-1]
                self._state.targets = batch[-1]
                self._on_start_iteration()
                self._state.t_data.append(time.time() - timestamp)

                self._on_each_iteration()
                self._state.running_loss.append(self._state.loss.item())
                self._on_end_iteration()
                self._state.t_iteration.append(time.time() - timestamp)
                timestamp = time.time()
                
            self._on_end_epoch()
        self._on_end()

    def _on_each_iteration(self):
        loss_dict = self._state.net(
            *self._state.inputs, targets=self._state.targets)
        # when the interaction-cls input is 0
        # if loss_dict['interaction_loss'] == 0:return
        if loss_dict['interaction_loss'].isnan():
            raise ValueError(f"The HOI loss is NaN for rank {self._rank}")

        self._state.loss = sum(loss for loss in loss_dict.values())
        self._state.optimizer.zero_grad(set_to_none=True)
        self._state.loss.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
        self._state.optimizer.step()
    
    def _on_end_epoch(self):
        # Save checkpoint in the master process
        if self._rank == 0:
            self.save_checkpoint()
        if self._state.lr_scheduler is not None:
            self._state.lr_scheduler.step()

    @torch.no_grad()
    def test_hico(self, dataloader):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        associate = BoxPairAssociation(min_iou=0.5)
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        hoi2act = torch.from_numpy(np.asarray(
            HOI_IDX_TO_ACT_IDX, dtype=int
        ))

        # conversion = torch.from_numpy(np.asarray(
        #     NEW_VERB_OBJ_TO_HOI, dtype=float
        # ))

        meter = DetectionAPMeter(
            600, nproc=1,
            num_gt=dataset.anno_interaction,
            algorithm='11P'
        )
        for batch in tqdm(dataloader):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            # output = net(inputs)
            output = net(inputs, batch[-1])

            # Skip images without detections
            if output is None or len(output) == 0:
                # print('output is none')
                continue
            
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            target = batch[-1][0]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            verb_scores = output['verb_scores']

            # interactions = conversion[objects, verbs]
            inter_v = verb_scores[torch.arange(len(verb_scores)), hoi2act[verbs]]
            scores = scores * inter_v
            # print(scores.sort()[0])
            # print(inter_v.shape, verbs.shape)
            # print(verbs[scores.sort()[1]])
            interactions = verbs
            # Recover target box scale
            gt_bx_h = net.module.recover_boxes(target['boxes_h'], target['size'])
            gt_bx_o = net.module.recover_boxes(target['boxes_o'], target['size'])
            # print(scores, interactions)
            # Associate detected pairs with ground truth pairs
            labels = torch.zeros_like(scores)
            unique_hoi = interactions.unique()
            for hoi_idx in unique_hoi:
                gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
                det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                if len(gt_idx):
                    labels[det_idx] = associate(
                        (gt_bx_h[gt_idx].view(-1, 4),
                        gt_bx_o[gt_idx].view(-1, 4)),
                        (boxes_h[det_idx].view(-1, 4),
                        boxes_o[det_idx].view(-1, 4)),
                        scores[det_idx].view(-1)
                    )

            meter.append(scores, interactions, labels)
            # break
            
        return meter.eval()
    
    @torch.no_grad()
    def test_swig(self, dataloader):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        associate = BoxPairAssociation(min_iou=0.5)
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))

        hoi2act = torch.from_numpy(np.asarray(
            dataset.interaction_to_verb, dtype=int
        ))
        # conversion = torch.from_numpy(np.asarray(
        #     NEW_VERB_OBJ_TO_HOI, dtype=float
        # ))

        meter = DetectionAPMeter(
            dataset.num_interation_cls, nproc=1,
            num_gt=dataset.anno_interaction,
            algorithm='11P'
        )
        for batch in tqdm(dataloader):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            # output = net(inputs)
            output = net(inputs, batch[-1])

            # Skip images without detections
            if output is None or len(output) == 0:
                print('output is none')
                continue
            
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            target = batch[-1][0]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            verb_scores = output['verb_scores']
            # interactions = conversion[objects, verbs]
            inter_v = verb_scores[torch.arange(len(verb_scores)), hoi2act[verbs]]
            scores = scores * inter_v 
            
            interactions = verbs
            # Recover target box scale
            gt_bx_h = net.module.recover_boxes(target['boxes_h'], target['size'])
            gt_bx_o = net.module.recover_boxes(target['boxes_o'], target['size'])
            # print(scores, interactions)
            # Associate detected pairs with ground truth pairs
            labels = torch.zeros_like(scores)
            unique_hoi = interactions.unique()
            for hoi_idx in unique_hoi:
                gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
                det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                if len(gt_idx):
                    labels[det_idx] = associate(
                        (gt_bx_h[gt_idx].view(-1, 4),
                        gt_bx_o[gt_idx].view(-1, 4)),
                        (boxes_h[det_idx].view(-1, 4),
                        boxes_o[det_idx].view(-1, 4)),
                        scores[det_idx].view(-1)
                    )

            meter.append(scores, interactions, labels)
            # break
            
        return meter.eval()

    @torch.no_grad()
    def test_vg(self, dataloader):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        associate = BoxPairAssociation(min_iou=0.5)
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        hoi2act = torch.from_numpy(np.asarray(
            dataset.interaction_to_verb, dtype=int
        ))
        # conversion = torch.from_numpy(np.asarray(
        #     NEW_VERB_OBJ_TO_HOI, dtype=float
        # ))

        meter = DetectionAPMeter(
            dataset.num_interation_cls, nproc=1,
            num_gt=dataset.anno_interaction,
            algorithm='11P'
        )
        for batch in tqdm(dataloader):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            # output = net(inputs)
            output = net(inputs, batch[-1])

            # Skip images without detections
            if output is None or len(output) == 0:
                print('output is none')
                continue
            
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            target = batch[-1][0]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            verb_scores = output['verb_scores']
            # interactions = conversion[objects, verbs]
            inter_v = verb_scores[torch.arange(len(verb_scores)), hoi2act[verbs]]
            scores = scores * inter_v
            # print(scores.sort())
            interactions = verbs
            # Recover target box scale
            gt_bx_h = net.module.recover_boxes(target['boxes_h'], target['size'])
            gt_bx_o = net.module.recover_boxes(target['boxes_o'], target['size'])
            # print(scores, interactions)
            # Associate detected pairs with ground truth pairs
            labels = torch.zeros_like(scores)
            unique_hoi = interactions.unique()
            for hoi_idx in unique_hoi:
                gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
                det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                if len(gt_idx):
                    labels[det_idx] = associate(
                        (gt_bx_h[gt_idx].view(-1, 4),
                        gt_bx_o[gt_idx].view(-1, 4)),
                        (boxes_h[det_idx].view(-1, 4),
                        boxes_o[det_idx].view(-1, 4)),
                        scores[det_idx].view(-1)
                    )

            meter.append(scores, interactions, labels)
            # break
            
        return meter.eval()

    @torch.no_grad()
    def cache_hico(self, dataloader, cache_dir='matlab'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        object2int = dataset.object_to_interaction

        # Include empty images when counting
        nimages = len(dataset.annotations)
        all_results = np.empty((600, nimages), dtype=object)

        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_idx = dataset._idx[i]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            interactions = conversion[objects, verbs]
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            # Convert box representation to pixel indices
            boxes_h[:, 2:] -= 1
            boxes_o[:, 2:] -= 1

            # Group box pairs with the same predicted class
            permutation = interactions.argsort()
            boxes_h = boxes_h[permutation]
            boxes_o = boxes_o[permutation]
            interactions = interactions[permutation]
            scores = scores[permutation]

            # Store results
            unique_class, counts = interactions.unique(return_counts=True)
            n = 0
            for cls_id, cls_num in zip(unique_class, counts):
                all_results[cls_id.long(), image_idx] = torch.cat([
                    boxes_h[n: n + cls_num],
                    boxes_o[n: n + cls_num],
                    scores[n: n + cls_num, None]
                ], dim=1).numpy()
                n += cls_num
        
        # Replace None with size (0,0) arrays
        for i in range(600):
            for j in range(nimages):
                if all_results[i, j] is None:
                    all_results[i, j] = np.zeros((0, 0))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # Cache results
        for object_idx in range(80):
            interaction_idx = object2int[object_idx]
            sio.savemat(
                os.path.join(cache_dir, f'detections_{(object_idx + 1):02d}.mat'),
                dict(all_boxes=all_results[interaction_idx])
            )

    @torch.no_grad()
    def cache_vcoco(self, dataloader, cache_dir='vcoco_cache'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        all_results = []
        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_id = dataset.image_id(i)
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            scores = output['scores']
            actions = output['labels']
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            for bh, bo, s, a in zip(boxes_h, boxes_o, scores, actions):
                a_name = dataset.actions[a].split()
                result = CacheTemplate(image_id=image_id, person_box=bh.tolist())
                result[a_name[0] + '_agent'] = s.item()
                result['_'.join(a_name)] = bo.tolist() + [s.item()]
                all_results.append(result)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(os.path.join(cache_dir, 'cache.pkl'), 'wb') as f:
            # Use protocol 2 for compatibility with Python2
            pickle.dump(all_results, f, 2)
    
    @torch.no_grad()
    def cache_hico_binary(self, dataloader):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        all_results = {'keys':[], "bboxes":[], "scores":[], 'size':[]}
        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            target = batch[-1][0]
            
            # Format detections
            image_idx = dataset._idx[i]
            key = int(dataset._filenames[image_idx].split('.')[0].split('_')[-1])
            boxes = output['boxes']
            # boxes = recover_boxes(boxes, target['size'])
            boxes_h, boxes_o = boxes[output['bi_pairing']].unbind(0)
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            objects = output['objects']
            bi_scores = output['bi_scores']
            verb = output['labels']
            keys = [key] * len(bi_scores)
            sizes = [target['size'].cpu().numpy()] * len(bi_scores)
            # results = []
            all_results['keys'].append(keys) 
            all_results["bboxes"].append(np.concatenate((boxes_h, boxes_o), axis=1)) 
            all_results["scores"].append(bi_scores)
            all_results['size'].append(sizes)
            
            

        # with open(os.path.join(self._cache_dir, 'binary_cache.pkl'), 'wb') as f:
        #     # Use protocol 2 for compatibility with Python2
        #     pickle.dump(all_results, f, 2)
        all_results['keys'] = np.concatenate(all_results['keys'])
        all_results['bboxes'] = np.concatenate(all_results['bboxes'])
        all_results['scores'] = np.concatenate(all_results['scores'])
        all_results['size'] = np.concatenate(all_results['size'])
        return all_results
    
    @torch.no_grad()
    def test_hico_knn(self, dataloader, trainset):
        net = self._state.net
        net.eval()

        num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
        rare = torch.nonzero(num_anno < 10).squeeze(1)
        non_rare = torch.nonzero(num_anno >= 10).squeeze(1)

        dataset = dataloader.dataset.dataset
        associate = BoxPairAssociation(min_iou=0.5)
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))

        meter = DetectionAPMeter(
            600, nproc=1,
            num_gt=dataset.anno_interaction,
            algorithm='11P'
        )
        for batch in tqdm(dataloader):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            target = batch[-1][0]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            interactions = conversion[objects, verbs]
            # Recover target box scale
            gt_bx_h = net.module.recover_boxes(target['boxes_h'], target['size'])
            gt_bx_o = net.module.recover_boxes(target['boxes_o'], target['size'])


            text_scores = output['text_scores']
            knn_scores = output['knn_scores']
            bii_scores = output['bii_scores']

            # scores = scores * text_scores[torch.arange(scores.shape[0]), interactions.long()]
            # print(knn_scores.topk(5, dim=-1))
            weight = 0.1
            knn_scores = knn_scores ** weight * text_scores.unsqueeze(-1).repeat(1, 1, 10).reshape(knn_scores.shape) ** (1-weight)
            knn_scores = knn_scores.reshape(knn_scores.shape[0], 600, 10)
            top_k_scores, top_k_indices = knn_scores.topk(k=5, dim=-1, largest=True, sorted=True)
            top_k_scores = top_k_scores.mean(-1)

            top_k_scores = text_scores
            
            weight = 0.3
            rare_idx = [True if inter in rare else False for inter in interactions]
            non_rare_idx = [False if inter in rare else True for inter in interactions]
            scores[rare_idx] = scores[rare_idx] * top_k_scores[torch.arange(scores.shape[0]), interactions.long()][rare_idx]
            scores[non_rare_idx] = scores[non_rare_idx] ** (1-weight) * top_k_scores[torch.arange(scores.shape[0]), interactions.long()][non_rare_idx] ** weight
            

            # Associate detected pairs with ground truth pairs
            labels = torch.zeros_like(scores)
            unique_hoi = interactions.unique()
            for hoi_idx in unique_hoi:
                gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
                det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                if len(gt_idx):
                    labels[det_idx] = associate(
                        (gt_bx_h[gt_idx].view(-1, 4),
                        gt_bx_o[gt_idx].view(-1, 4)),
                        (boxes_h[det_idx].view(-1, 4),
                        boxes_o[det_idx].view(-1, 4)),
                        scores[det_idx].view(-1)
                    )

            meter.append(scores, interactions, labels)
            
        return meter.eval()
        