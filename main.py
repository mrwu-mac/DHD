

import os
import sys
import torch
import random
import warnings
import argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
import json

from models import build_detector
from utils import custom_collate, CustomisedDLE, DataFactory

from hicodet.static_hico import HOI_IDX_TO_ACT_IDX, UA_HOI_IDX, UO_HOI_IDX, NEW_HOI, OBJ_IDX_TO_OBJ_NAME, UV_HOI_IDX, UC_HOI_IDX, \
                                HICO_INTERACTIONS, ACT_TO_ING
from vcoco.static_vcoco import vcoco_hoi_text_label

warnings.filterwarnings("ignore")



def main(rank, args):

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    # Fix seed
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.set_device(rank)

    trainset = DataFactory(name=args.dataset, partition=args.partitions[0], data_root=args.data_root, clip_model=args.clip_model, training_type=args.training_type)
    testset = DataFactory(name=args.dataset, partition=args.partitions[1], data_root=args.data_root, clip_model=args.clip_model)

    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        sampler=DistributedSampler(
            trainset, 
            num_replicas=args.world_size, 
            rank=rank)
    )
    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=1,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        sampler=torch.utils.data.SequentialSampler(testset)
    )

    args.human_idx = 0
    if args.dataset == 'hicodet':
        # object_to_target = train_loader.dataset.dataset.object_to_verb
        # args.num_classes = 117
        object_to_target = train_loader.dataset.dataset.object_to_interaction
        args.num_classes = train_loader.dataset.dataset.num_interation_cls

    elif args.dataset == 'vcoco':
        # object_to_target = list(train_loader.dataset.dataset.object_to_action.values())
        # args.num_classes = 24
        object_to_target = train_loader.dataset.dataset.object_to_interaction
        args.num_classes = len(train_loader.dataset.dataset.interactions)
    elif args.dataset == 'vg':
        object_to_target = train_loader.dataset.dataset.object_to_interaction
        args.num_classes = train_loader.dataset.dataset.num_interation_cls
    elif args.dataset == 'hicodet+vcoco':
        hico_object_to_target = train_loader.dataset.dataset.datasets[0].object_to_interaction
        vcoco_object_to_target = train_loader.dataset.dataset.datasets[1].object_to_interaction
        object_to_target = []
        for h, vcoco in zip(hico_object_to_target, vcoco_object_to_target):
            h = h + ([__ + train_loader.dataset.dataset.datasets[0].num_interation_cls for __ in vcoco])
            object_to_target.append(h)
        args.num_classes = train_loader.dataset.dataset.datasets[0].num_interation_cls + len(train_loader.dataset.dataset.datasets[1].interactions)
    elif args.dataset == 'hicodet+vcoco+vg':
       
        hico_object_to_target = train_loader.dataset.dataset.datasets[0].object_to_interaction
        vcoco_object_to_target = train_loader.dataset.dataset.datasets[1].object_to_interaction
        vg_object_to_target = train_loader.dataset.dataset.datasets[2].object_to_interaction
        object_to_target = []
        ## coco 80 to hico & vcoco
        for h, vcoco in zip(hico_object_to_target, vcoco_object_to_target):
            h = h + [__ + train_loader.dataset.dataset.datasets[0].num_interation_cls for __ in vcoco]
            object_to_target.append(h)
        ## 80 + vg_objs to vg
        h = [__ + train_loader.dataset.dataset.datasets[0].num_interation_cls + len(train_loader.dataset.dataset.datasets[1].interactions) for __ in vg_object_to_target[0]]
        object_to_target[0]=h
        assert len(object_to_target) == 80
        for vg in vg_object_to_target:
            h = [__ + train_loader.dataset.dataset.datasets[0].num_interation_cls + len(train_loader.dataset.dataset.datasets[1].interactions) for __ in vg]
            object_to_target.append(h)
        args.num_classes = train_loader.dataset.dataset.datasets[0].num_interation_cls + len(train_loader.dataset.dataset.datasets[1].interactions) + train_loader.dataset.dataset.datasets[2].num_interation_cls
    
    dhd = build_detector(args, object_to_target)

    if os.path.exists(args.resume):
        print(f"=> Rank {rank}: continue from saved checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        dhd.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if 'epoch' in checkpoint and 'iteration' in checkpoint:
            start_epoch = checkpoint['epoch']
            start_iter = checkpoint['iteration']
    else:
        print(f"=> Rank {rank}: start from a randomly initialised model")
        start_epoch = 0
        start_iter = 0
        # load from pretrained
        # if 'pretrain' in args.pretrained:
        #     print(f"=> Rank {rank}: continue from saved checkpoint {args.pretrained}")
        #     checkpoint = torch.load(args.pretrained, map_location='cpu')
        #     # for k in list(checkpoint['model_state_dict'].keys()):
        #     #     if 'predictor' in k:
        #     #         checkpoint['model_state_dict'].pop(k)
        #     dhd.load_state_dict(checkpoint['model_state_dict'], strict=False)

    engine = CustomisedDLE(
        dhd, train_loader,
        max_norm=args.clip_max_norm,
        num_classes=args.num_classes,
        print_interval=args.print_interval,
        find_unused_parameters=True,
        cache_dir=args.output_dir,
        start_epoch=start_epoch
    )

    if args.cache:
        if args.dataset == 'hicodet':
            engine.cache_hico(test_loader, args.output_dir)
        elif args.dataset == 'vcoco':
            engine.cache_vcoco(test_loader, args.output_dir)
        return

    if args.eval or args.eval_knn:
        if args.dataset == 'vcoco':
            raise NotImplementedError(f"Evaluation on V-COCO has not been implemented.")
        
        if args.dataset == 'hicodet':
            if args.eval_knn:
                ap = engine.test_hico_knn(test_loader, trainset)
            else:
                ap = engine.test_hico(test_loader)
            
            # Fetch indices for rare and non-rare classes
            num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
            rare = torch.nonzero(num_anno < 10).squeeze(1)
            non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
            print(
                f"The mAP is {ap.mean():.4f},"
                f" rare: {ap[rare].mean():.4f},"
                f" none-rare: {ap[non_rare].mean():.4f}"
            )
            if args.training_type != 'full':
                if args.training_type == 'ua':
                    unseen_idx = UA_HOI_IDX
                elif args.training_type == 'uv':
                    unseen_idx = UV_HOI_IDX
                elif args.training_type == 'uo':
                    unseen_idx = UO_HOI_IDX
                elif args.training_type == 'rfuc':
                    unseen_idx = UC_HOI_IDX['rare_first']
                elif args.training_type == 'nfuc':
                    unseen_idx = UC_HOI_IDX['non_rare_first']
                else:
                    print('training type is needed')
                    raise 
                unseen = torch.as_tensor(unseen_idx)
                seen = torch.as_tensor([ix for ix in range(600) if ix not in unseen_idx])
                print(
                    f" seen: {ap[seen].mean():.4f},"
                    f" unseen: {ap[unseen].mean():.4f}"
                )
            
            hoi2act = torch.as_tensor(HOI_IDX_TO_ACT_IDX)
            keep = torch.nonzero(hoi2act != 57).squeeze(1)
            rare_k = torch.nonzero(torch.logical_and(num_anno < 10, hoi2act != 57)).squeeze(1)
            non_rare_k = torch.nonzero(torch.logical_and(num_anno >= 10, hoi2act != 57)).squeeze(1)
            print(
                f"filter no_interaction:\n"
                f"The mAP is {ap[keep].mean():.4f},"
                f" rare: {ap[rare_k].mean():.4f},"
                f" none-rare: {ap[non_rare_k].mean():.4f}"
            )
        if args.dataset == 'vg':
            ap = engine.test_vg(test_loader)

            eval_hois = np.array([i for i in range(train_loader.dataset.dataset.num_interation_cls)])
            hico_ao_pair = [[ACT_TO_ING[d['action']], d['object'].replace('_', ' ')] for d in HICO_INTERACTIONS]
            
            hico_hois = np.array([i for i, vg_hoi in enumerate(train_loader.dataset.dataset.interaction_names) if vg_hoi in hico_ao_pair])
            print(len(hico_hois))
            
            # eval_hois = np.array(trainset.dataset.meta['interactions_for_eval'])
            # rare_hois = np.array(trainset.dataset.meta['rare_interaction_ids'])
            # novel_hois = np.array(trainset.dataset.meta['novel_interaction_ids'])
            # common_hois = np.array(trainset.dataset.meta['common_interaction_ids'])
            # hico_hois = 
            
            # eval_common_hois = np.intersect1d(eval_hois, common_hois)
            # eval_rare_hois = np.intersect1d(eval_hois, rare_hois)
            # eval_novel_hois = np.intersect1d(eval_hois, novel_hois)
            known_hois = hico_hois
            novel_hois = np.setdiff1d(eval_hois, hico_hois)
            
            full_map = ap.mean()
            print('pos_ap_num:', (ap>0).sum())
            fz_map = ap[ap>0].mean()
            # rare_map = ap[eval_rare_hois].mean()
            # non_rare_map = ap[eval_common_hois].mean()
            known_map = ap[known_hois].mean()
            novel_map = ap[novel_hois].mean()

            print(
                f"full: {full_map:.4f},"
                # f" rare: {rare_map:.4f},"
                f" known: {known_map:.4f},"
                f" novel: {novel_map:.4f},"
                f" filter zero ap: {fz_map:.4f},"
            )

        if args.dataset == 'swig':
            ap = engine.test_swig(test_loader)

            eval_hois = np.array(trainset.dataset.meta['interactions_for_eval'])
            rare_hois = np.array(trainset.dataset.meta['rare_interaction_ids'])
            novel_hois = np.array(trainset.dataset.meta['novel_interaction_ids'])
            common_hois = np.array(trainset.dataset.meta['common_interaction_ids'])
            
            eval_common_hois = np.intersect1d(eval_hois, common_hois)
            eval_rare_hois = np.intersect1d(eval_hois, rare_hois)
            eval_novel_hois = np.intersect1d(eval_hois, novel_hois)
            eval_known_hois = np.setdiff1d(eval_hois, novel_hois)
            
            # print(eval_novel_hois, ap[eval_novel_hois])
            full_map = ap[eval_known_hois].mean()
            rare_map = ap[eval_rare_hois].mean()
            non_rare_map = ap[eval_common_hois].mean()
            novel_map = ap[eval_novel_hois].mean()

            print(
                f"full: {full_map:.4f},"
                f" rare: {rare_map:.4f},"
                f" non_rare: {non_rare_map:.4f},"
                f" novel: {novel_map:.4f}"
            )
        
        return

    if args.eval_binary:
        from binary_evaluation import calc_binary
        res = engine.cache_hico_binary(test_loader)
        keys, bboxes, score, sizes = res["keys"], res["bboxes"], res["scores"], res["size"]
        bin_ap, bin_rec = calc_binary(keys, bboxes, score, sizes)
        print("interactiveness AP: ", 100 * bin_ap)
        return
    

    if not args.use_cache_box:
        for p in dhd.detector.parameters():
            p.requires_grad = False
    
    for n, p in dhd.teacher_model.named_parameters():
        p.requires_grad = False


    param_dicts = [{
            "params": [p for n, p in dhd.named_parameters()
            if "interaction_head" in n and p.requires_grad]
            }]

    
    optim = torch.optim.AdamW(
        param_dicts, lr=args.lr_head,
        weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_drop)
    # Override optimiser and learning rate scheduler
    if os.path.exists(args.resume) and not args.eval and 'optim_state_dict' in checkpoint and 'scheduler_state_dict' in checkpoint:
        optim.load_state_dict(checkpoint['optim_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)

    engine(start_epoch, args.epochs)

@torch.no_grad()
def sanity_check(args):
    dataset = DataFactory(name='hicodet', partition=args.partitions[0], data_root=args.data_root)
    args.human_idx = 0; args.num_classes = 117
    object_to_target = dataset.dataset.object_to_verb
    dhd = build_detector(args, object_to_target)
    if args.eval:
        dhd.eval()

    image, target = dataset[0]
    outputs = dhd([image], [target])

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr-head', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr-drop', default=10, type=int)
    parser.add_argument('--clip-max-norm', default=0.1, type=float)

    parser.add_argument('--dino_config', default='GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py', type=str)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--binary', action='store_true')
    parser.add_argument('--eval_binary', action='store_true')
    parser.add_argument('--cls_dec_layers', default=3, type=int)
    parser.add_argument('--binary_thres', default=0.3, type=float)
    parser.add_argument('--clip_model', default='ViT-B/32',
                        help='clip pretrained model path')
    parser.add_argument('--eval_knn', action='store_true')
    # parser.add_argument('--gdino', action='store_true')
    parser.add_argument('--eval_with_gt', action='store_true')
    parser.add_argument('--cl', action='store_true')
    parser.add_argument('--neg_pair_wieght', default=0.2, type=float)
    parser.add_argument('--training_type', default='full', type=str, choices=('ua', 'uv', 'uo', 'rfuc', 'nfuc', 'uc0', 'uc1', 'uc2', 'uc3', 'uc4'))


    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position-embedding', default='sine', type=str, choices=('sine', 'learned'))

    parser.add_argument('--repr-dim', default=512, type=int)
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--enc-layers', default=6, type=int)
    parser.add_argument('--dec-layers', default=6, type=int)
    parser.add_argument('--dim-feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num-queries', default=100, type=int)
    parser.add_argument('--pre-norm', action='store_true')

    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set-cost-class', default=1, type=float)
    parser.add_argument('--set-cost-bbox', default=5, type=float)
    parser.add_argument('--set-cost-giou', default=2, type=float)
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument('--eos-coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.2, type=float)

    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--data-root', default='./hicodet')

    # training parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--port', default='1234', type=str)
    parser.add_argument('--seed', default=66, type=int)
    parser.add_argument('--pretrained', default='', help='Path to a pretrained detector')
    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--print-interval', default=100, type=int)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--fg-iou-thresh', default=0.5, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)


    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='/comp_robot/cv_public_dataset/COCO2017/')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')
    parser.add_argument('--use_cache_box', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")


    args = parser.parse_args()

    print(args)

    if args.sanity:
        sanity_check(args)
        sys.exit()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port

    mp.spawn(main, nprocs=args.world_size, args=(args,))
