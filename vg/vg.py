import os
import json
import numpy as np

from typing import Optional, List, Callable, Tuple
from pocket.pocket.data import ImageDataset, DataSubset


class VGSubset(DataSubset):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def filename(self, idx: int) -> str:
        """Override: return the image file name in the subset"""
        return self._filenames[self._idx[self.pool[idx]]]

    def image_size(self, idx: int) -> Tuple[int, int]:
        """Override: return the size (width, height) of an image in the subset"""
        return self._image_sizes[self._idx[self.pool[idx]]]

    @property
    def anno_interaction(self) -> List[int]:
        """Override: Number of annotated box pairs for each interaction class"""
        num_anno = [0 for _ in range(self.num_interation_cls)]
        intra_idx = [self._idx[i] for i in self.pool]
        for idx in intra_idx:
            for hoi in self._anno[idx]['hoi']:
                num_anno[hoi] += 1
        return num_anno

    @property
    def anno_object(self) -> List[int]:
        """Override: Number of annotated box pairs for each object class"""
        num_anno = [0 for _ in range(self.num_object_cls)]
        anno_interaction = self.anno_interaction
        for corr in self._class_corr:
            num_anno[corr[1]] += anno_interaction[corr[0]]
        return num_anno

    @property
    def anno_action(self) -> List[int]:
        """Override: Number of annotated box pairs for each action class"""
        num_anno = [0 for _ in range(self.num_action_cls)]
        anno_interaction = self.anno_interaction
        for corr in self._class_corr:
            num_anno[corr[2]] += anno_interaction[corr[0]]
        return num_anno


class VG(ImageDataset):
    """
    Arguments:
        root(str): Root directory where images are downloaded to
        anno_file(str): Path to json annotation file
        transform(callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version
        target_transform(callable, optional): A function/transform that takes in the
            target and transforms it
        transforms (callable, optional): A function/transform that takes input sample
            and its target as entry and returns a transformed version.
    """

    def __init__(self, root: str, anno_file: str, keep_names_file: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None) -> None:
        super(VG, self).__init__(root, transform, target_transform, transforms)
        with open(anno_file, 'r') as f:
            anno = json.load(f)
        with open(keep_names_file, "r") as f:
            vg_keep_names = json.load(f)

        self.object_names = vg_keep_names["objects"]
        self.verb_names = vg_keep_names['verbs']
        self.interaction_names = vg_keep_names["interactions"]
        self.num_object_cls = len(self.object_names)
        self.num_interation_cls = len(self.interaction_names)
        self.num_action_cls = len(self.verb_names)
        self._anno_file = anno_file
        # self.coco_names = [
        #     'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        #     'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
        #     'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        #     'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
        #     'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite',
        #     'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard',
        #     'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon',
        #     'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        #     'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted_plant',
        #     'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        #     'keyboard', 'cell_phone', 'microwave', 'oven', 'toaster', 'sink',
        #     'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy_bear',
        #     'hair_drier', 'toothbrush'
        # ]
        with open('vg/vg/gdino_vg_anno_xyxy.json', 'r') as f:
            gdino_boxes = json.load(f)

        # Load annotations
        self._load_annotation_and_metadata(anno, gdino_boxes)

    def __len__(self) -> int:
        """Return the number of images"""
        return len(self._idx)

    def __getitem__(self, i: int) -> tuple:
        """
        Arguments:
            i(int): Index to an image

        Returns:
            tuple[image, target]: By default, the tuple consists of a PIL image and a
                dict with the following keys:
                    "boxes_h": list[list[4]]
                    "boxes_o": list[list[4]]
                    "hoi":: list[N]
                    "verb": list[N]
                    "object": list[N]
        """
        intra_idx = self._idx[i]
        return self._transforms(
            self.load_image(os.path.join(self._root, self._filenames[intra_idx])),
            self._anno[intra_idx]
        )

    def __repr__(self) -> str:
        """Return the executable string representation"""
        reprstr = self.__class__.__name__ + '(root=' + repr(self._root)
        reprstr += ', anno_file='
        reprstr += repr(self._anno_file)
        reprstr += ')'
        # Ignore the optional arguments
        return reprstr

    def __str__(self) -> str:
        """Return the readable string representation"""
        reprstr = 'Dataset: ' + self.__class__.__name__ + '\n'
        reprstr += '\tNumber of images: {}\n'.format(self.__len__())
        reprstr += '\tImage directory: {}\n'.format(self._root)
        reprstr += '\tAnnotation file: {}\n'.format(self._root)
        return reprstr

    @property
    def annotations(self) -> List[dict]:
        return self._anno

    @property
    def class_corr(self) -> List[Tuple[int, int, int]]:
        """
        Class correspondence matrix in zero-based index
        [
            [hoi_idx, obj_idx, verb_idx],
            ...
        ]

        Returns:
            list[list[3]]
        """
        return self._class_corr.copy()

    @property
    def object_n_verb_to_interaction(self) -> List[list]:
        """
        The interaction classes corresponding to an object-verb pair

        HICODet.object_n_verb_to_interaction[obj_idx][verb_idx] gives interaction class
        index if the pair is valid, None otherwise

        Returns:
            list[list[117]]
        """
        lut = np.full([self.num_object_cls, self.num_action_cls], None)
        for i, j, k in self._class_corr:
            lut[j, k] = i
        return lut.tolist()
    
    @property
    def interaction_to_verb(self) -> List[list]:
        """
        The interaction classes corresponding to an object-verb pair

        HICODet.object_n_verb_to_interaction[obj_idx][verb_idx] gives interaction class
        index if the pair is valid, None otherwise

        Returns:
            list[list[117]]
        """
        lut = np.full([self.num_interation_cls], None)
        for i, j, k in self._class_corr:
            lut[i] = k
        return lut.tolist()

    @property
    def object_to_interaction(self) -> List[list]:
        """
        The interaction classes that involve each object type

        Returns:
            list[list]
        """
        obj_to_int = [[] for _ in range(self.num_object_cls)]
        for corr in self._class_corr:
            obj_to_int[corr[1]].append(corr[0])
        return obj_to_int

    @property
    def object_to_verb(self) -> List[list]:
        """
        The valid verbs for each object type

        Returns:
            list[list]
        """
        obj_to_verb = [[] for _ in range(self.num_object_cls)]
        for corr in self._class_corr:
            obj_to_verb[corr[1]].append(corr[2])
        return obj_to_verb

    @property
    def anno_interaction(self) -> List[int]:
        """
        Number of annotated box pairs for each interaction class

        Returns:
            list[600]
        """
        return self._num_anno.copy()

    @property
    def anno_object(self) -> List[int]:
        """
        Number of annotated box pairs for each object class

        Returns:
            list[80]
        """
        num_anno = [0 for _ in range(self.num_object_cls)]
        for corr in self._class_corr:
            num_anno[corr[1]] += self._num_anno[corr[0]]
        return num_anno

    @property
    def anno_action(self) -> List[int]:
        """
        Number of annotated box pairs for each action class

        Returns:
            list[117]
        """
        num_anno = [0 for _ in range(self.num_action_cls)]
        for corr in self._class_corr:
            num_anno[corr[2]] += self._num_anno[corr[0]]
        return num_anno

    # @property
    # def objects(self) -> List[str]:
    #     """
    #     Object names

    #     Returns:
    #         list[str]
    #     """
    #     return self._objects.copy()

    # @property
    # def verbs(self) -> List[str]:
    #     """
    #     Verb (action) names

    #     Returns:
    #         list[str]
    #     """
    #     return self._verbs.copy()

    # @property
    # def interactions(self) -> List[str]:
    #     """
    #     Combination of verbs and objects

    #     Returns:
    #         list[str]
    #     """
    #     return [self._verbs[j] + ' ' + self.objects[i]
    #             for _, i, j in self._class_corr]

    def split(self, ratio: float) -> Tuple[VGSubset, VGSubset]:
        """
        Split the dataset according to given ratio

        Arguments:
            ratio(float): The percentage of training set between 0 and 1
        Returns:
            train(Dataset)
            val(Dataset)
        """
        perm = np.random.permutation(len(self._idx))
        n = int(len(perm) * ratio)
        return VGSubset(self, perm[:n]), VGSubset(self, perm[n:])

    def filename(self, idx: int) -> str:
        """Return the image file name given the index"""
        return self._filenames[self._idx[idx]]

    def image_size(self, idx: int) -> Tuple[int, int]:
        """Return the size (width, height) of an image"""
        return self._image_sizes[self._idx[idx]]

    # def map2coco(self, obj):
    #     obj_name = self.object_names[obj]
    #     if obj_name not in self.coco_names:return -1
    #     return self.coco_names.index(obj_name)

    def _load_annotation_and_metadata(self, f: dict, gdino_boxes) -> None:
        """
        Arguments:
            f(dict): Dictionary loaded from {anno_file}.json
        """
        idx = list(range(len(f)))

        num_anno = [0 for _ in range(self.num_interation_cls)]
        # for anno in f['annotation']:
        #     for hoi in anno['hoi']:
        #         num_anno[hoi] += 1

        
        annotations = []
        filenames = []
        emptys = []

        gdino_anno = {}
        for gdino_box in gdino_boxes:
            keep = [i for i in range(len(gdino_box['labels'])) if gdino_box['labels'][i] in self.object_names]
            gdino_box['labels_g'] = [self.object_names.index(label) for label in np.array(gdino_box['labels'])[keep]]
            gdino_box['scores_g'] = np.array(gdino_box['scores'], dtype=np.float32)[keep]
            gdino_box['boxes_g'] = np.array(gdino_box['boxes'], dtype=np.float32)[keep]
            gdino_anno[gdino_box['filename']] = gdino_box
            
        # print(self.interaction_names)
        assert self.object_names.index('person') == 0
        for i in idx:
            annos = f[i]['annotations']
            boxes_h, boxes_o, hois, objects, verbs = [], [], [], [], []
            labels_g, scores_g, boxes_g = [], [], []
             
            filename = str(f[i]['image_id']) + '.jpg'
            filenames.append(filename)
            for anno in annos:
                # print(anno)
                obj = anno['obj']
                verb = anno['verb']
                
                boxes_h.append(anno['box_h'])
                boxes_o.append(anno['box_o'])
                hois.append(self.interaction_names.index([verb, obj]))
                num_anno[self.interaction_names.index([verb, obj])] += 1
                objects.append(self.object_names.index(obj))
                verbs.append(self.verb_names.index(verb))
            if len(hois) == 0:
                emptys.append(i)
            else:
                labels_g = gdino_anno[filename]['labels_g']
                scores_g = gdino_anno[filename]['scores_g']
                boxes_g = gdino_anno[filename]['boxes_g']
                ### jump the img without box
                if len(labels_g) == 0:
                    emptys.append(i)

            annotations.append({'boxes_h':boxes_h, 'boxes_o':boxes_o, 'hoi':hois, 'objects':objects, 'verbs':verbs,
                                'labels_g':labels_g, 'scores_g':scores_g, 'boxes_g':boxes_g})

        for empty_idx in emptys:
            idx.remove(empty_idx)
        
        self._num_anno = num_anno
        self._idx = idx
        self._anno = annotations
        self._filenames = filenames
        # self._image_sizes = f['size']
        self._class_corr = [[self.interaction_names.index(inter), self.object_names.index(inter[1]), self.verb_names.index(inter[0])] for inter in self.interaction_names]
        self._empty_idx = emptys

        # self._objects = f['objects']
        # self._verbs = f['verbs']