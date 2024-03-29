# HOTS: **H**ousehold **O**bjects in **T**abletop **S**cenarios
<img src="assets/catalogue_temp.jpg" width="400"/> <img src="assets/robot_setup.png" width="400"/> 

HOTS in an ongoing effort to collect an RGB-D dataset aimed for benchmarking visual and 3D recognition, sim-to-real domain adaptation (and more) in robotics-related domains. It contains a broad range of typical household objects annotated in multiple levels of granularity for class labels (supercategory, category and instance), as well as visual attribute annotations (color, material etc.) and pair-wise spatial relations in the scene-level (scene graphs). The current version of the dataset contains only single view RGB-D data captured from an ASUS Xtion sensor mounted on a real-robot setup.

The dataset enumerates a total of 46 object instances organized as
| Annotation        | Number           | Classes  |
| ------------- |:-------------:| :-----|
| Supercategory      | 5 | edible products(13), electronics(5), fruits(6), kitchenware(11), stationery(11) | 
| Category    | 25   |   apple(1), banana(1), book(3), bowl(1), soda can(5), cup(3), fork(2), juice box(3), keyboard(1), knife(1), laptop(1), lemon(1), marker(2), milk box(2), monitor(1), mouse(2), orange(1), peach(1), pear(1), pen(3), plate(2), pringles box(3), scissors(2), spoon(2), stapler(1)   |
| Color | 11      |    red(6), yellow(4), blue(5), white(6), purple(2), green(4), black(9), transparent(1), silver(6), orange(2), pink(1) |
| Material | 7      |    organic(6), paper(7), ceramic(5), aluminium(5), glass(1), metal(8), plastic(14) |

## Data Structure
Download and unzip the dataset from [here](https://drive.google.com/file/d/1dldyn6CnUe_K-eDqr5lKmcIq1FkAp-Or/view?usp=sharing)

Inside the root directory you will find two sub-folders, namely: a) *object*, that containts object-level RGB-D images aimed for object recognition task, cropped from the original scene frames according to their bounding box annotation, and b) *scene*, that contains scene-level RGB-D images, organized by title in different splits according to the type of objects appearing (table, kitchen, office, mix). Annotations contain bounding boxes for object detection and pixel-level masks for semantic / instance segmentation tasks.

The object-level directory structure follows the classic *ImageFolder* class-per-folder style:
```bash
#  - ./HOTS/object
#      - apple
#        - 0.png
#        - 1.png
#        - ...
#      ...
#      - stapler
#        - 0.png
#        - 1.png
#        - ...
```

The scene-level directory follows the VOC-style structure for each task

Image data
```bash
#  - ./HOTS/scene
#    - RGB
#      - kitchen_5_top_raw_0.png
#      - ...
#      - table_8_top_raw_9.png
```

Object Detection
```bash
#  - ./HOTS/scene
#    - ObjectDetection
#      - Annotations
#        - kitchen_5_top_raw_0.xml
#        - ...
#      - AnnotationsVisualization
#        - kitchen_5_top_raw_0.jpg
#        - ...
#      - class_names.txt
```

Semantic / Instance Segmentation
```bash
#  - ./HOTS/scene
#    - SemanticSegmentation
#      - SegmentationClass
#        - kitchen_5_top_raw_0.npy
#        - ...
#      - SegmentationClassPNG
#        - kitchen_5_top_raw_0.png
#        - ...
#      - SegmentationClassVisualization
#        - kitchen_5_top_raw_0.jpg
#        - ...
#
#    - InstanceSegmentation
#      - SegmentationObject
#        - kitchen_5_top_raw_0.npy
#        - ...
#      - SegmentationObjectPNG
#        - kitchen_5_top_raw_0.png
#        - ...
#      - SegmentationObjectVisualization
#        - kitchen_5_top_raw_0.jpg
#        - ...
#    - class_names.txt
``` 
The -Class folder contains raw pixel-level masks using the unique index of each class (as ordered in class_names.txt) or instance. The -PNG folder ommits the background and show coloured version of the pixel-level annotations. The *Visualization* folders contain the RGB data with drawn annotations for each task
| Object Detection | Semantic Segmentation | Instance Segmentation  |
| :---------------------: |:---------------------:| :---------------------:|
| <img src="assets/obj_det.gif" width="400"/> | <img src="assets/sem_segm.gif" width="400"/> |<img src="assets/inst_segm.gif" width="400"/> |


## APIs
Python API for loading the datasets. Use ```transform=True``` to tensorize the data. Alternatively, pass a ```torchvision.transforms``` object to transform the data according to your desired preprocessing.

Object-level:
```python
from hots import load_HOTS_objects

train_dataset, test_dataset = load_HOTS_objects(transform=False)
image, label = train_dataset[0]
# image: np.array[uint8] (H x W x 3, raw pixel values)
# label: str

train_dataset, test_dataset = load_HOTS_objects(transform=True)
image, label = train_dataset[0]
# image: torch.Tensor[float32] (3 x H x W, normalized)
# label: torch.Tensor[int64] (unique index from labels.txt)
```

Scene-level:
```python
from hots import load_HOTS_scenes

train_dataset, test_dataset = load_HOTS_scenes(transform=False)
image, target = train_dataset[0]
# image: np.array[uint8] (H x W x 3, raw pixel values)
# target: Dict[str, Any]:
#     'boxes'          :   np.int32 (4, raw bounding box coordinates)
#     'labels'         :   np.int32 (N, unique index of all N appearing objects according to labels.txt order)
#     'class_masks'    :   np.uint8 (N x H x W, pixel-level mask containing class labels as pixel values)
#     'instance_masks' :   np.uint8 (N x H x W, pixel-level mask containing an integer [0, N-1] for each object instance as pixel values)
#     'image_id'       :   int32 (A unique indentifier of the input image)
#     'area'           :   int32 (The total area of the bounding box, used for COCO-like object detection evaluation)
#     'image_size'     :   Tuple[int3, int32] (The resolution of the input image)
```
Similarly, use ```transform=True``` or a pre-defined ```torchvision``` transform to preprocess the raw image data. Using ```transform=True``` will also tensorize the values of the ```target``` dictionary and will additionaly normalize bounding box coordinates according to the ```image_size``` field.


