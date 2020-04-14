import torch
from collections import OrderedDict


param_dict_nvidia = torch.load('/home/weik/PycharmProjects/FreeAnchor/pretrainedmodel/retinanet_rn50fpn.pth')
weights_nvidia= param_dict_nvidia['state_dict']
param_dict_freeanchor = torch.load('/home/weik/PycharmProjects/FreeAnchor/pretrainedmodel/free_anchor_R-50-FPN_1x.pth')
weights_freeanchor = param_dict_freeanchor['model']



###################### key_map: free_anchor --> nvidia #############################################
original_keys = sorted(param_dict_freeanchor['model'].keys())
layer_keys = sorted(param_dict_freeanchor['model'].keys())

# backbone conv1
layer_keys = [k.replace("module.backbone.body.stem", "backbones.ResNet50FPN.features") for k in layer_keys]
# backbone other
layer_keys = [k.replace("module.backbone.body", "backbones.ResNet50FPN.features") for k in layer_keys]
# head
layer_keys = [k.replace("module.rpn.head.cls_tower", "cls_head") for k in layer_keys]
layer_keys = [k.replace("module.rpn.head.cls_logits", "cls_head.8") for k in layer_keys]
layer_keys = [k.replace("module.rpn.head.bbox_tower", "box_head") for k in layer_keys]
layer_keys = [k.replace("module.rpn.head.bbox_pred", "box_head.8") for k in layer_keys]
# fpn
layer_keys = [k.replace("module.backbone.fpn.fpn_inner2", "backbones.ResNet50FPN.lateral3") for k in layer_keys]
layer_keys = [k.replace("module.backbone.fpn.fpn_layer2", "backbones.ResNet50FPN.smooth3") for k in layer_keys]
layer_keys = [k.replace("module.backbone.fpn.fpn_inner3", "backbones.ResNet50FPN.lateral4") for k in layer_keys]
layer_keys = [k.replace("module.backbone.fpn.fpn_layer3", "backbones.ResNet50FPN.smooth4") for k in layer_keys]
layer_keys = [k.replace("module.backbone.fpn.fpn_inner4", "backbones.ResNet50FPN.lateral5") for k in layer_keys]
layer_keys = [k.replace("module.backbone.fpn.fpn_layer4", "backbones.ResNet50FPN.smooth5") for k in layer_keys]
layer_keys = [k.replace("module.backbone.fpn.top_blocks.p", "backbones.ResNet50FPN.pyramid") for k in layer_keys]

key_map = {k: v for k, v in zip(original_keys, layer_keys)}


new_weights = OrderedDict()
count = 0
for k in original_keys:
    if 'anchor_generator' in k:
        continue
    v_freeanchor = weights_freeanchor[k]
    v_nvidia = weights_nvidia[key_map[k]]

    count = count + 1
    # assert v_freeanchor.shape == v_nvidia.shape
    print('{}  {}: {} --> {}'.format(count, k, v_freeanchor.shape, v_nvidia.shape))
    new_weights[key_map[k]] = v_freeanchor

new_weights[]

"""

###################### key_map: nvidia --> free_anchor #############################################
original_keys = sorted(param_dict_nvidia['state_dict'].keys())
layer_keys = sorted(param_dict_nvidia['state_dict'].keys())

# backbone conv1
layer_keys = [k.replace("backbones.ResNet50FPN.features.conv1", "module.backbone.body.stem") for k in layer_keys]
# backbone resnet layer
layer_keys = [k.replace("backbones.ResNet50FPN.features", "module.backbone.body") for k in layer_keys]
# head
layer_keys = [k.replace("cls_head", "module.rpn.head.cls_tower") for k in layer_keys]
layer_keys = [k.replace("cls_head.8", "module.rpn.head.cls_logits") for k in layer_keys]
layer_keys = [k.replace("box_head", "module.rpn.head.bbox_tower") for k in layer_keys]
layer_keys = [k.replace("box_head.8", "module.rpn.head.bbox_pred") for k in layer_keys]
# fpn
layer_keys = [k.replace("backbones.ResNet50FPN.lateral3", "module.backbone.fpn.fpn_inner2") for k in layer_keys]
layer_keys = [k.replace("backbones.ResNet50FPN.smooth3", "module.backbone.fpn.fpn_layer2") for k in layer_keys]
layer_keys = [k.replace("backbones.ResNet50FPN.lateral4", "module.backbone.fpn.fpn_inner3") for k in layer_keys]
layer_keys = [k.replace("backbones.ResNet50FPN.smooth4", "module.backbone.fpn.fpn_layer3") for k in layer_keys]
layer_keys = [k.replace("backbones.ResNet50FPN.lateral5", "module.backbone.fpn.fpn_inner4") for k in layer_keys]
layer_keys = [k.replace("backbones.ResNet50FPN.smooth5", "module.backbone.fpn.fpn_layer4") for k in layer_keys]
layer_keys = [k.replace("backbones.ResNet50FPN.pyramid", "module.backbone.fpn.top_blocks.p") for k in layer_keys]

key_map = {k: v for k, v in zip(original_keys, layer_keys)}


new_weights = OrderedDict()
count = 0
for k in original_keys:
    v_nvidia = weights_nvidia[k]
    if 'fc' in k:
        v_freeanchor = v_nvidia
    else:
        v_freeanchor = weights_freeanchor[key_map[k]]

    count = count + 1
    print('{}  {}: {} --> {}'.format(count, k, v_freeanchor.shape, v_nvidia.shape))
    new_weights[k] = v_freeanchor
"""

transferred_model = param_dict_nvidia.copy()
transferred_model['state_dict'] = new_weights
torch.save(transferred_model, 'transferred_freeanchor_resnet50fpn.pth')