# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

from mmcv.runner.checkpoint import _load_checkpoint, load_state_dict


from collections import OrderedDict
import warnings


def match_layers_and_load_checkpoint(model, filename,
                           map_location='cpu',
                           logger=None):  
    checkpoint = _load_checkpoint(filename, map_location)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # state_dict = checkpoint['model']
        state_dict = checkpoint

    model_dict = model.state_dict()
    model_layers_by_ind = OrderedDict((i, k) for (i, k) in enumerate(model_dict))
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers, missing_layers = list(), list(), list()
    err_msg = []
    
    # Topological order of model and checkpoint layers is almost the same for non 'upsampling' / 'fc' layers
    for (i, k) in enumerate(state_dict.keys()):
        model_layer = model_layers_by_ind[i]
        k_split = k.split('.')
        if 'fc' in k_split[0]:
            discarded_layers.append(k)
            continue
        if not model_dict[model_layer].shape == state_dict[k].shape:
            # checkpoint layers order is shifted w.r.t model layers
            if k_split[-3] == 'c' and k_split[-4] == 'f':
                k_split[-3] = 'a'
            elif k_split[-3] == 'b' and k_split[-4] == 'f':
                k_split[-3] = 'c'
            elif k_split[-3] == 'a' and k_split[-4] == 'f':
                k_split[-3] = 'b'
            k = '.'.join(k_split)
        if not model_dict[model_layer].shape == state_dict[k].shape:
            discarded_layers.append(k)
        else:
            matched_layers.append(k)
        new_state_dict[model_layer] = state_dict[k]        
    missing_layers = [k for k in model_dict if k not in new_state_dict]

    # Older attempt - works, but perhaps above is better
    # for (k1, k2) in zip(model_dict.keys(), state_dict.keys()):
    #     k1_split = k1.split('.')
    #     k2_split = k2.split('.')
    #     if 'upsample' in k1_split[0] or 'fc' in k2_split[0]:
    #         discarded_layers.append((k1, k2))
    #         continue
    #     if not model_dict[k1].shape == state_dict[k2].shape:
    #         # checkpoint layers order is shifted w.r.t model layers
    #         if k2_split[-3] == 'c' and k2_split[-4] == 'f':
    #             k2_split[-3] = 'a'
    #         elif k2_split[-3] == 'b' and k2_split[-4] == 'f':
    #             k2_split[-3] = 'c'
    #         elif k2_split[-3] == 'a' and k2_split[-4] == 'f':
    #             k2_split[-3] = 'b'
    #         k2 = '.'.join(k2_split)
    #     if not model_dict[k1].shape == state_dict[k2].shape:
    #         discarded_layers.append((k1, k2))
    #     else:
    #         matched_layers.append((k1, k2))
    #         assert model_dict[k1].shape == state_dict[k2].shape,\
    #             f'layers {k1} and {k2} are matched but have '\
    #                 f'different size of {model_dict[k1].shape} and {state_dict[k2].shape} respectively'
    #     new_state_dict[k1] = state_dict[k2]
    #     done_layers.append(k1)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        msg = f'The pretrained weights "{filename}" cannot be loaded, '\
            'please check the key names manually '\
            '(** ignored and continue **)'
        if logger is not None:
            logger.warning(msg)
        else:
            print(msg)
    else:
        print('Successfully loaded pretrained weights\n')
        msg = f'** the following layers are discarded: {", ".join(discarded_layers)}\n'
        if len(discarded_layers) > 0:
            if logger is not None:
                logger.warning(msg)
            else:
                print(msg)
        msg = f'** missing layers in source state_dict: {", ".join(missing_layers)}\n'
        if len(missing_layers) > 0:
            if logger is not None:
                logger.warning(msg)
            else:
                print(msg)


def load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict_tmp = checkpoint['state_dict']
    else:
        state_dict_tmp = checkpoint

    state_dict = OrderedDict()
    # strip prefix of state_dict
    for k, v in state_dict_tmp.items():
        if k.startswith('module.backbone.'):
            state_dict[k[16:]] = v
        elif k.startswith('module.'):
            state_dict[k[7:]] = v
        elif k.startswith('backbone.'):
            state_dict[k[9:]] = v
        else:
            state_dict[k] = v
    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def get_state_dict(filename, map_location='cpu'):
    """Get state_dict from a file or URI.

    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``.
        map_location (str): Same as :func:`torch.load`.

    Returns:
        OrderedDict: The state_dict.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict_tmp = checkpoint['state_dict']
    else:
        state_dict_tmp = checkpoint

    state_dict = OrderedDict()
    # strip prefix of state_dict
    for k, v in state_dict_tmp.items():
        if k.startswith('module.backbone.'):
            state_dict[k[16:]] = v
        elif k.startswith('module.'):
            state_dict[k[7:]] = v
        elif k.startswith('backbone.'):
            state_dict[k[9:]] = v
        else:
            state_dict[k] = v

    return state_dict
