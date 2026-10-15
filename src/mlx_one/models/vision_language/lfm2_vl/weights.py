"""Exact native LFM2-VL tensor contract; publisher conversion is never inferred."""
from mlx_one.core.weights import WeightContract
from mlx_one.models.language.lfm2.weights import weight_contract as text_contract


def sanitize_weights(weights,config):
    return dict(weights)


def weight_contract(config):
    expected={'language_model.'+name.removeprefix('model.'):shape
              for name,shape in text_contract(config.text_config).expected.items()
              if name.startswith('model.')}
    text,vision=config.text_config,config.vision_config
    hidden=vision.hidden_size
    patch_dim=vision.num_channels*vision.patch_size**2
    merged=hidden*vision.downsample_factor**2
    expected.update({'vision_model.patch_embedding.weight':(hidden,patch_dim),
        'vision_model.patch_embedding.bias':(hidden,),
        'vision_model.position_embedding.weight':((vision.image_size//vision.patch_size)**2,hidden),
        'vision_model.post_layernorm.weight':(hidden,),'vision_model.post_layernorm.bias':(hidden,),
        'vision_model.projector.weight':(vision.projection_dim,merged),
        'vision_model.projector.bias':(vision.projection_dim,)})
    if vision.use_projection_layernorm:
        expected.update({'vision_model.projector_norm.weight':(merged,),
                         'vision_model.projector_norm.bias':(merged,)})
    for index in range(vision.num_hidden_layers):
        prefix=f'vision_model.blocks.{index}.'
        for norm in ('layer_norm1','layer_norm2'):
            expected[prefix+norm+'.weight']=(hidden,)
            expected[prefix+norm+'.bias']=(hidden,)
        for name in ('q_proj','k_proj','v_proj','out_proj'):
            expected[prefix+'self_attn.'+name+'.weight']=(hidden,hidden)
            expected[prefix+'self_attn.'+name+'.bias']=(hidden,)
        expected.update({prefix+'fc1.weight':(vision.intermediate_size,hidden),
            prefix+'fc1.bias':(vision.intermediate_size,),
            prefix+'fc2.weight':(hidden,vision.intermediate_size),prefix+'fc2.bias':(hidden,)})
    if not config.tie_word_embeddings:
        expected['lm_head.weight']=(text.vocab_size,text.hidden_size)
    return WeightContract(expected)
