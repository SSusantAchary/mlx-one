"""Declared native-format static-image preprocessing for LFM2-VL."""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PreparedLFMImage:
    input_ids: object
    pixel_values: object
    spatial_shapes: tuple


class LFMImageProcessor:
    def __init__(self,tokenizer,config,metadata):
        if metadata.get('image_processor_type')!='MLXOneLFM2VLImageProcessor':
            raise ValueError('LFM2-VL requires a validated native-format processor contract')
        self.tokenizer,self.config=tokenizer,config
        self.mean,self.std=metadata['image_mean'],metadata['image_std']
        if (len(self.mean)!=3 or len(self.std)!=3
                or any(not math.isfinite(v) for v in (*self.mean,*self.std))
                or any(v<=0 for v in self.std)):
            raise ValueError('invalid LFM2-VL normalization metadata')

    def prepare(self,prompt,images):
        import mlx.core as mx
        from PIL import Image

        if not images or prompt.count('<image>')!=len(images):
            raise ValueError('image placeholders must match images')
        v=self.config.vision_config
        if v.num_channels!=3:
            raise ValueError('LFM2-VL processor requires RGB')
        patches=[]
        ids=self.tokenizer.encode(prompt.split('<image>')[0])
        grid=v.image_size//v.patch_size
        for image,text in zip(images,prompt.split('<image>')[1:],strict=True):
            source=image if isinstance(image,Image.Image) else Image.open(image)
            try:
                pixels=mx.array(list(source.convert('RGB').resize((v.image_size,v.image_size),
                    Image.Resampling.BICUBIC).getdata()),dtype=mx.float32).reshape(
                        v.image_size,v.image_size,3)/255.0
            finally:
                if source is not image:
                    source.close()
            pixels=(pixels-mx.array(self.mean))/mx.array(self.std)
            patches.append(pixels.reshape(grid,v.patch_size,grid,v.patch_size,3).transpose(
                0,2,4,1,3).reshape(grid*grid,-1))
            ids+=[self.config.image_token_id]*(grid*grid//v.downsample_factor**2)
            ids+=self.tokenizer.encode(text)
        return PreparedLFMImage(mx.array([ids]),mx.stack(patches),
                                tuple((grid,grid) for _ in images))
