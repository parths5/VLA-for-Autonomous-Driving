"""Backbone implementation wrapping the Qwen3-VL instruct checkpoint."""

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
import numpy as np
import torchvision.io as io
import logging
from typing import Union, List, Dict

class Backbone:
    """Thin wrapper that provides multimodal query access to Qwen3 models."""

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-4B-Instruct"):
        """Initialise the model weights and paired processor for the chosen checkpoint.

        Parameters
        ----------
        model_name:
            HuggingFace model identifier to load. Defaults to ``Qwen/Qwen3-VL-4B-Instruct``.
        """
        self.model_name = model_name
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, dtype="auto", device_map="auto", low_cpu_mem_usage=True
        )

        self.processor = AutoProcessor.from_pretrained(model_name)

    def query(
        self,
        visual: Union[np.ndarray, torch.Tensor, None] = None,
        text: Union[str, None] = None,
        messages: Union[List[Dict], None] = None,
    ) -> List[str]:
        """Execute a single-turn or multi-turn conversation with the backbone.

        Parameters
        ----------
        visual:
            Image (H, W, C) or video (T, C, H, W) tensor/array to embed alongside
            ``text`` when ``messages`` is not supplied.
        text:
            Natural-language prompt to pair with ``visual`` when constructing a
            minimal message sequence.
        messages:
            Fully formatted conversation that already follows the processor chat
            template. If provided, ``visual`` and ``text`` are ignored.

        Returns
        -------
        list[str]
            Generated responses corresponding to the supplied conversations.

        Raises
        ------
        ValueError
            Raised when neither ``messages`` nor both ``visual`` and ``text`` are
            provided.
        """
        if messages is None:
            if visual is None or text is None:
                raise ValueError(
                    "Provide either (visual, text) or a prepared messages list."
                )
            content_type = "image" if len(visual.shape) == 3 else "video"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": content_type,
                            content_type: visual,
                        },
                        {"type": "text", "text": text},
                    ],
                }
            ]
        else:
            logging.debug("Querying model with %d-message conversation.", len(messages))
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        for key, value in list(inputs.items()):
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(next(self.model.parameters()).device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text

    def encode(self, visual: Union[np.ndarray, torch.Tensor]):
        if visual is np.ndarray:
            visual = torch.Tensor(visual)
        print(visual.shape)
        inputs = self.processor(
            images=visual,
            return_tensors="pt",
            text=self.processor.image_token
        ).to(self.model.device)

        features, deepstack_features = self.model.get_image_features(
                inputs['pixel_values'], 
                inputs['image_grid_thw']
            )
        return features, deepstack_features



if __name__ == "__main__":
    backbone = Backbone()
    image = io.read_image("media/ego-image-speed-limit.png", io.ImageReadMode.RGB)
    print(image.shape)
    answer = backbone.query(image, "What is the speed limit?")
    features, deepstack_features = backbone.encode(image)
    print(len(features))
    for feature in features:
        print(feature.shape)
    print("DeepStack Features:::")
    for feature in deepstack_features:
        print(feature.shape)
    print(answer)
    # vframes, aframes, _ = io.read_video(
    #     "media/speed-limit-signage.mp4",
    #     output_format="TCHW",
    # )
    # print(vframes.shape)
    # vframes_cropped = vframes[-128:, ...]
    # print(vframes_cropped.shape)
    # answer = backbone.query(
    #     vframes_cropped,
    #     "Imagine you are the driver of this car. What do you do next? Explain in detail.",
    # )
    # print(answer[0])