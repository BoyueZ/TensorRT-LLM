# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tensorrt_llm import SamplingParams
from tensorrt_llm.inputs.multimodal_data import VideoData

from ._engine import shortcut_text_engine


@pytest.mark.parametrize("precision", ["bf16", "mxfp8", "nvfp4"])
def test_checkpoint_video_sampling(model_root, precision, tmp_path):
    import cv2
    from transformers import AutoProcessor, AutoTokenizer

    from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config
    from tensorrt_llm.inputs.media_io import VideoMediaIO
    from tensorrt_llm.inputs.registry import (
        create_input_processor,
        create_input_processor_with_hash,
    )

    model = model_root / precision
    if not model.is_dir():
        pytest.skip(f"Checkpoint unavailable: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    processor = create_input_processor(str(model), tokenizer, video_pruning_rate=0.5)
    assert load_pretrained_config(str(model)).get_media_config().video_temporal_patch_size == 1
    text = {"prompt": "A photo of"}
    assert processor({**text, "multi_modal_data": {"video": []}}, SamplingParams()) == processor(
        text, SamplingParams()
    )
    frames = np.zeros((8, 56, 56, 3), dtype=np.uint8)
    frames[:, :, :, :] = np.arange(8)[:, None, None, None]
    video = VideoData(frames=frames, metadata={"fps": 2, "total_num_frames": 8})
    options = SamplingParams(add_special_tokens=False)
    request = {"prompt": "<nemo:video> Describe.", "multi_modal_data": {"video": [video]}}
    official = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    _, times = official.sample_video_frames(8, 2, temporal_patch_size=1)
    reference = official.get_video_prompt(times, temporal_patch_size=1, frame_size=(56, 56))
    plain = create_input_processor(str(model), tokenizer)
    raw_ids, raw_extra = create_input_processor_with_hash(plain)(request, options)
    assert raw_ids == tokenizer.encode(reference + " Describe.", add_special_tokens=False)
    ids, extra = create_input_processor_with_hash(processor)(request, options)
    data = extra["multimodal_data"]["video"]
    # Eight 2x2 frames have 32 projected positions. Exactly 16 survive at rate 0.5.
    full_ids = data["templates"][0]
    patch = processor.config.image_token_id
    assert full_ids.count(patch) == 32
    assert ids.count(patch) == len(full_ids) - 16
    assert len(data["patches"]) == 8 * 16
    assert data["patches"].equal(raw_extra["multimodal_data"]["video"]["patches"])
    frame_text = tokenizer.decode(full_ids, skip_special_tokens=False)
    assert "Frame 2 sampled at 0.56 seconds" in frame_text
    assert "Frame 8 sampled at 2.81 seconds" in frame_text
    expected = [0, 1, 1, 2, 3, 4, 4, 5]
    first_rows = data["patches"].reshape(8, 16, -1)[:, 0]
    for actual, index in zip(first_rows, expected, strict=True):
        reference = processor.preprocess_image(frames[index])[0][0]
        assert actual.equal(reference)
    clip = tmp_path / "frames.avi"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 2, (56, 56))
    try:
        assert writer.isOpened()
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    video_options = processor.get_preferred_media_io_kwargs()["video"]
    with pytest.raises(ValueError, match="Only the model processor may set frame_sampler"):
        VideoMediaIO.create(video_options, {"frame_sampler": None, "num_frames": 4})
    decoded = VideoMediaIO.create(video_options, {}).load_bytes(clip.read_bytes())
    assert decoded.metadata["sampled_frame_indices"] == expected
    assert decoded.metadata["sampled_timestamps"] == times
    assert decoded.metadata["frames_indices"] == sorted(set(expected))
    assert len(decoded.frames) == len(set(expected))


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("precision", ["bf16", "mxfp8", "nvfp4"])
@pytest.mark.parametrize("rate", [0.0, 0.5], ids=["unpruned", "evs"])
def test_evs_conditioned_video(model_root, precision, rate, record_property):
    from transformers import AutoProcessor

    model = model_root / precision
    with shortcut_text_engine(
        model,
        disable_mm_encoder=False,
        multimodal_config={"video_pruning_rate": rate},
        max_batch_size=1,
        max_seq_len=2048,
        max_num_tokens=2048,
        cuda_graph_config=None,
        enable_chunked_prefill=False,
        kv_cache_config={
            "use_kv_cache_manager_v2": True,
            "enable_block_reuse": False,
        },
    ) as llm:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {
                        "type": "text",
                        "text": "Is the foreground subject a person or a cat? Answer with one word.",
                    },
                ],
            }
        ]
        prompt = llm.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        assets = Path(__file__).resolve().parents[6] / "examples/visual_gen"
        for name, filename in (
            ("cat", "cat_piano.png"),
            ("person", "serve/media/woman_skyline_original_720p.jpeg"),
        ):
            with Image.open(assets / filename) as source:
                image = source.convert("RGB")
                image.thumbnail((224, 224), Image.Resampling.LANCZOS)
                frame = np.asarray(image)
            frames = np.stack([np.roll(frame, shift, axis=1) for shift in range(8)])
            video = VideoData(frames=frames, metadata={"fps": 2.0, "total_num_frames": 8})
            sampling = SamplingParams(temperature=0, top_k=1, max_tokens=16)
            output = llm.generate(
                {"prompt": prompt, "multi_modal_data": {"video": [video]}},
                sampling,
            )
            reply = output.outputs[0]
            record_property(f"{name}_video", (reply.text, list(reply.token_ids)))
            if not rate:
                # For T=1 the checkpoint's get_video_features calls its image tower on
                # these exact frames. Compare both ingress paths all the way to decode.
                official = AutoProcessor.from_pretrained(model, trust_remote_code=True)
                indices, times = official.sample_video_frames(8, 2, temporal_patch_size=1)
                oracle = llm.generate(
                    {
                        "prompt": prompt.replace("<nemo:video>", official.get_video_prompt(times)),
                        "multi_modal_data": {"image": [frames[index] for index in indices]},
                    },
                    sampling,
                )
                assert output.prompt_token_ids == oracle.prompt_token_ids
                record_property(f"{name}_reference", list(oracle.outputs[0].token_ids))
                assert reply.token_ids == oracle.outputs[0].token_ids
            labels = {"person", "woman", "human"} if name == "person" else {"cat", "kitten"}
            assert reply.text.strip().casefold().rstrip(".!") in labels, reply.text
