"""Multimodal batch parity on a tiny Qwen3.5, without downloaded weights."""
import base64
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (
    PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor,
    Qwen3_5Config, Qwen3_5Model, Qwen3_5TextConfig, Qwen3_5VisionConfig,
)

from kev.model import PointerHead, SPECIAL
from kev.serve import INFER_MAX_BRANCH, INFER_MAX_STATE
from kev.vision import VisionPredictor, decode_image


@pytest.fixture
def example():
    vocabulary = {"[UNK]": 0, "[PAD]": 1, "a": 2, "b": 3, "c": 4, "long": 5}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]")
    tok.add_special_tokens({"additional_special_tokens": SPECIAL + [
        "<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<|video_pad|>"]})
    # Exercise the explicit right-padding override without changing shared state.
    tok.padding_side = "left"
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(
            patch_size=16, temporal_patch_size=2, merge_size=2,
            size={"shortest_edge": 1024, "longest_edge": 16384}),
        tokenizer=tok,
        video_processor=Qwen3VLVideoProcessor(),
    )
    text = Qwen3_5TextConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention"], pad_token_id=tok.pad_token_id,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]},
    )
    vision = Qwen3_5VisionConfig(
        depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
        out_hidden_size=32, num_position_embeddings=16,
        patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
    )
    config = Qwen3_5Config(
        text_config=text.to_dict(), vision_config=vision.to_dict(),
        image_token_id=processor.image_token_id, video_token_id=processor.video_token_id,
        vision_start_token_id=processor.vision_start_token_id,
        vision_end_token_id=processor.vision_end_token_id,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        backbone = Qwen3_5Model(config).eval()
        head = PointerHead(32, dp=8).eval()
    backbone.set_attn_implementation("eager")
    predictor = VisionPredictor(processor, backbone, SimpleNamespace(head=head, device="cpu"), tok)
    images = []
    for size, color in [((64, 64), "red"), ((64, 32), "blue")]:
        buffer = BytesIO()
        Image.new("RGB", size, color).save(buffer, format="PNG")
        images.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    rec = {"state": "a b", "questions": [
        {"instr": "a", "options": ["a", "b"]},
        {"instr": "long " * 5, "options": ["c", "a b", "b"]},
        {"instr": "b", "options": ["c", "b"]},
    ]}
    return predictor, rec, images


def test_prefix_matches_separate_and_encodes_image_state_once(example):
    predictor, rec, images = example
    batches = []
    vision_calls, language_rows = [], []
    hook = predictor.backbone.register_forward_pre_hook(
        lambda module, args, kwargs: batches.append(kwargs["input_ids"].shape[0]), with_kwargs=True)
    vision_hook = predictor.backbone.visual.register_forward_pre_hook(lambda *args: vision_calls.append(True))
    def language_hook(module, args, kwargs):
        x = kwargs.get("input_ids")
        if x is None:
            x = kwargs["inputs_embeds"]
        language_rows.append(tuple(x.shape[:2]))
    language_handle = predictor.backbone.language_model.register_forward_pre_hook(language_hook, with_kwargs=True)
    together, metrics = predictor.probs(rec, images, max_state=INFER_MAX_STATE, max_branch=INFER_MAX_BRANCH)
    hook.remove()
    vision_hook.remove()
    language_handle.remove()
    assert batches == [1]
    assert vision_calls == [True]
    assert language_rows[0] == (1, metrics["state_tokens"])
    _, branches, _ = predictor.encode_prefix(rec, [decode_image(x) for x in images])
    assert language_rows[1:] == [tuple(branches["input_ids"].shape)]
    tokens = 0
    for probability, question in zip(together, rec["questions"]):
        separate, one_metrics = predictor.probs(
            {**rec, "questions": [question]}, images, max_state=INFER_MAX_STATE, max_branch=INFER_MAX_BRANCH)
        torch.testing.assert_close(probability, separate[0], atol=1e-6, rtol=1e-5)
        tokens += one_metrics["tokens"]
        assert metrics["state_tokens"] == one_metrics["state_tokens"]
    assert metrics["tokens"] == tokens


def full_probabilities(predictor, rec, images):
    """Reference: the original complete multimodal forward, without any cache."""
    inputs, readouts = predictor.encode_batch(rec, [decode_image(x) for x in images])
    with torch.inference_mode():
        hidden = predictor.backbone(**inputs, use_cache=False).last_hidden_state.float()
        return [predictor.head(hidden[i, d], hidden[i, oi]).softmax(-1)
                for i, (_, oi, d) in enumerate(readouts)]


def test_cached_matches_full_multimodal_forward(example):
    predictor, rec, images = example
    expected = full_probabilities(predictor, rec, images)
    actual, _ = predictor.probs(rec, images, max_state=INFER_MAX_STATE, max_branch=INFER_MAX_BRANCH)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_prefix_positions_reuse_and_chunks(example):
    predictor, rec, images = example
    state, branches, readouts = predictor.encode_prefix(rec, [decode_image(x) for x in images])
    # Two images with different grids are preprocessed once, not repeated per question.
    assert state["image_grid_thw"].tolist() == [[1, 4, 4], [1, 2, 4]]
    prefix = predictor.prefix(state)
    # Spatial M-RoPE positions differ from the physical token count.
    assert prefix.next_position != prefix.length

    def snapshot():
        tensors = {}
        for i, layer in enumerate(prefix.cache.layers):
            for name in ("keys", "values", "conv_states", "recurrent_states"):
                value = getattr(layer, name, None)
                items = value.items() if isinstance(value, dict) else [(None, value)]
                for key, tensor in items:
                    if isinstance(tensor, torch.Tensor):
                        tensors[i, name, key] = tensor.clone()
        return tensors

    before, length = snapshot(), prefix.cache.get_seq_length()
    one = predictor.probs_with_prefix(branches, readouts, prefix, rows_per_pass=1)
    # An unrelated image request must not overwrite this prefix's position state.
    predictor.probs({**rec, "state": "long " * 7}, images[:1],
                    max_state=INFER_MAX_STATE, max_branch=INFER_MAX_BRANCH)
    together = predictor.probs_with_prefix(branches, readouts, prefix, rows_per_pass=len(readouts))
    again = predictor.probs_with_prefix(branches, readouts, prefix, rows_per_pass=2)
    for a, b, c, expected in zip(one, together, again, full_probabilities(predictor, rec, images)):
        torch.testing.assert_close(a, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(a, c, atol=1e-6, rtol=1e-5)
    assert length == prefix.cache.get_seq_length()
    after = snapshot()
    assert before.keys() == after.keys()
    assert any(key[1] == "recurrent_states" for key in before)
    assert any(key[1] == "conv_states" for key in before)
    for key in before:
        torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)


def test_right_padding_and_multi_image_order(example):
    predictor, rec, images = example
    inputs, readouts = predictor.encode_batch(rec, [decode_image(x) for x in images])
    lengths = inputs["attention_mask"].sum(-1).tolist()
    assert min(lengths) < max(lengths)
    assert sum(lengths) < inputs["input_ids"].numel()
    assert inputs["image_grid_thw"].tolist() == [[1, 4, 4], [1, 2, 4]] * 3
    assert (inputs["mm_token_type_ids"] == 1).sum(-1).tolist() == [6, 6, 6]
    for row, (length, (_, _, decide)) in enumerate(zip(lengths, readouts)):
        assert decide == length - 1
        assert inputs["attention_mask"][row, :length].all()
        assert not inputs["attention_mask"][row, length:].any()
    assert predictor.processor.tokenizer.padding_side == "left"


def test_changing_one_question_preserves_other_rows(example):
    predictor, rec, images = example
    before, _ = predictor.probs(rec, images, max_state=INFER_MAX_STATE, max_branch=INFER_MAX_BRANCH)
    changed = {**rec, "questions": [rec["questions"][0],
        {"instr": "c " * 12, "options": ["b", "c", "a", "long"]}, rec["questions"][2]]}
    after, _ = predictor.probs(changed, images, max_state=INFER_MAX_STATE, max_branch=INFER_MAX_BRANCH)
    for index in (0, 2):
        torch.testing.assert_close(before[index], after[index], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("limits,match", [
    ({"max_state": 1, "max_branch": INFER_MAX_BRANCH}, "image state exceeds"),
    ({"max_state": INFER_MAX_STATE, "max_branch": 1}, "image question exceeds"),
])
def test_limits_reject_batch_before_forward(example, limits, match):
    predictor, rec, images = example
    calls = []
    hook = predictor.backbone.register_forward_pre_hook(lambda *args: calls.append(True))
    with pytest.raises(ValueError, match=match):
        predictor.probs(rec, images, **limits)
    hook.remove()
    assert not calls
