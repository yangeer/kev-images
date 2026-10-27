"""Image decisions using a Qwen3.5 backbone and the existing Kev head."""

import base64
import binascii
import copy
from dataclasses import dataclass
from io import BytesIO

import torch
from PIL import Image, UnidentifiedImageError

from .model import SPECIAL, user_text


def decode_image(value):
    try:
        raw = base64.b64decode(value, validate=True)
        with Image.open(BytesIO(raw)) as image:
            return image.convert("RGB")
    except (
        binascii.Error,
        ValueError,
        OSError,
        UnidentifiedImageError,
    ) as exc:
        raise ValueError("images must contain base64-encoded image files") from exc


@dataclass(frozen=True)
class VisionPrefix:
    cache: object
    length: int
    next_position: int


class VisionPredictor:
    def __init__(self, processor, backbone, model, tok):
        self.processor = processor
        self.backbone = backbone
        self.head = model.head
        self.device = model.device

        self.q_id = tok.convert_tokens_to_ids(SPECIAL[1])
        self.option_end_id = tok.convert_tokens_to_ids(SPECIAL[3])
        self.decide_id = tok.convert_tokens_to_ids(SPECIAL[4])

        # Use the same tokenizer as the existing Kev checkpoint.
        self.processor.tokenizer = tok

    def _state_prompt(self, rec, images):
        return SPECIAL[0] + user_text(rec["state"]) + (
            "<|vision_start|><|image_pad|><|vision_end|>" * len(images)
        )

    def _branch_prompts(self, rec):
        _, q_id, option_id, end_id, decide_id = SPECIAL
        prompts = []
        for question in rec["questions"]:
            prompt = q_id + user_text(question["instr"])
            for option in question["options"]:
                prompt += option_id + user_text(option) + end_id
            prompts.append(prompt + decide_id)
        return prompts

    def encode_batch(self, rec, images):
        """Full rows without caching, retained as the multimodal parity reference."""
        prefix = self._state_prompt(rec, images)
        prompts = [prefix + branch for branch in self._branch_prompts(rec)]
        inputs = self.processor(
            text=prompts,
            # Flatten images in sequence order: each row owns a full image set.
            images=images * len(prompts),
            add_special_tokens=False,
            padding=True,
            padding_side="right",
            return_mm_token_type_ids=True,
            return_tensors="pt",
        )

        return inputs, self._readouts(inputs["input_ids"], rec)

    def _readouts(self, ids_batch, rec):
        # Locate markers in either complete rows or question-only branches.
        readouts = []
        for ids, question in zip(ids_batch, rec["questions"]):
            question_idx = (ids == self.q_id).nonzero(as_tuple=True)[0]
            option_idx = (ids == self.option_end_id).nonzero(as_tuple=True)[0]
            decide_idx = (ids == self.decide_id).nonzero(as_tuple=True)[0]

            if question_idx.numel() != 1:
                raise ValueError("expected one question marker")
            if option_idx.numel() != len(question["options"]):
                raise ValueError("option marker count does not match the options")
            if decide_idx.numel() != 1:
                raise ValueError("expected one decision marker")
            readouts.append((question_idx.item(), option_idx, decide_idx.item()))

        return readouts

    def encode_prefix(self, rec, images):
        """Preprocess each image once and tokenize the independent text branches."""
        state = self.processor(
            text=[self._state_prompt(rec, images)],
            images=images,
            add_special_tokens=False,
            return_mm_token_type_ids=True,
            return_tensors="pt",
        )
        branches = self.processor.tokenizer(
            self._branch_prompts(rec),
            add_special_tokens=False,
            padding=True,
            padding_side="right",
            return_token_type_ids=False,
            return_tensors="pt",
        )
        return state, branches, self._readouts(branches["input_ids"], rec)

    @torch.inference_mode()
    def prefix(self, state):
        """Cache the image state, including hybrid recurrent states and its M-RoPE offset."""
        state = state.to(self.device)
        positions, _ = self.backbone.get_rope_index(
            state["input_ids"],
            image_grid_thw=state["image_grid_thw"],
            attention_mask=state["attention_mask"],
            mm_token_type_ids=state["mm_token_type_ids"],
        )
        out = self.backbone(
            **state, position_ids=positions, use_cache=True, return_dict=True,
        )
        return VisionPrefix(out.past_key_values, state["input_ids"].shape[1], int(positions.max()) + 1)

    @torch.inference_mode()
    def probs_with_prefix(self, branches, readouts, prefix, rows_per_pass):
        """Each chunk continues independent copies of a pristine request-local prefix."""
        branches = branches.to(self.device)
        probabilities = []
        for start in range(0, len(readouts), rows_per_pass):
            picks = readouts[start:start + rows_per_pass]
            ids = branches["input_ids"][start:start + len(picks)]
            mask = branches["attention_mask"][start:start + len(picks)]
            # Trim padding left over from longer branches in other chunks.
            width = int(mask.sum(-1).max())
            ids, mask = ids[:, :width], mask[:, :width]
            cache = copy.deepcopy(prefix.cache)
            cache.reorder_cache(torch.zeros(len(picks), dtype=torch.long, device=self.device))
            positions = torch.arange(width, device=self.device) + prefix.next_position
            positions = positions.view(1, 1, -1).expand(3, len(picks), -1)
            attention_mask = torch.cat([mask.new_ones((len(picks), prefix.length)), mask], dim=1)
            hidden = self.backbone.language_model(
                input_ids=ids, position_ids=positions, attention_mask=attention_mask,
                past_key_values=cache, use_cache=True,
            ).last_hidden_state.float()
            for row, (_, option_idx, decide_idx) in enumerate(picks):
                logits = self.head(hidden[row, decide_idx], hidden[row, option_idx.to(self.device)])
                probabilities.append(logits.softmax(dim=-1).cpu())
        return probabilities

    @torch.inference_mode()
    def probs(self, rec, encoded_images, max_state, max_branch):
        images = [decode_image(value) for value in encoded_images]
        state, branches, readouts = self.encode_prefix(rec, images)
        state_len = state["input_ids"].shape[1]
        lengths = [state_len + n for n in branches["attention_mask"].sum(-1).tolist()]

        for sequence_len in lengths:
            # Current Kev's max_branch bounds state + one question.
            # Do not truncate an expanded image-token sequence.
            if state_len > max_state:
                raise ValueError(
                    f"image state exceeds {max_state} tokens: {state_len}"
                )
            if sequence_len > max_branch:
                raise ValueError(
                    f"image question exceeds {max_branch} tokens: "
                    f"{sequence_len}"
                )

        prefix = self.prefix(state)
        # Bound replicated state caches and branch activations by a full-row token budget.
        rows_per_pass = max(1, max_branch // max(lengths))
        probabilities = self.probs_with_prefix(branches, readouts, prefix, rows_per_pass)

        return probabilities, {
            # Preserve logical input-token accounting, even though the state is computed once.
            "tokens": sum(lengths),
            "state_tokens": state_len,
        }
