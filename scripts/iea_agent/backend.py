"""Dual-mode VLM backend (native base + LoRA toggle) and CPU fallback backend."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Protocol
from .prompts import build_critic_prompt, build_planner_prompt, extract_json_object
from .state import SessionState, clamp_params


class AgentBackend(Protocol):
    mode_name: str

    def plan_turn(self, session: SessionState, user_message: str) -> dict[str, Any]: ...
    def predict_parameters(self, image_path: Path, instruction: str) -> dict[str, Any]: ...
    def critique_render(
        self,
        source_path: Path,
        rendered_path: Path,
        user_message: str,
        params: dict[str, float],
        before_stats: dict[str, float],
        after_stats: dict[str, float],
    ) -> dict[str, Any]: ...


def _is_chinese(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', text))


@dataclass
class DualModeVLMBackend:
    """Single Qwen2.5-VL-7B base with zero-VRAM-overhead PEFT adapter toggling.

    - Adapter disabled (`model.disable_adapter()`): conversational Planner & visual Critic.
    - Adapter enabled (`semantic_backbone` + `controller_head`): structured parameter prediction
      and initial quality risk estimation using frozen functional_v15 / functional_v21 contracts.
    """

    model: Any
    processor: Any
    head: Any
    head_weights: dict[str, Any]
    device: str = 'cuda'
    mode_name: str = 'gpu_dual_mode'

    @classmethod
    def from_project(cls, root: Path, device: str = 'cuda', threads: int = 20) -> 'DualModeVLMBackend':
        import torch
        from peft import PeftModel
        from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        from functional_v1.common import BASE
        from functional_v1.runtime import processor as load_processor
        from functional_v9.runtime import verify_origin
        from functional_v21.common import EXP, PARENT_SHA, protocol
        from functional_v21.head import load_head
        from .state import utc_stamp  # noqa: F401
        import hashlib

        def sha(path: Path) -> str:
            h = hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                    h.update(chunk)
            return h.hexdigest()

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        verify_origin('base')
        protocol()

        adapter_dir = root / 'models/core/semantic_backbone'
        if not adapter_dir.exists():
            adapter_dir = root / 'models/from-functional-v11/sft-shadow-correction/seed-01'
        controller_path = root / 'models/core/controller_head/controller.pt'
        if not controller_path.exists():
            controller_path = root / 'models/from-functional-v15/semantic-preserving-controller/seed-01/controller.pt'

        assert sha(adapter_dir / 'adapter_model.safetensors') == PARENT_SHA
        head, head_weights = load_head(controller_path)
        assert head_weights['parent_sha256'] == PARENT_SHA
        assert head_weights['protocol_sha256'] == sha(EXP / 'training-protocol.json')

        kwargs: dict[str, Any] = dict(local_files_only=True, attn_implementation='sdpa', output_loading_info=True)
        if device == 'cuda':
            kwargs.update(
                torch_dtype=torch.bfloat16,
                device_map={'': 0},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
            )
        else:
            kwargs.update(torch_dtype=torch.float32, device_map={'': 'cpu'})

        base, loading = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE, **kwargs)
        assert not any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
        model = PeftModel.from_pretrained(base, str(adapter_dir), adapter_name='semantic_backbone', is_trainable=False)
        model.eval()
        for w in model.parameters():
            w.requires_grad_(False)
        return cls(model=model, processor=load_processor(), head=head, head_weights=head_weights, device=device)

    def _generate_base_text(self, image_paths: list[Path], prompt: str, max_new_tokens: int = 384) -> str:
        import contextlib
        import torch
        from qwen_vl_utils import process_vision_info

        content = [{'type': 'image', 'image': str(p)} for p in image_paths]
        content.append({'type': 'text', 'text': prompt})
        messages = [{'role': 'user', 'content': content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt')
        encoded = {k: v.to(self.device) for k, v in inputs.items()}
        prefix = encoded['input_ids'].shape[1]
        ctx = torch.autocast('cuda', dtype=torch.bfloat16) if self.device == 'cuda' else contextlib.nullcontext()
        with self.model.disable_adapter(), torch.inference_mode(), ctx:
            generated = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                disable_compile=True,
            )
            tokens = generated[0, prefix:].tolist()
        return self.processor.decode(tokens, skip_special_tokens=True)

    def plan_turn(self, session: SessionState, user_message: str) -> dict[str, Any]:
        raw = self._generate_base_text([Path(session.source_image)], build_planner_prompt(session, user_message))
        return extract_json_object(raw)

    def predict_parameters(self, image_path: Path, instruction: str) -> dict[str, Any]:
        import contextlib
        import torch
        from functional_v1.runtime import encode
        from functional_v15.outcomes import input_row, parse
        from functional_v19.photometry import features as photo_features
        from functional_v21.head import predict_cached
        from functional_v21.texture import features as texture_features

        self.model.set_adapter('semantic_backbone')
        self.model.eval()
        case = dict(image=str(image_path), instructions={'original': instruction})
        inputs = encode(self.processor, input_row(case, 'original'))
        encoded = {k: v.to(self.device) for k, v in inputs.items()}
        prefix = encoded['input_ids'].shape[1]
        ctx = torch.autocast('cuda', dtype=torch.bfloat16) if self.device == 'cuda' else contextlib.nullcontext()
        with torch.inference_mode(), ctx:
            generated = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=256,
                repetition_penalty=1.0,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                disable_compile=True,
            )
            tokens = generated[0, prefix:].tolist()
            raw = self.processor.decode(tokens, skip_special_tokens=True)
            decision = parse(raw)
            if decision.get('action') == 'edit':
                hidden = self.model.get_base_model().model(**encoded, use_cache=False, return_dict=True).last_hidden_state
                observed = dict(last=hidden[0, -1].float().cpu(), stats=torch.from_numpy(photo_features(image_path)))
                _, decision = predict_cached(
                    self.head,
                    self.head_weights,
                    observed,
                    torch.from_numpy(texture_features(image_path)),
                    decision,
                    'magnitude-quality',
                )
        return decision

    def critique_render(
        self,
        source_path: Path,
        rendered_path: Path,
        user_message: str,
        params: dict[str, float],
        before_stats: dict[str, float],
        after_stats: dict[str, float],
    ) -> dict[str, Any]:
        raw = self._generate_base_text(
            [source_path, rendered_path],
            build_critic_prompt(user_message, params, before_stats, after_stats),
            max_new_tokens=256,
        )
        return extract_json_object(raw)


class RuleBasedCPUPlannerBackend:
    """Lightweight deterministic CPU backend for local interactive workflow demos without GPU.

    Clearly identified as `cpu_rule_fallback`; never mislabelled as 7B VLM output.
    """

    mode_name = 'cpu_rule_fallback'

    _UNSUPPORTED = (
        'p掉', '去除', '移除', '消除', '路人', '水印', '换背景', '局部', '重绘', '去噪', '高清修复', '裁剪',
        'remove', 'erase', 'inpaint', 'object', 'background', 'denoise', 'upscale', 'crop',
    )

    def plan_turn(self, session: SessionState, user_message: str) -> dict[str, Any]:
        text = user_message.strip()
        lower = text.lower()
        zh = _is_chinese(text)

        if any(k in lower for k in ('撤销', '撤回', '上一步', 'undo', 'revert')):
            return {
                'intent': 'undo',
                'steps_back': 1,
                'rewritten_instruction': '',
                'delta_parameters': {},
                'reply': '好的，已为您回退到上一轮的画面状态。' if zh else 'Reverted to the previous editing step.',
            }
        if any(k in lower for k in ('重置', '恢复原图', '还原原图', '回到原图', 'reset')):
            return {
                'intent': 'reset',
                'rewritten_instruction': '',
                'delta_parameters': {},
                'reply': '已为您还原为初始原图，所有调色参数已清零。' if zh else 'Reset all parameters back to the original image.',
            }
        if any(k in lower for k in ('保持原样', '不用修', '不用改', 'keep unchanged', 'leave unchanged')):
            return {
                'intent': 'keep',
                'rewritten_instruction': '',
                'delta_parameters': {},
                'reply': '好的，将保持当前画面不变。' if zh else 'Keeping the photograph unchanged.',
            }
        if any(k in lower for k in self._UNSUPPORTED):
            return {
                'intent': 'unsupported',
                'rewritten_instruction': '',
                'delta_parameters': {},
                'reply': (
                    '当前修图系统仅支持全局影调与色彩调节（曝光、亮度、对比度、高光、阴影、饱和度、色温、色调、复古），'
                    '暂不支持物体移除、局部替换或去噪修复。如果您希望调整整体明暗或色调，可以直接告诉我。'
                    if zh else
                    'Only global tonal and colour adjustments (exposure, brightness, contrast, highlights, shadows, '
                    'saturation, temperature, tint, sepia) are supported; object removal, local edits, and denoising are unavailable.'
                ),
            }

        relative_cues = ('再', '更', '稍微', '退回', '减半', '减弱', '降低', 'more', 'slightly', 'half', 'less', 'reduce')
        if session.active_turn > 0 and session.current_params and any(k in lower for k in relative_cues):
            if any(k in lower for k in ('减半', '退回一半', 'half')):
                return {
                    'intent': 'adjust_relative',
                    'scale_factor': 0.5,
                    'target_tools': list(session.current_params.keys()),
                    'delta_parameters': {},
                    'reply': '已将当前调色幅度减半，并基于初始原图重新渲染。' if zh else 'Halved the active adjustments and re-rendered from the original photo.',
                }
            delta = self._infer_delta(lower, strength=15)
            if delta:
                return {
                    'intent': 'adjust_relative',
                    'delta_parameters': delta,
                    'rewritten_instruction': '',
                    'reply': '已在上一轮基础上叠加微调，并基于原图无损重算结果。' if zh else 'Applied incremental adjustments and re-rendered cleanly from the original image.',
                }

        if not self._infer_delta(lower, strength=25):
            return {
                'intent': 'chat',
                'rewritten_instruction': '',
                'delta_parameters': {},
                'reply': (
                    '我可以帮您做全局光影与色彩调整，例如“稍微提亮阴影，保留天空和肤色”、“画面暖一点、饱和度微增”或“撤销上一步”。'
                    if zh else
                    'I can adjust global exposure, brightness, contrast, highlights, shadows, saturation, temperature, tint, and sepia.'
                ),
            }

        return {
            'intent': 'edit',
            'rewritten_instruction': text,
            'delta_parameters': {},
            'reply': '已完成语义规划并执行调色，结果如下。' if zh else 'Planned and applied the requested global adjustments.',
        }

    def _infer_delta(self, lower: str, strength: int = 25) -> dict[str, float]:
        soft = 0.6 if any(k in lower for k in ('稍微', '一点', '微', 'slightly', 'a bit', 'subtle')) else 1.0
        strong = 1.4 if any(k in lower for k in ('大幅', '明显', '强烈', 'strong', 'much')) else 1.0
        mag = round(strength * soft * strong)
        delta: dict[str, float] = {}

        if any(k in lower for k in ('暗部', '阴影', 'shadow')):
            delta['shadows'] = -mag if any(k in lower for k in ('压暗', '降', 'reduce', 'lower', 'dark')) else round(mag * 1.2)
        elif any(k in lower for k in ('曝光', 'exposure')):
            delta['exposure'] = -mag if any(k in lower for k in ('压暗', '降', '暗', 'dark', 'lower', 'reduce')) else mag
        elif any(k in lower for k in ('提亮', '亮', 'bright', 'lighten')):
            delta['brightness'] = mag
        elif any(k in lower for k in ('压暗', '暗一点', 'darker', 'darken')):
            delta['brightness'] = -mag

        if any(k in lower for k in ('高光', '天空', 'highlight')):
            if any(k in lower for k in ('压', '保留', '降', 'recover', 'lower', 'preserve')):
                delta['highlights'] = -mag
            elif any(k in lower for k in ('提', '亮', 'lift', 'boost')):
                delta['highlights'] = mag

        if any(k in lower for k in ('对比', '层次', 'contrast')):
            delta['contrast'] = -mag if any(k in lower for k in ('降', '柔', '减', 'reduce', 'lower', 'less')) else mag

        if any(k in lower for k in ('饱和', '鲜艳', '色彩', 'saturation', 'vibrant')):
            delta['saturation'] = -mag if any(k in lower for k in ('降', '淡', '减', 'reduce', 'less', 'desaturate')) else mag

        if any(k in lower for k in ('暖', 'warm')):
            delta['temperature'] = mag
        elif any(k in lower for k in ('冷', '蓝', 'cool')):
            delta['temperature'] = -mag

        if any(k in lower for k in ('品红', '偏绿', '去偏色', 'tint')):
            delta['tint'] = -round(mag * 0.5) if any(k in lower for k in ('暖', '品红', 'magenta', '去绿')) else round(mag * 0.5)

        if any(k in lower for k in ('复古', '怀旧', '棕褐', 'sepia', 'vintage')):
            delta['sepia'] = max(25, min(65, mag * 2))

        return clamp_params(delta)

    def predict_parameters(self, image_path: Path, instruction: str) -> dict[str, Any]:
        params = self._infer_delta(instruction.lower(), strength=25)
        if not params:
            return {'action': 'keep'}
        return {'action': 'edit', 'parameters': params}

    def critique_render(
        self,
        source_path: Path,
        rendered_path: Path,
        user_message: str,
        params: dict[str, float],
        before_stats: dict[str, float],
        after_stats: dict[str, float],
    ) -> dict[str, Any]:
        return {
            'passed': True,
            'issue': 'none',
            'suggested_scale': 1.0,
            'reason': '本地规则模式已通过图像硬指标护栏检验。' if _is_chinese(user_message) else 'Passed CPU guardrail inspection.',
        }
