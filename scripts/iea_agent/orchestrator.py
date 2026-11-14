"""Multi-turn conversational editing orchestrator with non-destructive render and Critic loop."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
from .backend import AgentBackend
from .guardrails import compute_statistics, evaluate_cpu_guardrail
from .prompts import format_params_brief
from .state import SessionState, clamp_params, diff_params, merge_params, scale_delta, validate_delta


def default_render_fn(source_path: Path, params: dict[str, float], output_path: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Render directly from immutable source_rgb using the frozen v15 bounded-fixed editor."""
    from functional_v15.editor import edit, load_rgb, save
    source_rgb = load_rgb(source_path)
    before_stats = compute_statistics(source_rgb)
    rendered_rgb = edit(source_rgb, clamp_params(params), 'bounded-fixed')
    after_stats = compute_statistics(rendered_rgb)
    save(output_path, rendered_rgb)
    return before_stats, after_stats


class ImageEditingAgent:
    """Orchestrates planning, parameter decision, non-destructive rendering, and Critic retries."""

    def __init__(
        self,
        backend: AgentBackend,
        render_fn: Callable[[Path, dict[str, float], Path], tuple[dict[str, float], dict[str, float]]] = default_render_fn,
        max_retries: int = 2,
        attenuation: float = 0.6,
    ) -> None:
        self.backend = backend
        self.render_fn = render_fn
        self.max_retries = max_retries
        self.attenuation = attenuation

    def handle_turn(self, session: SessionState, session_dir: Path, user_message: str) -> dict[str, Any]:
        text = user_message.strip()
        if not text:
            raise ValueError('empty_message')

        source_path = session_dir / session.source_image
        session.ensure_initial_snapshot()
        session.append_message('user', text)

        plan = self.backend.plan_turn(session, text)
        intent = plan.get('intent', 'chat')
        reply = str(plan.get('reply') or '').strip()

        if intent == 'undo':
            steps = max(1, int(plan.get('steps_back') or 1))
            return self._handle_undo(session, steps, reply)

        if intent == 'reset':
            return self._handle_checkout(session, 0, reply or '已还原到初始原图。', action='reset')

        if intent in ('keep', 'unsupported', 'chat'):
            action = intent
            session.append_message(
                'assistant',
                reply or '当前未对图像进行修改。',
                action=action,
                params=dict(session.current_params),
                turn=session.active_turn,
                backend_mode=self.backend.mode_name,
            )
            return {'session': session, 'status': action}

        # Determine target parameter delta / candidate parameters relative to active turn.
        base_params = dict(session.current_params)
        rewritten = str(plan.get('rewritten_instruction') or text).strip()

        if intent == 'adjust_relative':
            scale_factor = plan.get('scale_factor')
            if scale_factor is not None:
                targets = plan.get('target_tools') or list(base_params.keys())
                target_params = dict(base_params)
                for key in targets:
                    if key in target_params:
                        val = round(float(target_params[key]) * float(scale_factor), 2)
                        if abs(val) < 1.0:
                            target_params.pop(key, None)
                        else:
                            target_params[key] = val
                target_params = clamp_params(target_params)
                delta = diff_params(base_params, target_params)
            else:
                delta = validate_delta(plan.get('delta_parameters') or {})
        else:
            decision = self.backend.predict_parameters(source_path, rewritten)
            if decision.get('action') == 'decline':
                reason = decision.get('reason', 'unreliable_quality')
                msg = (
                    '质量风险控制评估认为在当前画质条件下直接编辑容易产生严重噪点或失真，已保留当前画面不变。'
                    if reason == 'unreliable_quality' else
                    '该需求超出了当前支持的 9 类全局调色工具范围，已保留当前画面不变。'
                )
                session.append_message(
                    'assistant',
                    msg,
                    action='decline',
                    reason=reason,
                    params=dict(base_params),
                    turn=session.active_turn,
                    backend_mode=self.backend.mode_name,
                )
                return {'session': session, 'status': 'declined', 'reason': reason}
            if decision.get('action') == 'keep':
                msg = reply or '评估认为当前图像无需改动，已保持不变。'
                session.append_message('assistant', msg, action='keep', params=dict(base_params), turn=session.active_turn)
                return {'session': session, 'status': 'keep'}

            proposed = clamp_params(decision.get('parameters') or {})
            delta = diff_params(base_params, proposed) if base_params else proposed
            if not delta:
                delta = proposed

        if not delta:
            msg = '未能识别出有效的调色参数变化，您可以尝试更具体地描述想要的明暗或色彩效果。'
            session.append_message('assistant', msg, action='keep', params=dict(base_params), turn=session.active_turn)
            return {'session': session, 'status': 'unchanged'}

        return self._render_with_critic(
            session=session,
            session_dir=session_dir,
            source_path=source_path,
            user_message=text,
            rewritten=rewritten,
            base_params=base_params,
            initial_delta=delta,
            base_reply=reply,
        )

    def _handle_undo(self, session: SessionState, steps: int, reply: str) -> dict[str, Any]:
        target_turn = max(0, session.active_turn - steps)
        return self._handle_checkout(session, target_turn, reply or f'已撤销并回到第 {target_turn} 轮状态。', action='undo')

    def _handle_checkout(self, session: SessionState, turn_index: int, reply: str, action: str = 'checkout') -> dict[str, Any]:
        snap = session.checkout_turn(turn_index)
        session.append_message(
            'assistant',
            reply,
            action=action,
            params=dict(snap['params']),
            turn=snap['turn'],
            backend_mode=self.backend.mode_name,
        )
        return {'session': session, 'status': action, 'snapshot': snap}

    def _render_with_critic(
        self,
        *,
        session: SessionState,
        session_dir: Path,
        source_path: Path,
        user_message: str,
        rewritten: str,
        base_params: dict[str, float],
        initial_delta: dict[str, float],
        base_reply: str,
    ) -> dict[str, Any]:
        next_turn_num = (session.history[-1]['turn'] + 1) if session.history else 1
        output_name = f'turn-{next_turn_num}.png'
        output_path = session_dir / output_name

        delta = dict(initial_delta)
        last_issues: list[str] = []
        last_reason = ''
        retries_used = 0

        for attempt in range(self.max_retries + 1):
            candidate = merge_params(base_params, delta)
            if not candidate:
                break

            before_stats, after_stats = self.render_fn(source_path, candidate, output_path)
            if not session.source_stats:
                session.source_stats = dict(before_stats)
                if session.history:
                    session.history[0]['stats'] = dict(before_stats)

            guard = evaluate_cpu_guardrail(before_stats, after_stats)
            if not guard.passed:
                last_issues = list(guard.issues)
                last_reason = guard.summary
            else:
                vlm_critic = self.backend.critique_render(
                    source_path, output_path, user_message, candidate, before_stats, after_stats
                )
                if vlm_critic.get('passed', True):
                    critic_info = {
                        'passed': True,
                        'stage': 'cpu_and_vlm',
                        'retries': attempt,
                        'issues': [],
                        'summary': str(vlm_critic.get('reason') or guard.summary),
                        'guardrail': guard.to_dict(),
                    }
                    snapshot = session.commit_snapshot(
                        label=f'第 {next_turn_num} 轮',
                        action='edit',
                        params=candidate,
                        delta=diff_params(base_params, candidate),
                        image_file=output_name,
                        stats=after_stats,
                        critic=critic_info,
                        rewritten_instruction=rewritten,
                    )
                    retry_note = f'（经 {attempt} 次自检微调幅度）' if attempt > 0 else ''
                    final_reply = (
                        f"{base_reply or '已完成本轮调色。'}{retry_note}\n"
                        f"当前累计参数：{format_params_brief(candidate)}。"
                    )
                    session.append_message(
                        'assistant',
                        final_reply,
                        action='edit',
                        params=dict(candidate),
                        delta=dict(snapshot['delta']),
                        turn=snapshot['turn'],
                        critic=critic_info,
                        backend_mode=self.backend.mode_name,
                    )
                    return {'session': session, 'status': 'edited', 'snapshot': snapshot}

                last_issues = [str(vlm_critic.get('issue') or 'visual_critic_failed')]
                last_reason = str(vlm_critic.get('reason') or '视觉复检未通过')

            retries_used = attempt + 1
            if attempt < self.max_retries:
                delta = scale_delta(delta, self.attenuation)
                if not delta:
                    break

        output_path.unlink(missing_ok=True)
        rollback_msg = (
            f"尝试执行后未能通过画质自检（{last_reason}），"
            f"即使按 {self.attenuation} 倍衰减幅度重试后仍有风险，已自动回退到上一轮画面以保护画质。"
        )
        session.append_message(
            'assistant',
            rollback_msg,
            action='rollback',
            issues=last_issues,
            retries=retries_used,
            params=dict(base_params),
            turn=session.active_turn,
            backend_mode=self.backend.mode_name,
        )
        return {'session': session, 'status': 'rollback', 'issues': last_issues, 'reason': last_reason}
