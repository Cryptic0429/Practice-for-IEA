"""Multi-turn conversational Agent session routes for IEA Studio."""
from __future__ import annotations

import io
import threading
import uuid
import zipfile
from pathlib import Path

from flask import abort, jsonify, request, send_file, send_from_directory
from PIL import Image, ImageOps

from iea_agent.backend import RuleBasedCPUPlannerBackend
from iea_agent.guardrails import compute_statistics
from iea_agent.orchestrator import ImageEditingAgent
from iea_agent.state import SessionState, TOOLS
from . import common

session_lock = threading.Lock()
_agent_backend = None


def set_agent_backend(backend) -> None:
    global _agent_backend
    _agent_backend = backend


def get_agent() -> ImageEditingAgent:
    backend = _agent_backend or RuleBasedCPUPlannerBackend()
    return ImageEditingAgent(backend=backend)


def public_session(session: SessionState, backend_mode: str) -> dict:
    sid = session.session_id
    data = session.to_dict()
    data['source_url'] = f'/api/sessions/{sid}/files/{session.source_image}'
    history_out = []
    active_image_url = data['source_url']
    for item in data['history']:
        row = dict(item)
        row['image_url'] = f"/api/sessions/{sid}/files/{item['image_file']}"
        if item['turn'] == session.active_turn:
            active_image_url = row['image_url']
        history_out.append(row)
    data['history'] = history_out
    data['active_image_url'] = active_image_url
    data['backend_mode'] = backend_mode
    data['supported_tools'] = list(TOOLS)
    return data


def _load_session(session_id: str) -> tuple[Path, SessionState]:
    folder = common.session_path(session_id)
    return folder, SessionState.from_dict(common.read(folder / 'session.json'))


def _save_session(folder: Path, session: SessionState) -> None:
    common.write(folder / 'session.json', session.to_dict())


def register_session_routes(app) -> None:
    @app.get('/api/sessions')
    def list_sessions():
        common.SESSIONS.mkdir(parents=True, exist_ok=True)
        items = sorted(
            (common.read(p) for p in common.SESSIONS.glob('*/session.json')),
            key=lambda s: s.get('updated_at', s.get('created_at', '')),
            reverse=True,
        )
        return jsonify(sessions=[
            dict(
                session_id=s['session_id'],
                title=s['title'],
                active_turn=s.get('active_turn', 0),
                turns=max(0, len(s.get('history', [])) - 1),
                updated_at=s.get('updated_at', s.get('created_at')),
                sample_id=s.get('sample_id'),
                thumbnail=f"/api/sessions/{s['session_id']}/files/{s.get('source_image', 'source.png')}",
            )
            for s in items[:50]
        ])

    @app.post('/api/sessions')
    def create_session():
        common.SESSIONS.mkdir(parents=True, exist_ok=True)
        sample_id = request.form.get('sample_id')
        selected = common.samples().get(sample_id) if sample_id else None
        if selected:
            image_source = common.ROOT / selected['image']
            title = selected['title']
        elif 'image' in request.files:
            image_source = request.files['image'].stream
            title = Path(request.files['image'].filename or '我的照片').name
        else:
            return jsonify(error='请上传一张图片或从示例库选择照片。'), 400

        with Image.open(image_source) as im:
            if im.width * im.height > 40_000_000:
                return jsonify(error='图片不能超过4000万像素。'), 400
            image = ImageOps.exif_transpose(im).convert('RGB')
            sid = uuid.uuid4().hex
            folder = common.SESSIONS / sid
            folder.mkdir(parents=True, exist_ok=True)
            src_path = folder / 'source.png'
            image.save(src_path, format='PNG', compress_level=1)

        try:
            import numpy as np
            source_stats = compute_statistics(np.asarray(image))
        except Exception:
            source_stats = {}

        session = SessionState(
            session_id=sid,
            title=title[:120],
            source_image='source.png',
            source_sha256=common.sha(src_path),
            image_size=list(image.size),
            sample_id=sample_id if selected else None,
            training_use=False,
        )
        session.ensure_initial_snapshot(source_stats=source_stats)
        agent = get_agent()
        mode = agent.backend.mode_name
        intro = (
            f'已载入「{session.title}」（{image.width} × {image.height}）。'
            '您可以直接告诉我修图想法，例如“稍微提亮阴影，保留天空和高光细节”，之后还可以继续微调、退回一半或撤销。'
        )
        session.append_message('assistant', intro, action='welcome', turn=0, params={}, backend_mode=mode)
        _save_session(folder, session)
        return jsonify(public_session(session, mode)), 201

    @app.get('/api/sessions/<session_id>')
    def get_session(session_id: str):
        _, session = _load_session(session_id)
        return jsonify(public_session(session, get_agent().backend.mode_name))

    @app.post('/api/sessions/<session_id>/messages')
    def post_message(session_id: str):
        payload = request.get_json(silent=True) or {}
        message = str(payload.get('message', '')).strip()
        if not message or len(message) > 3000:
            return jsonify(error='请输入 1~3000 字的修图指令或对话内容。'), 400

        with session_lock:
            folder, session = _load_session(session_id)
            agent = get_agent()
            outcome = agent.handle_turn(session, folder, message)
            _save_session(folder, session)
            return jsonify(
                status=outcome['status'],
                session=public_session(session, agent.backend.mode_name),
            )

    @app.post('/api/sessions/<session_id>/checkout')
    def checkout_session(session_id: str):
        payload = request.get_json(silent=True) or {}
        with session_lock:
            folder, session = _load_session(session_id)
            agent = get_agent()
            if payload.get('action') == 'undo':
                target = max(0, session.active_turn - 1)
            elif payload.get('action') == 'reset':
                target = 0
            elif 'turn' in payload:
                target = int(payload['turn'])
            else:
                return jsonify(error='请指定目标轮次或操作类型。'), 400
            snap = session.checkout_turn(target)
            msg = '已还原到初始原图状态。' if target == 0 else f'已切换到第 {target} 轮状态（基于原图无损渲染）。'
            session.append_message(
                'assistant',
                msg,
                action='checkout',
                params=dict(snap['params']),
                turn=target,
                backend_mode=agent.backend.mode_name,
            )
            _save_session(folder, session)
            return jsonify(session=public_session(session, agent.backend.mode_name))

    @app.get('/api/sessions/<session_id>/files/<filename>')
    def session_file(session_id: str, filename: str):
        folder = common.session_path(session_id)
        if not (filename == 'source.png' or (filename.startswith('turn-') and filename.endswith('.png'))):
            abort(404)
        target = folder / filename
        if not target.is_file():
            abort(404)
        return send_from_directory(folder, filename, as_attachment=request.args.get('download') == '1')

    @app.get('/api/sessions/<session_id>/download')
    def download_session(session_id: str):
        folder = common.session_path(session_id)
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as z:
            for path in sorted(folder.iterdir()):
                if path.suffix in ('.json', '.png'):
                    z.write(path, path.name)
        stream.seek(0)
        return send_file(
            stream,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'IEA-session-{session_id[:8]}.zip',
        )
