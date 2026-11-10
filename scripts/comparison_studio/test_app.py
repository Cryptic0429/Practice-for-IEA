"""Contract checks for uploaded comparisons; no model inference and no GPU use."""
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from . import app as web, common


class ComparisonContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        jobs = root / 'jobs'
        jobs.mkdir()
        self.patches = [patch.object(web, 'RUN', root), patch.object(web, 'JOBS', jobs),
                        patch.object(common, 'JOBS', jobs), patch.object(web, 'runner', None)]
        for p in self.patches:
            p.start()
        self.client = web.app.test_client()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def submit(self, **extra):
        stream = io.BytesIO()
        Image.new('RGB', (48, 32), (80, 100, 120)).save(stream, format='PNG')
        stream.seek(0)
        payload = dict(zh='保持原样。', en='Keep unchanged.', image=(stream, 'example.png'))
        payload.update(extra)
        return self.client.post('/api/jobs', data=payload, headers={'X-IEA-Source': 'agent-smoke'})

    def test_four_models_two_literal_requests_and_isolation(self):
        response = self.submit()
        self.assertEqual(response.status_code, 202)
        job = response.get_json()
        self.assertEqual(len(job['results']), 8)
        self.assertEqual({(r['model'], r['language']) for r in job['results']},
                         {(m, l) for m in common.MODELS for l in ('zh', 'en')})
        self.assertEqual(job['instructions'], {'zh': '保持原样。', 'en': 'Keep unchanged.'})
        self.assertFalse(job['training_use'])
        self.assertEqual(job['source'], 'agent_smoke_test')
        with self.client.get(job['before']) as original:
            self.assertEqual(original.status_code, 200)
        self.assertEqual(self.client.get(f"/files/{job['id']}/r21-zh.png").status_code, 404)
        response = self.client.get(f"/api/jobs/{job['id']}/download")
        with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
            self.assertEqual(set(archive.namelist()), {'job.json', 'input.png'})

    def test_language_validation_and_cross_origin(self):
        self.assertEqual(self.submit(en='').status_code, 400)
        self.assertEqual(self.client.post('/api/jobs', headers={'Origin': 'https://unrelated.example'}).status_code, 403)
        self.assertEqual(self.client.get('/api/jobs/not-a-job').status_code, 400)

    def test_feedback_is_bound_to_job_and_invalid_votes_rejected(self):
        job = self.submit().get_json()
        url = f"/api/jobs/{job['id']}/feedback"
        self.assertEqual(self.client.post(url, json={'preferred': {'zh': 'invented'}}).status_code, 400)
        response = self.client.post(url, json={'ratings': {'r1-zh': 'uncertain'}, 'preferred': {'zh': 'none'},
                                              'comment': 'interface contract test'},
                                    headers={'X-IEA-Source': 'agent-smoke'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get(f"/api/jobs/{job['id']}").get_json()['feedback'], {})
        stored = self.client.get(f"/api/jobs/{job['id']}", headers={'X-IEA-Source': 'agent-smoke'}).get_json()['feedback']
        self.assertEqual(stored['source'], 'agent_smoke_test')
        self.assertFalse(stored['training_use'])
        self.assertEqual(stored['job_id'], job['id'])


if __name__ == '__main__':
    unittest.main()
