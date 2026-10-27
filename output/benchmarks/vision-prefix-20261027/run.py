"""Repeat the original five-question street-image request against the local service."""
import base64
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import httpx

ROOT = Path('/home/AI/kev')
IMAGE = ROOT / 'output/imagegen/street-decision-20261026.png'
image_bytes = IMAGE.read_bytes()
payload = {
    'state': '附图是一张街景照片。根据实际可见内容回答。左右均以观看图片的人的视角为准。不要根据常识补充图片中看不到的信息。',
    'images': [base64.b64encode(image_bytes).decode('ascii')],
    'questions': {
        'people_count': {
            'type': 'choice',
            'instructions': '图片中能辨认出多少个人？包括部分被遮挡但能确认是人的目标。',
            'criteria': {'zero': '0人', 'one': '1人', 'two': '2人', 'three_four': '3至4人', 'five_plus': '5人及以上', 'unclear': '画面过于模糊，无法判断'},
        },
        'traffic_light': {
            'type': 'choice',
            'instructions': '图片中最明显的一组机动车交通信号灯，当前亮着什么颜色？',
            'criteria': {'red': '红灯', 'yellow': '黄灯', 'green': '绿灯', 'off': '可见信号灯，但没有灯亮', 'absent': '没有可见的机动车交通信号灯', 'unclear': '有信号灯，但颜色无法辨认'},
        },
        'spatial_relation': {
            'type': 'choice',
            'instructions': '找出画面中面积最大的行人和面积最大的汽车。比较两者外接框的水平中心，行人在汽车的哪一侧？',
            'criteria': {'left': '行人在汽车左侧', 'right': '行人在汽车右侧', 'aligned': '两者水平中心基本对齐', 'missing': '缺少行人或汽车，无法比较', 'unclear': '目标无法辨认'},
        },
        'person_on_crosswalk': {
            'type': 'noul',
            'instructions': '是否至少有一名行人的脚落在可见的斑马线区域内？仅站在斑马线旁边不算。没有斑马线或行人则回答否。',
        },
        'visible_text': {
            'type': 'noul',
            'instructions': '图片中是否出现清晰可辨的完整英文单词 STOP？只判断可见文字，不根据标志的形状或颜色推测。',
        },
    },
}
body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
results = []
with httpx.Client(base_url='http://127.0.0.1:8009', timeout=180, trust_env=False) as client:
    meta = client.get('/v1/models').raise_for_status().json()
    for index in range(6):
        start = time.perf_counter()
        response = client.post('/v1/systemone', content=body, headers={'Content-Type': 'application/json'})
        elapsed = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        result = response.json()
        results.append({'request': index + 1, 'client_elapsed_ms': round(elapsed, 2), 'response': result})
        print(f"request {index + 1}: server={result['latency_ms']} ms client={elapsed:.2f} ms", flush=True)

warm = [r['response']['latency_ms'] for r in results[1:]]
summary = {
    'historical_single_request_ms': 6077.4,
    'first_request_ms': results[0]['response']['latency_ms'],
    'warm_requests': len(warm),
    'warm_mean_ms': statistics.mean(warm),
    'warm_median_ms': statistics.median(warm),
    'warm_min_ms': min(warm),
    'warm_max_ms': max(warm),
    'historical_vs_warm_mean_ratio': 6077.4 / statistics.mean(warm),
    'warm_mean_reduction_percent': (1 - statistics.mean(warm) / 6077.4) * 100,
}
report = {
    'date': '2026-10-27',
    'image_path': str(IMAGE),
    'image_sha256': hashlib.sha256(image_bytes).hexdigest(),
    'payload_sha256': hashlib.sha256(body).hexdigest(),
    'questions': payload['questions'],
    'state': payload['state'],
    'server': meta,
    'base_snapshot': str(Path('/home/AI/models/Qwen3.5-0.8B-Base').resolve()),
    'checkpoint_snapshot': str(Path('/home/AI/models/kev-0.8b').resolve()),
    'vision_source_sha256': hashlib.sha256((ROOT / 'kev/vision.py').read_bytes()).hexdigest(),
    'load_environment': {k: os.environ.get(k) for k in ('KEV_DTYPE', 'KEV_MERGE', 'KEV_ATTN', 'KEV_TEMPERATURE')},
    'summary': summary,
    'requests': results,
    'comparison_limit': 'Historical baseline is one earlier observation; this is not a contemporaneous old/new benchmark.',
}
destination = Path(__file__).with_name('result.json')
destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
print('Saved:', destination)
