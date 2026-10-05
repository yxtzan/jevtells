"""Read burned captions on their own frame timeline using CPU RapidOCR."""
from __future__ import annotations

from collections import Counter
import json
import re
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from ..render.reframe import detect_subtitles
from .asr import _parse_srt


def clean_text(text: str) -> str:
    return re.sub(r'(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])', '', text).strip()


def select_lines(rows: list, language: str, threshold: float) -> tuple[str, str, float]:
    """Group detections into rows and keep the speech language, preserving case."""
    lines: list[list] = []
    for box, text, confidence in sorted(rows, key=lambda r: np.asarray(r[0])[:, 1].mean()):
        if confidence < threshold:
            continue
        box = np.asarray(box)
        y = float(box[:, 1].mean())
        h = float(np.ptp(box[:, 1]))
        if not lines or abs(y - lines[-1][0]) > h * .6:
            lines.append([y, []])
        lines[-1][1].append((float(box[:, 0].min()), clean_text(text), float(confidence)))
    parsed = []
    for _, parts in lines:
        parts.sort()
        text = clean_text(' '.join(p[1] for p in parts))
        parsed.append((text, float(np.mean([p[2] for p in parts]))))
    chinese = lambda text: bool(re.search(r'[\u3400-\u9fff]', text))
    primary = [p for p in parsed if chinese(p[0]) == language.startswith('zh')]
    translation = [p[0] for p in parsed if chinese(p[0]) != language.startswith('zh')]
    return clean_text(' '.join(p[0] for p in primary)), clean_text(' '.join(translation)), min((p[1] for p in primary), default=0.)


def restore_word_spaces(image: np.ndarray, rows: list, engine: Any, settings: Mapping[str, Any]) -> list:
    result = []
    for box, text, confidence in rows:
        if re.search(r'[\u3400-\u9fff]', text):
            result.append([box,text,confidence]);continue
        xy = np.asarray(box)
        x0,y0 = np.maximum(0,np.floor(xy.min(axis=0)).astype(int))
        x1,y1 = np.ceil(xy.max(axis=0)).astype(int)
        crop = image[y0:y1,x0:x1]
        if not crop.size:
            continue
        hsv = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
        mask = (hsv[:,:,2]>=settings.get('text_value',180)) & ((hsv[:,:,1]<=settings.get('text_saturation',90)) | ((hsv[:,:,0]>=15)&(hsv[:,:,0]<=45)))
        yellow = (hsv[:,:,2]>=settings.get('word_text_value',100)) & (hsv[:,:,1]>settings.get('word_text_saturation',50)) & (hsv[:,:,0]>=15) & (hsv[:,:,0]<=45)
        if np.count_nonzero(yellow)>np.count_nonzero(mask)*float(settings.get('word_colour_ratio',.3)):
            mask = yellow
        dark_rows = (hsv[:,:,2] < settings.get('background_value',120)).mean(axis=1) >= float(settings.get('word_background_ratio',.5))
        if not np.count_nonzero(yellow) and dark_rows.any():
            mask &= dark_rows[:,None]
        columns = np.flatnonzero(mask.any(axis=0))
        if len(columns):
            gap = max(2, crop.shape[0]*float(settings.get('word_gap_height_ratio',.30)))
            starts = columns[np.r_[True,np.diff(columns)>gap]]
            ends = columns[np.r_[np.diff(columns)>gap,True]]
            words = []
            for a,b in zip(starts,ends):
                recognized,_ = engine(crop[:,max(0,a-2):b+3],use_det=False,use_cls=False)
                if not recognized or recognized[0][1]<settings.get('confidence',.80):
                    words=[];break
                words.append(recognized[0][0].strip())
            letters = lambda value: re.sub(r'[^A-Za-z0-9]','',value).lower()
            # Spacing may change only when every recognized letter still
            # matches the whole-line result. Never correct words with ASR.
            if len(words)>1 and letters(''.join(words)) == letters(text):
                compact = re.sub(r'\s+','',text)
                parts=[];cursor=0
                for word in words:
                    start=cursor;remaining=len(letters(word))
                    while cursor<len(compact) and remaining:
                        if compact[cursor].isalnum():remaining-=1
                        cursor+=1
                    while cursor<len(compact) and not compact[cursor].isalnum():cursor+=1
                    parts.append(compact[start:cursor])
                text=' '.join(parts)+compact[cursor:]
        result.append([box,text,confidence])
    return result


def caption_mask(frame: np.ndarray, settings: Mapping[str, Any]) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] >= settings.get('text_value', 180)) & ((hsv[:, :, 1] <= settings.get('text_saturation', 90)) | ((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 45)))).astype('uint8')
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    result = np.zeros_like(mask)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        pad = int(settings.get('component_padding', 3))
        surround = hsv[max(0,y-pad):min(len(hsv),y+h+pad),max(0,x-pad):min(frame.shape[1],x+w+pad),2]
        dark = float((surround < settings.get('background_value', 120)).mean())
        if settings.get('min_component_height', 8) <= h <= frame.shape[0] * .5 and area >= settings.get('min_component_area', 8) and w < frame.shape[1] * .15 and dark >= settings.get('background_ratio', .15):
            result[labels == i] = 1
    return result


def frame_spans(masks: list[np.ndarray], settings: Mapping[str, Any]) -> list[tuple[int, int]]:
    if not masks:
        return []
    boundaries = [0]
    reference = masks[0]
    for i, mask in enumerate(masks[1:], 1):
        union = np.count_nonzero(mask | reference)
        radius = int(settings.get('pixel_tolerance', 1))
        kernel = np.ones((2*radius+1,2*radius+1), dtype='uint8')
        change = (np.count_nonzero(mask & (1-cv2.dilate(reference,kernel))) + np.count_nonzero(reference & (1-cv2.dilate(mask,kernel)))) / max(1, union)
        if change > float(settings.get('change_ratio', .35)):
            boundaries.append(i)
            reference = mask
    boundaries.append(len(masks))
    return list(zip(boundaries, boundaries[1:]))


def timestamp(seconds: float) -> str:
    ms = round(seconds * 1000)
    return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'


def run(clip: Path, out: Path, language: str, config: Mapping[str, Any], *, force: bool = False, detection: Mapping[str, Any] | None = None) -> dict[str, Any]:
    settings = config.get('ocr', {})
    destination = out / 'subtitles_ocr.srt'
    meta_path = out / 'subtitles_ocr.json'
    signature = {'settings': dict(settings), 'language': language, 'version': 6}
    if destination.exists() and meta_path.exists() and not force and json.loads(meta_path.read_text()).get('signature') == signature:
        meta = json.loads(meta_path.read_text())
    else:
        from rapidocr_onnxruntime import RapidOCR
        engine = RapidOCR(intra_op_num_threads=int(settings.get('threads', 2)), inter_op_num_threads=1)
        detection = detection or detect_subtitles(clip, config.get('render', {}).get('reframe', {}))
        capture = cv2.VideoCapture(str(clip))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        top = int(detection['strip_y'])
        masks = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            masks.append(caption_mask(frame[top:], settings))
        cues, rejected = [], []
        for start, end in frame_spans(masks, settings):
            if (end-start)/fps < float(settings.get('min_seconds', .12)):
                rejected.append({'t0':start/fps, 't1':end/fps, 'reason':'short span'})
                continue
            samples = []
            for index in sorted(set(round(start+(end-start-1)*fraction) for fraction in settings.get('sample_fractions', [.35, .5, .65]))):
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError('OCR sample decoding failed')
                crop = frame[max(0,top-int(settings.get('crop_padding', 6))):]
                rows, _ = engine(crop, unclip_ratio=float(settings.get("unclip_ratio", 2.0)))
                if not language.startswith('zh'):
                    rows = restore_word_spaces(crop,rows or [],engine,settings)
                text, translation, confidence = select_lines(rows or [], language, float(settings.get('confidence', .80)))
                samples.append((text, translation, confidence))
            counts = Counter(item[0] for item in samples if item[0])
            text, votes = counts.most_common(1)[0] if counts else ('', 0)
            selected = [s for s in samples if s[0] == text]
            confidence = min((s[2] for s in selected), default=0.)
            if len(re.findall(r'[A-Za-z0-9\u3400-\u9fff]', text)) < int(settings.get('min_characters_zh', 1) if language.startswith('zh') else settings.get('min_characters', 2)) or votes < min(int(settings.get('min_votes', 2)), len(samples)) or confidence < float(settings.get('confidence', .80)):
                rejected.append({'t0':start/fps,'t1':end/fps,'samples':samples,'reason':'low confidence or disagreement'})
                continue
            translation = Counter(s[1] for s in selected).most_common(1)[0][0]
            if cues and cues[-1]['text'] == text and start/fps-cues[-1]['t1'] <= float(settings.get('merge_gap', .15)):
                cues[-1]['t1'] = end/fps
            else:
                cues.append({'t0':start/fps,'t1':end/fps,'text':text,'subtitle_translation':translation,'confidence':confidence})
        capture.release()
        destination.write_text('\n\n'.join(f"{i}\n{timestamp(c['t0'])} --> {timestamp(c['t1'])}\n{c['text']}" for i,c in enumerate(cues,1))+'\n', encoding='utf-8')
        meta = {'signature':signature,'detection':dict(detection),'cues':cues,'rejected':rejected}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    transcript = _parse_srt(destination)
    transcript.update(language=language, subtitle_source='ocr')
    for segment, cue in zip(transcript['segments'], meta['cues']):
        segment['subtitle_translation'] = cue['subtitle_translation']
    return transcript


def resolve_source(requested: str, srt: str | None, subtitles: str, detected: bool) -> str:
    if requested == 'auto':
        return 'srt' if srt else 'ocr' if subtitles == 'off' and detected else 'asr'
    if requested == 'srt' and not srt:
        raise ValueError('--subtitle-source srt requires --srt')
    return requested
