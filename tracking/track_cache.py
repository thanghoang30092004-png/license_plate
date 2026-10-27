class TrackCache:
    """
    Quản lý cache biển số theo track_id, quyết định khi nào cần OCR lại,
    khi nào dùng kết quả cache — tránh OCR mọi frame cho mọi track.
    """

    def __init__(self, min_frames_to_confirm=3,
                 recheck_interval_found=60,
                 recheck_interval_missing=15,
                 miss_limit=5,
                 absent_grace=30):
        self.min_frames_to_confirm = min_frames_to_confirm
        self.recheck_interval_found = recheck_interval_found
        self.recheck_interval_missing = recheck_interval_missing
        self.miss_limit = miss_limit
        # DeepSort còn giữ track khoảng max_age frame, nên đừng xóa cache ngay frame vắng.
        self.absent_grace = absent_grace

        # track_id -> {
        #   'candidates': [text1, text2, ...],  # đang vote
        #   'confirmed_text': str hoặc None,
        #   'last_checked_frame': int,
        #   'not_found_count': int,
        #   'absent_frames': int
        # }
        self._cache = {}

    def _blank_entry(self, current_frame):
        return {
            'candidates': [],
            'confirmed_text': None,
            'last_checked_frame': current_frame,
            'not_found_count': 0,
            'absent_frames': 0,
            'snapshot_saved': False,  # đã lưu ảnh snapshot track_id này chưa
        }


    def should_save_snapshot(self, track_id):
        """Quyết định track này có cần lưu ảnh snapshot (ảnh crop biển số) không."""
        entry = self._cache.get(track_id)
        if entry is None:
            return False 
        return entry['confirmed_text'] is not None and not entry['snapshot_saved']


        
    def should_run_ocr(self, track_id, current_frame):
        """Quyết định track này có cần chạy OCR ở frame hiện tại không."""
        if track_id not in self._cache:
            return True  # track mới, luôn cần OCR

        entry = self._cache[track_id]
        frames_since_check = current_frame - entry['last_checked_frame']
        not_found = entry['not_found_count']

        if not_found >= self.miss_limit:
            return frames_since_check >= self.recheck_interval_missing

        if entry['confirmed_text'] is not None:
            interval = self.recheck_interval_found if not_found == 0 else self.recheck_interval_missing
            return frames_since_check >= interval

        return True  # đang vote, cần check tiếp mỗi frame

    def _read_text(self, ocr_result):
        """Nhận dict đã qua PlateOCR. valid=False hoặc text rỗng thì bỏ."""
        if not isinstance(ocr_result, dict) or not ocr_result.get('valid'):
            return None
        text = str(ocr_result.get('text') or '').strip()
        return text or None

    def _mark_missing(self, entry):
        entry['not_found_count'] += 1
        if entry['not_found_count'] >= self.miss_limit:
            entry['candidates'].clear()
            entry['confirmed_text'] = None

    def _try_confirm(self, entry):
        if len(entry['candidates']) < self.min_frames_to_confirm:
            return
        counts = {}
        for text in entry['candidates']:
            counts[text] = counts.get(text, 0) + 1
        winner, votes = max(counts.items(), key=lambda item: item[1])
        others = len(entry['candidates']) - votes
        if votes > others:
            entry['confirmed_text'] = winner
            entry['candidates'] = [winner]
            return
        # Cửa sổ quá dài mà không có đa số: bỏ phiếu cũ, vote lại từ đầu.
        if len(entry['candidates']) >= self.min_frames_to_confirm * 2:
            entry['candidates'].clear()

    def update(self, track_id, ocr_result, current_frame):
        """
        ocr_result: dict {'text', 'confidence', 'valid'} hoặc None nếu không đọc được.
        Dict thiếu valid hoặc text rỗng cũng tính là không đọc được.
        Text đã được PlateOCR format trước khi truyền vào.
        """
        entry = self._cache.get(track_id)
        if entry is None:
            entry = self._blank_entry(current_frame)
            self._cache[track_id] = entry

        entry['last_checked_frame'] = current_frame
        entry['absent_frames'] = 0
        text = self._read_text(ocr_result)

        if text is None:
            self._mark_missing(entry)
            return

        entry['not_found_count'] = 0

        if entry['confirmed_text'] is not None:
            # Đã chốt dạng biển Việt Nam. OCR lại không được ghi đè bằng một chuỗi khác.
            return

        entry['candidates'].append(text)
        self._try_confirm(entry)


    def mark_snapshot_saved(self, track_id):
        """Đánh dấu track này đã lưu ảnh snapshot (ảnh crop biển số) rồi."""
        entry = self._cache.get(track_id)
        if entry is not None:
            entry['snapshot_saved'] = True



    def get_text(self, track_id):
        """Trả về text đã confirm (hoặc None nếu chưa có)."""
        entry = self._cache.get(track_id)
        if entry is None:
            return None
        return entry['confirmed_text']

    def retain_tracks(self, retained_track_ids):
        """ByteTrack owns expiry: keep Tracked/Lost; drop IDs it has removed."""
        retained = set(retained_track_ids)
        for track_id in list(self._cache):
            if track_id not in retained:
                del self._cache[track_id]

    def cleanup(self, active_track_ids):
        """Xóa cache của track vắng mặt đủ absent_grace frame."""
        active = set(active_track_ids)
        stale_ids = []
        for track_id, entry in self._cache.items():
            if track_id in active:
                entry['absent_frames'] = 0
                continue
            entry['absent_frames'] += 1
            if entry['absent_frames'] >= self.absent_grace:
                stale_ids.append(track_id)
        for track_id in stale_ids:
            del self._cache[track_id]
