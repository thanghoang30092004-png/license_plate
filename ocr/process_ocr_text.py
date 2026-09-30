import re


class PlateFormatter:
    """
    Format text biển số về chuẩn Việt Nam, và correction các ký tự
    OCR dễ nhầm lẫn (O/0, B/8, S/5...).
    """

    CHAR_TO_DIGIT = {
        'O': '0', 'D': '0', 'Q': '0',
        'I': '1', 'L': '1', 'J': '1',
        'S': '5', 'B': '8',
        'G': '6', 'Z': '2', 'A': '4', 'T': '7'
    }

    # Chiều ngược lại: chữ của series bị OCR đọc thành số
    DIGIT_TO_CHAR = {
        '0': 'D', '1': 'L', '2': 'Z',
        '4': 'A', '5': 'S', '6': 'G', '8': 'B'
    }

    # (số chữ cái của series, series có kèm số hay không, số chữ số cuối).
    # Thứ tự quyết định layout nào thắng khi số lần correction bằng nhau.
    LAYOUTS = (
        (1, False, 5),   # 30A-123.45
        (1, True, 5),    # 29H1-234.56  (xe máy)
        (2, False, 5),   # 51LD-123.45  (series 2 chữ: LD, NG, KT...)
        (1, False, 4),   # 30A-1234
        (2, False, 4),   # 51AB-1234
        (1, True, 4)     # 29H1-1234
    )

    # Biển số VN không dùng I, J, O, Q, W cho series vì dễ lẫn với chữ số
    SERIES_LETTERS = "ABCDEFGHKLMNPRSTUVXYZ"

    FORMATTED_PATTERN = (
        r'^\d{2}[' + SERIES_LETTERS + r']{1,2}\d?\-(\d{4}|\d{3}\.\d{2})$'
    )

    def format(self, text):
        """
        Cleans and formats license plate text to standard Vietnamese format.
        Example: 51G10096 -> 51G-100.96
        Trả về text đã clean (chưa format) nếu không khớp layout biển số nào.
        """
        text = re.sub(r'[^A-Z0-9\-\.]', '', text.upper())

        if re.match(self.FORMATTED_PATTERN, text):
            return text

        raw = re.sub(r'[\-\.]', '', text)

        # Chọn layout cần ít lần sửa ký tự nhất -> ưu tiên cách đọc sát OCR nhất
        best = None
        best_edits = None
        for layout in self.LAYOUTS:
            parts, edits = self._coerce_to_layout(raw, layout)
            if parts is None:
                continue
            if best_edits is None or edits < best_edits:
                best, best_edits = parts, edits
                if edits == 0:
                    break

        if best is None:
            return text
        return self._build_formatted(*best)

    def _coerce_to_layout(self, raw, layout):
        """
        Thử đọc raw theo một layout, sửa ký tự nhầm ở từng vị trí.
        return: ((city_code, series, numbers), số lần sửa) hoặc (None, None) nếu không khớp.
        """
        n_letters, has_series_digit, n_numbers = layout
        series_len = n_letters + int(has_series_digit)
        if len(raw) != 2 + series_len + n_numbers:
            return None, None

        # Kiểu ký tự mong đợi ở từng vị trí
        slots = ['digit'] * 2 + ['alpha'] * n_letters
        if has_series_digit:
            slots.append('digit')
        slots += ['digit'] * n_numbers

        chars = list(raw)
        edits = 0
        for i, slot in enumerate(slots):
            char = chars[i]
            if slot == 'digit' and not char.isdigit():
                if char not in self.CHAR_TO_DIGIT:
                    return None, None
                chars[i] = self.CHAR_TO_DIGIT[char]
                edits += 1
            elif slot == 'alpha':
                if not char.isalpha():
                    if char not in self.DIGIT_TO_CHAR:
                        return None, None
                    chars[i] = self.DIGIT_TO_CHAR[char]
                    edits += 1
                if chars[i] not in self.SERIES_LETTERS:
                    return None, None

        city_code = "".join(chars[:2])
        series = "".join(chars[2:2 + series_len])
        numbers = "".join(chars[2 + series_len:])
        return (city_code, series, numbers), edits

    def _build_formatted(self, city_code, series, numbers):
        if len(numbers) == 5:
            numbers = f"{numbers[:3]}.{numbers[3:]}"
        return f"{city_code}{series}-{numbers}"

    def is_valid(self, text):
        """True nếu text là biển số đã format đúng chuẩn."""
        return re.match(self.FORMATTED_PATTERN, text) is not None
