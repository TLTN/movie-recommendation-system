import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class RawDataInspector:
    """Kiểm tra dữ liệu THÔ"""

    def __init__(self, data_path: str = "../../data/raw/ml-100k"):
        self.data_path = Path(data_path) if isinstance(data_path, str) else data_path
        print("=" * 80)
        print("MOVIELENS 100K - RAW DATA INSPECTOR")
        print("=" * 80)

    def print_separator(self, title="", char="=", length=80):
        """đường phân cách"""
        if title:
            print("\n" + char * length)
            print(title.center(length))
            print(char * length)
        else:
            print(char * length)

    def list_all_files(self):
        """Liệt kê tất cả files trong thư mục"""
        self.print_separator("0. LIỆT KÊ TẤT CẢ FILES")

        print("\n=====CÁC FILE TRONG THỦ MỤC:")
        if not self.data_path.exists():
            print(f"❌ Thư mục không tồn tại: {self.data_path}")
            return

        all_files = sorted(self.data_path.glob('*')) # lấy file, sort

        print(f"\n   Thư mục: {self.data_path}")
        print(f"   Tổng số files: {len(all_files)}\n")

        data_files = {
            'u.data': 'Ratings data (100k ratings)',
            'u.item': 'Movies data (1682 movies)',
            'u.user': 'Users data (943 users)',
            'u.info': 'Dataset information',
            'u.genre': 'Genre list (19 genres)',
            'u.occupation': 'Occupation list (21 occupations)'
        }

        print("=====MAIN DATA FILES:")
        for filename, description in data_files.items():
            filepath = self.data_path / filename
            if filepath.exists():
                size = filepath.stat().st_size / 1024
                print(f"      ✓ {filename:15} - {description:35} ({size:>8.2f} KB)")
            else:
                print(f"      ✗ {filename:15} - {description:35} (NOT FOUND)")

        print("\n=====OTHER FILES:")
        other_files = [f for f in all_files if f.name not in data_files.keys()]
        if other_files:
            for f in other_files:
                size = f.stat().st_size / 1024
                file_type = "DIR" if f.is_dir() else "FILE"
                print(f"      • {f.name:30} ({file_type}, {size:.2f} KB)")
        else:
            print("      (Không có file khác)")

    def inspect_u_data(self):
        """Kiểm tra file u.data (RATINGS)"""
        self.print_separator("1. FILE THÔ: u.data (RATINGS)")

        filepath = self.data_path / 'u.data'
        if not filepath.exists():
            print(f"❌ File không tồn tại: {filepath}")
            return

        print("\n=====THÔNG TIN FILE:")
        print(f"   • Tên file: u.data")
        print(f"   • Đường dẫn: {filepath}")
        print(f"   • Kích thước: {filepath.stat().st_size / 1024:.2f} KB ({filepath.stat().st_size:,} bytes)")

        print("\n=====ĐẶC TẢ FORMAT:")
        print("   • Định dạng: Tab-separated values (TSV)")
        print("   • Encoding: UTF-8")
        print("   • Delimiter: TAB (\\t)")
        print("   • Số cột: 4")
        print("   • Cột 1: userId (int)")
        print("   • Cột 2: movieId (int)")
        print("   • Cột 3: rating (int, 1-5)")
        print("   • Cột 4: timestamp (int, Unix timestamp)")

        print("\n=====10 dòng đầu:")
        print("   " + "─" * 75)
        print(f"   │ {'STT':<4} │ {'userId':<8} │ {'movieId':<8} │ {'rating':<8} │ {'timestamp':<12} │")
        print("   " + "─" * 75)

        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                parts = line.strip().split('\t')
                if len(parts) == 4:
                    print(f"   │ {i:<4} │ {parts[0]:<8} │ {parts[1]:<8} │ {parts[2]:<8} │ {parts[3]:<12} │")
                if i >= 10:
                    break
        print("   " + "─" * 75)

        print("\n=====10 dòng cuối:")
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_lines = len(lines)

            print("   " + "─" * 75)
            print(f"   │ {'STT':<6} │ {'userId':<8} │ {'movieId':<8} │ {'rating':<8} │ {'timestamp':<12} │")
            print("   " + "─" * 75)

            for i, line in enumerate(lines[-10:], total_lines-9):
                parts = line.strip().split('\t')
                if len(parts) == 4:
                    print(f"   │ {i:<6} │ {parts[0]:<8} │ {parts[1]:<8} │ {parts[2]:<8} │ {parts[3]:<12} │")
            print("   " + "─" * 75)

        print("\n=====PHÂN TÍCH CẤU TRÚC:")
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline().rstrip()
            parts = first_line.split('\t')

            print(f"   • Dòng đầu tiên (raw): {first_line}")
            print(f"   • Số cột thực tế: {len(parts)}")
            print(f"   • Delimiter phát hiện: TAB")
            print(f"\n   • Phân tích từng cột:")
            col_names = ['userId', 'movieId', 'rating', 'timestamp']
            col_types = ['int', 'int', 'int (1-5)', 'int (Unix timestamp)']
            for i, (name, value, dtype) in enumerate(zip(col_names, parts, col_types)):
                print(f"     Cột {i+1} - {name:10}: '{value}' (type: {dtype})")

        print("\n=====THỐNG KÊ FILE:")
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            print(f"   • Tổng số dòng: {len(lines):,}")

            # Check empty lines
            empty_lines = sum(1 for line in lines if line.strip() == '')
            print(f"   • Dòng trống: {empty_lines}")

            # Check column count consistency
            col_counts = {}
            for i, line in enumerate(lines, 1):
                parts = line.strip().split('\t')
                col_count = len(parts)
                if col_count not in col_counts:
                    col_counts[col_count] = 0
                col_counts[col_count] += 1

            print(f"   • Phân bố số cột:")
            for col_count, count in sorted(col_counts.items()):
                print(f"     - {col_count} cột: {count:,} dòng")

            if len(col_counts) > 1:
                print(f"   ⚠️  CẢNH BÁO: Số cột không đồng nhất!")

        print("\n✓ HOÀN TẤT KIỂM TRA u.data")

    def inspect_u_item(self):
        """Kiểm tra file u.item (MOVIES)"""
        self.print_separator("2. FILE THÔ: u.item (MOVIES)")

        filepath = self.data_path / 'u.item'
        if not filepath.exists():
            print(f"❌ File không tồn tại: {filepath}")
            return

        print("\n=====THÔNG TIN FILE:")
        print(f"   • Tên file: u.item")
        print(f"   • Đường dẫn: {filepath}")
        print(f"   • Kích thước: {filepath.stat().st_size / 1024:.2f} KB ({filepath.stat().st_size:,} bytes)")

        print("\n=====ĐẶC TẢ FORMAT:")
        print("   • Định dạng: Pipe-separated values (PSV)")
        print("   • Encoding: ISO-8859-1 (Latin-1)")
        print("   • Delimiter: PIPE (|)")
        print("   • Số cột: 24")
        print("   • Cột 1-5: movieId, title, release_date, video_release_date, IMDb_URL")
        print("   • Cột 6-24: 19 binary genre columns (0 hoặc 1)")

        print("\n=====5 dòng đầu:")
        print("   " + "─" * 115)
        print(f"   │ {'#':<3} │ {'movieId':<7} │ {'title':<40} │ {'release_date':<12} │ {'genres (first 10)':<30} │")
        print("   " + "─" * 115)

        with open(filepath, 'r', encoding='latin-1') as f:
            for i, line in enumerate(f, 1):
                parts = line.strip().split('|')
                if len(parts) >= 24:
                    movie_id = parts[0]
                    title = parts[1][:38] + '..' if len(parts[1]) > 40 else parts[1]
                    release_date = parts[2]

                    # Get genres
                    genre_names = ['Unk', 'Act', 'Adv', 'Ani', 'Chi', 'Com', 'Cri', 'Doc', 'Dra', 'Fan',
                                 'FNr', 'Hor', 'Mus', 'Mys', 'Rom', 'ScF', 'Thr', 'War', 'Wes']
                    active_genres = [genre_names[j] for j, val in enumerate(parts[5:24]) if val == '1']
                    genres_str = ','.join(active_genres[:5]) if active_genres else 'None'

                    print(f"   │ {i:<3} │ {movie_id:<7} │ {title:<40} │ {release_date:<12} │ {genres_str:<30} │")
                if i >= 5:
                    break
        print("   " + "─" * 115)

        print("\n=====PHÂN TÍCH CẤU TRÚC (Dòng đầu tiên):")
        with open(filepath, 'r', encoding='latin-1') as f:
            first_line = f.readline().rstrip()
            parts = first_line.split('|')

            print(f"   • Số cột thực tế: {len(parts)}")
            print(f"   • Delimiter: PIPE (|)")
            print(f"\n   • Các cột chính:")
            print(f"     Cột 1  (movieId)         : '{parts[0]}'")
            print(f"     Cột 2  (title)           : '{parts[1]}'")
            print(f"     Cột 3  (release_date)    : '{parts[2]}'")
            print(f"     Cột 4  (video_release)   : '{parts[3]}' (empty OK)")
            print(f"     Cột 5  (IMDb_URL)        : '{parts[4][:50]}...'")
            print(f"     Cột 6-24 (genres)        : {parts[5:]} (19 binary: 0/1)")

        print("\n=====19 thể loại (6-24):")
        genres = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

        print("   " + "─" * 70)
        print(f"   │ {'Cột':<6} │ {'Genre':<20} │ {'Cột':<6} │ {'Genre':<20} │")
        print("   " + "─" * 70)
        for i in range(0, len(genres), 2):
            col1 = i + 6
            genre1 = genres[i]
            if i + 1 < len(genres):
                col2 = i + 7
                genre2 = genres[i + 1]
                print(f"   │ {col1:<6} │ {genre1:<20} │ {col2:<6} │ {genre2:<20} │")
            else:
                print(f"   │ {col1:<6} │ {genre1:<20} │ {'':6} │ {'':20} │")
        print("   " + "─" * 70)

        print("\n=====THỐNG KÊ FILE:")
        with open(filepath, 'r', encoding='latin-1') as f:
            lines = f.readlines()

            print(f"   • Tổng số dòng: {len(lines):,}")

            empty_lines = sum(1 for line in lines if line.strip() == '')
            print(f"   • Dòng trống: {empty_lines}")

            col_counts = {}
            for i, line in enumerate(lines, 1):
                parts = line.strip().split('|')
                col_count = len(parts)
                if col_count not in col_counts:
                    col_counts[col_count] = 0
                col_counts[col_count] += 1

            print(f"   • Phân bố số cột:")
            for col_count, count in sorted(col_counts.items()):
                print(f"     - {col_count} cột: {count:,} dòng")

        print("\n✓ HOÀN TẤT KIỂM TRA u.item")

    def inspect_u_user(self):
        """Kiểm tra file u.user (USERS)"""
        self.print_separator("3. FILE THÔ: u.user (USERS)")

        filepath = self.data_path / 'u.user'
        if not filepath.exists():
            print(f"❌ File không tồn tại: {filepath}")
            return

        print("\n=====THÔNG TIN FILE:")
        print(f"   • Tên file: u.user")
        print(f"   • Đường dẫn: {filepath}")
        print(f"   • Kích thước: {filepath.stat().st_size / 1024:.2f} KB ({filepath.stat().st_size:,} bytes)")

        print("\n=====ĐẶC TẢ FORMAT:")
        print("   • Định dạng: Pipe-separated values (PSV)")
        print("   • Encoding: UTF-8")
        print("   • Delimiter: PIPE (|)")
        print("   • Số cột: 5")
        print("   • Cột 1: userId (int)")
        print("   • Cột 2: age (int)")
        print("   • Cột 3: gender (char: M/F)")
        print("   • Cột 4: occupation (string)")
        print("   • Cột 5: zip_code (string)")

        print("\n=====10 dòng đầu:")
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                print(f"   {i:3}│ {line.rstrip()}")
                if i >= 10:
                    break

        print("\n=====PHÂN TÍCH CẤU TRÚC (Dòng đầu tiên):")
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline().rstrip()
            parts = first_line.split('|')

            print(f"   • Dòng đầu tiên: {first_line}")
            print(f"   • Số cột thực tế: {len(parts)}")
            print(f"   • Delimiter: PIPE (|)")
            print(f"\n   • Phân tích từng cột:")
            col_names = ['userId', 'age', 'gender', 'occupation', 'zip_code']
            for i, (name, value) in enumerate(zip(col_names, parts), 1):
                print(f"     Cột {i} ({name:12}): '{value}'")

        print("\n=====THỐNG KÊ FILE:")
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            print(f"   • Tổng số dòng: {len(lines):,}")

            empty_lines = sum(1 for line in lines if line.strip() == '')
            print(f"   • Dòng trống: {empty_lines}")

            col_counts = {}
            for i, line in enumerate(lines, 1):
                parts = line.strip().split('|')
                col_count = len(parts)
                if col_count not in col_counts:
                    col_counts[col_count] = 0
                col_counts[col_count] += 1

            print(f"   • Phân bố số cột:")
            for col_count, count in sorted(col_counts.items()):
                print(f"     - {col_count} cột: {count:,} dòng")

        print("\n✓ HOÀN TẤT KIỂM TRA u.user")

    def inspect_u_genre(self):
        """Kiểm tra file u.genre"""
        self.print_separator("4. FILE THÔ: u.genre (GENRE LIST)")

        filepath = self.data_path / 'u.genre'
        if not filepath.exists():
            print(f"❌ File không tồn tại: {filepath}")
            return

        print("\n=====THÔNG TIN FILE:")
        print(f"   • Tên file: u.genre")
        print(f"   • Kích thước: {filepath.stat().st_size / 1024:.2f} KB")

        print("\n=====ĐẶC TẢ:")
        print("   • Format: genre_name|genre_id")
        print("   • Delimiter: PIPE (|)")
        print("   • Encoding: UTF-8")

        print("\n=====FULL:")
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"   Tổng số genres: {len(lines)}\n")

            print("   " + "─" * 50)
            print(f"   │ {'ID':<5} │ {'Genre Name':<40} │")
            print("   " + "─" * 50)

            for line in lines:
                parts = line.strip().split('|')
                if len(parts) == 2:
                    genre_name = parts[0]
                    genre_id = parts[1]
                    print(f"   │ {genre_id:<5} │ {genre_name:<40} │")
            print("   " + "─" * 50)

        print("\n✓ HOÀN TẤT KIỂM TRA u.genre")

    def inspect_u_occupation(self):
        """Kiểm tra file u.occupation"""
        self.print_separator("5. FILE THÔ: u.occupation")

        filepath = self.data_path / 'u.occupation'
        if not filepath.exists():
            print(f"❌ File không tồn tại: {filepath}")
            return

        print("\n=====THÔNG TIN FILE:")
        print(f"   • Tên file: u.occupation")
        print(f"   • Kích thước: {filepath.stat().st_size / 1024:.2f} KB")

        print("\n=====ĐẶC TẢ:")
        print("   • Format: occupation_name (one per line)")
        print("   • Encoding: UTF-8")

        print("\n=====FULL:")
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"   Tổng số occupations: {len(lines)}\n")

            print("   " + "─" * 50)
            print(f"   │ {'#':<5} │ {'Occupation Name':<40} │")
            print("   " + "─" * 50)

            for i, line in enumerate(lines, 1):
                occupation = line.strip()
                print(f"   │ {i:<5} │ {occupation:<40} │")
            print("   " + "─" * 50)

        print("\n✓ HOÀN TẤT KIỂM TRA u.occupation")

    def inspect_u_info(self):
        """Kiểm tra file u.info"""
        self.print_separator("6. FILE THÔ: u.info (DATASET INFO)")

        filepath = self.data_path / 'u.info'
        if not filepath.exists():
            print(f"❌ File không tồn tại: {filepath}")
            return

        print("\n=====THÔNG TIN FILE:")
        print(f"   • Tên file: u.info")
        print(f"   • Kích thước: {filepath.stat().st_size / 1024:.2f} KB")

        print("\n=====FULL:")
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            print(content)

        print("\n✓ HOÀN TẤT KIỂM TRA u.info")

    def check_data_quality(self):
        """Kiểm tra chất lượng dữ liệu"""
        self.print_separator("7. KIỂM TRA CHẤT LƯỢNG DỮ LIỆU")

        issues = []

        # Check u.data
        print("\n=====RATINGS (u.data):")
        filepath = self.data_path / 'u.data'
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()

                # Check empty lines
                empty_count = sum(1 for line in lines if line.strip() == '')
                print(f"   • Tổng dòng: {len(lines):,}")
                print(f"   • Dòng trống: {empty_count}")
                if empty_count > 0:
                    issues.append(f"u.data: {empty_count} dòng trống")

                # Check column count
                col_counts = {}
                for line in lines:
                    col_count = len(line.strip().split('\t'))
                    col_counts[col_count] = col_counts.get(col_count, 0) + 1

                print(f"   • Số cột: {list(col_counts.keys())}")
                if len(col_counts) > 1:
                    print(f"   ⚠️  Số cột không đồng nhất!")
                    issues.append(f"u.data: Số cột không đồng nhất - {col_counts}")

                # Check missing values
                missing_count = 0
                for line in lines:
                    parts = line.strip().split('\t')
                    if len(parts) == 4:
                        if any(p.strip() == '' for p in parts):
                            missing_count += 1

                print(f"   • Dòng có giá trị thiếu: {missing_count}")
                if missing_count > 0:
                    issues.append(f"u.data: {missing_count} dòng có giá trị thiếu")

                # Check data types (first 5 lines)
                print(f"   • Kiểm tra kiểu dữ liệu (5 dòng đầu):")
                for i, line in enumerate(lines[:5], 1):
                    parts = line.strip().split('\t')
                    if len(parts) == 4:
                        try:
                            int(parts[0])  # userId
                            int(parts[1])  # movieId
                            int(parts[2])  # rating
                            int(parts[3])  # timestamp
                            print(f"     Dòng {i}: ✓ OK")
                        except ValueError:
                            print(f"     Dòng {i}: ✗ LỖI kiểu dữ liệu")
                            issues.append(f"u.data dòng {i}: Lỗi kiểu dữ liệu")

        # Check u.item
        print("\n=====MOVIES (u.item):")
        filepath = self.data_path / 'u.item'
        if filepath.exists():
            with open(filepath, 'r', encoding='latin-1') as f:
                lines = f.readlines()

                empty_count = sum(1 for line in lines if line.strip() == '')
                print(f"   • Tổng dòng: {len(lines):,}")
                print(f"   • Dòng trống: {empty_count}")

                col_counts = {}
                for line in lines:
                    col_count = len(line.strip().split('|'))
                    col_counts[col_count] = col_counts.get(col_count, 0) + 1

                print(f"   • Số cột: {list(col_counts.keys())}")
                if len(col_counts) > 1:
                    print(f"   ⚠️  Số cột không đồng nhất!")
                    issues.append(f"u.item: Số cột không đồng nhất - {col_counts}")

                # Check missing in important columns
                missing_title = 0
                missing_date = 0
                for line in lines:
                    parts = line.strip().split('|')
                    if len(parts) >= 24:
                        if parts[1].strip() == '':
                            missing_title += 1
                        if parts[2].strip() == '':
                            missing_date += 1

                print(f"   • Title thiếu: {missing_title}")
                print(f"   • Release date thiếu: {missing_date}")

        # Check u.user
        print("\n=====USERS (u.user):")
        filepath = self.data_path / 'u.user'
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()

                empty_count = sum(1 for line in lines if line.strip() == '')
                print(f"   • Tổng dòng: {len(lines):,}")
                print(f"   • Dòng trống: {empty_count}")

                col_counts = {}
                for line in lines:
                    col_count = len(line.strip().split('|'))
                    col_counts[col_count] = col_counts.get(col_count, 0) + 1

                print(f"   • Số cột: {list(col_counts.keys())}")

                # Check missing values
                missing_count = 0
                for line in lines:
                    parts = line.strip().split('|')
                    if len(parts) == 5:
                        if any(p.strip() == '' for p in parts):
                            missing_count += 1

                print(f"   • Dòng có giá trị thiếu: {missing_count}")

        # Summary
        print("\n" + "=" * 80)
        if issues:
            print(f"⚠️  PHÁT HIỆN {len(issues)} VẤN ĐỀ:")
            for issue in issues:
                print(f"   • {issue}")
        else:
            print("KHÔNG PHÁT HIỆN VẤN ĐỀ NGHIÊM TRỌNG")

        print("\n✓ HOÀN TẤT KIỂM TRA CHẤT LƯỢNG")

    def run_all_inspections(self):
        """Chạy tất cả các kiểm tra"""
        self.list_all_files()
        input("\n[Nhấn Enter để tiếp tục...]")

        self.inspect_u_data()
        input("\n[Nhấn Enter để tiếp tục...]")

        self.inspect_u_item()
        input("\n[Nhấn Enter để tiếp tục...]")

        self.inspect_u_user()
        input("\n[Nhấn Enter để tiếp tục...]")

        self.inspect_u_genre()
        input("\n[Nhấn Enter để tiếp tục...]")

        self.inspect_u_occupation()
        input("\n[Nhấn Enter để tiếp tục...]")

        self.inspect_u_info()
        input("\n[Nhấn Enter để tiếp tục...]")

        self.check_data_quality()

        self.print_separator("HOÀN TẤT TẤT CẢ KIỂM TRA", "=", 80)

    def interactive_menu(self):
        """Menu tương tác"""
        while True:
            print("\n" + "=" * 80)
            print("MENU KIỂM TRA DỮ LIỆU")
            print("=" * 80)
            print("\n=====Chọn file để kiểm tra:")
            print("  0. Thoát")
            print("  1. Liệt kê tất cả files")
            print("  2. u.data (Ratings) - 100,000 dòng")
            print("  3. u.item (Movies) - 1,682 dòng")
            print("  4. u.user (Users) - 943 dòng")
            print("  5. u.genre (Genres) - 19 dòng")
            print("  6. u.occupation (Occupations) - 21 dòng")
            print("  7. u.info (Dataset info)")
            print("  8. Kiểm tra chất lượng dữ liệu")
            print("  9. CHẠY TẤT CẢ")

            choice = input("\nChọn (0-9): ").strip()

            if choice == '0':
                print("\nRaw Data Inspector!")
                break
            elif choice == '1':
                self.list_all_files()
            elif choice == '2':
                self.inspect_u_data()
            elif choice == '3':
                self.inspect_u_item()
            elif choice == '4':
                self.inspect_u_user()
            elif choice == '5':
                self.inspect_u_genre()
            elif choice == '6':
                self.inspect_u_occupation()
            elif choice == '7':
                self.inspect_u_info()
            elif choice == '8':
                self.check_data_quality()
            elif choice == '9':
                self.run_all_inspections()
                break
            else:
                print("❌ Lựa chọn không hợp lệ!")

            if choice in ['1', '2', '3', '4', '5', '6', '7', '8']:
                input("\n[Nhấn Enter để quay lại menu...]")


def main():
    import sys

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '../../data/raw/ml-100k')

    print(f"*****Script: {script_dir}")
    print(f"*****Data: {data_path}")

    if not os.path.exists(data_path):
        print(f"\n❌ Không tìm thấy dữ liệu tại {data_path}")
        return

    print(f"✓ Tìm thấy dữ liệu!\n")

    inspector = RawDataInspector(data_path=data_path)

    print("\nMode:")
    print("1. Tự động")
    print("2. Tương tác")

    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        choice = '1'

    else:
        choice = input("\nLựa chọn (1/2): ").strip()

    if choice == '1':
        inspector.run_all_inspections()
    elif choice == '2':
        inspector.interactive_menu()
    else:
        print("Lựa chọn không hợp lệ. Chạy menu tương tác...")
        inspector.interactive_menu()

    print("\n" + "=" * 80)
    print("✓ HOÀN TẤT!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()