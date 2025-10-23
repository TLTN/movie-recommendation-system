import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Configuration
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


class AdvancedMovieDataExplorer:
    """Khai phá dữ liệu"""

    def __init__(self, data_path: str = "../../data/raw/ml-100k", output_dir: str = None):
        self.data_path = Path(data_path) if isinstance(data_path, str) else data_path

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.output_dir = Path(os.path.join(script_dir, '../../results/plots2'))

        self.create_output_dir()
        # null dtframe
        self.ratings_df = None
        self.movies_df = None
        self.users_df = None
        self.load_data()

    def create_output_dir(self):
        """Tạo thư mục output"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"✓ Đã tạo thư mục: {self.output_dir}\n")

    def load_data(self):
        print("=" * 80)
        print("ĐANG TẢI DỮ LIỆU MOVIELENS 100K")
        print("=" * 80)

        try:
            # Load ratings
            self.ratings_df = pd.read_csv(
                self.data_path / "u.data",
                sep='\t',
                names=['userId', 'movieId', 'rating', 'timestamp'],
                engine='python'
            )
            # Convert timestamp to datetime
            self.ratings_df['datetime'] = pd.to_datetime(self.ratings_df['timestamp'], unit='s')
            self.ratings_df['year'] = self.ratings_df['datetime'].dt.year
            self.ratings_df['month'] = self.ratings_df['datetime'].dt.month
            print(f"Ratings: {self.ratings_df.shape}")

            # Load movies
            movie_columns = ['movieId', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] + \
                            ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                             'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                             'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

            self.movies_df = pd.read_csv(
                self.data_path / "u.item",
                sep='|',
                names=movie_columns,
                encoding='latin-1',
                engine='python'
            )

            genre_columns = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                             'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                             'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

            # thêm cột genres
            self.movies_df['genres'] = self.movies_df[genre_columns].apply(
                lambda x: '|'.join([col for col, val in x.items() if val == 1]), axis=1
            )
            # thêm cột main_genre
            self.movies_df['main_genre'] = self.movies_df['genres'].str.split('|').str[0]
            # chuyển release_date thành datetime
            self.movies_df['release_date'] = pd.to_datetime(
                self.movies_df['release_date'], format='%d-%b-%Y', errors='coerce'
            )
            # trích xuất release_year
            self.movies_df['release_year'] = self.movies_df['release_date'].dt.year
            print(f"✅ Movies: {self.movies_df.shape}")

            # Load users
            self.users_df = pd.read_csv(
                self.data_path / "u.user",
                sep='|',
                names=['userId', 'age', 'gender', 'occupation', 'zip_code'],
                engine='python'
            )
            print(f"✅ Users: {self.users_df.shape}\n")

        except FileNotFoundError as e:
            print(f"❌ Lỗi: {e}")
            raise
        # Chuẩn bị dữ liệu cho phân tích EDA

    def basic_overview(self):
        """1. Tổng quan"""
        print("\n" + "=" * 80)
        print("1. TỔNG QUAN CƠ BẢN")
        print("=" * 80)

        print("\n=====Thông tin datasets:")
        print(f"   • Ratings: {self.ratings_df.shape}")
        print(f"   • Movies: {self.movies_df.shape}")
        print(f"   • Users: {self.users_df.shape}")

        print("\n=====Missing values:")
        print(f"   • Ratings: {self.ratings_df.isnull().sum().sum()}")
        print(f"   • Movies (release_date): {self.movies_df['release_date'].isnull().sum()}")
        print(f"   • Users: {self.users_df.isnull().sum().sum()}")

        print("\n=====Duplicates:")
        print(f"   • Ratings: {self.ratings_df.duplicated().sum()}")
        print(f"   • Movies: {self.movies_df.duplicated(subset=['movieId']).sum()}")
        print(f"   • Users: {self.users_df.duplicated(subset=['userId']).sum()}")

        n_users = self.ratings_df['userId'].nunique()
        n_movies = self.ratings_df['movieId'].nunique()
        n_ratings = len(self.ratings_df)
        sparsity = 1 - (n_ratings / (n_users * n_movies))

        print(f"\n=====Thống kê:")
        print(f"   • Số users: {n_users:,}")
        print(f"   • Số movies: {n_movies:,}")
        print(f"   • Số ratings: {n_ratings:,}")
        print(f"   • Sparsity: {sparsity:.4f} ({sparsity * 100:.2f}%)")
        print(f"   • Rating trung bình: {self.ratings_df['rating'].mean():.2f}")
        print(f"   • Rating median: {self.ratings_df['rating'].median():.1f}")

        # biểu đồ cột phân phối điểm số
        fig, ax = plt.subplots(figsize=(10, 6))
        rating_counts = self.ratings_df['rating'].value_counts().sort_index()
        ax.bar(rating_counts.index, rating_counts.values, color='skyblue', edgecolor='navy', alpha=0.7)
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        ax.set_title('Rating Distribution', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(rating_counts.values):
            ax.text(rating_counts.index[i], v + 1000, f'{v:,}', ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / '01_rating_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Đã lưu: 01_rating_distribution.png")
        plt.close()

    def movie_analysis(self):
        """2. Phân tích phim"""
        print("\n" + "=" * 80)
        print("2. PHÂN TÍCH PHIM")
        print("=" * 80)

        # Tính thống kê cho mỗi phim
        movie_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean', 'std']
        }).reset_index()
        movie_stats.columns = ['movieId', 'rating_count', 'rating_mean', 'rating_std']
        # lấy tiêu đề và thể loại
        movie_stats = movie_stats.merge(self.movies_df[['movieId', 'title', 'main_genre']],
                                        on='movieId', how='left')

        print("\n=====Top 10 phim được đánh giá nhiều nhất:")
        top_movies = movie_stats.nlargest(10, 'rating_count')
        for i, row in enumerate(top_movies.itertuples(), 1):
            print(f"   {i:2}. {row.title[:45]:45} : {row.rating_count:3} ratings ({row.rating_mean:.2f}⭐)")

        print("\n=====Top 10 phim điểm cao nhất (≥50 ratings):")
        top_rated = movie_stats[movie_stats['rating_count'] >= 50].nlargest(10, 'rating_mean')
        for i, row in enumerate(top_rated.itertuples(), 1):
            print(f"   {i:2}. {row.title[:45]:45} : {row.rating_mean:.2f}⭐ ({row.rating_count} ratings)")

        # Top 15 phim được đánh giá nhiều nhất
        fig, ax = plt.subplots(figsize=(12, 8))
        top_15 = movie_stats.nlargest(15, 'rating_count')
        ax.barh(range(len(top_15)), top_15['rating_count'].values, color='gold', edgecolor='orange')
        ax.set_yticks(range(len(top_15)))
        ax.set_yticklabels([t[:35] for t in top_15['title']], fontsize=9)
        ax.set_xlabel('Number of Ratings')
        ax.set_title('Top 15 Most Rated Movies', fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / '02_top_movies_by_count.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Đã lưu: 02_top_movies_by_count.png")
        plt.close()

        # Top 15 phim có điểm trung bình cao nhất (>=50)
        fig, ax = plt.subplots(figsize=(12, 8))
        top_15_rated = movie_stats[movie_stats['rating_count'] >= 50].nlargest(15, 'rating_mean')
        ax.barh(range(len(top_15_rated)), top_15_rated['rating_mean'].values, color='salmon', edgecolor='darkred')
        ax.set_yticks(range(len(top_15_rated)))
        ax.set_yticklabels([t[:35] for t in top_15_rated['title']], fontsize=9)
        ax.set_xlabel('Average Rating')
        ax.set_title('Top 15 Highest Rated Movies (≥50 ratings)', fontweight='bold')
        ax.set_xlim([0, 5])
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / '03_top_movies_by_rating.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Đã lưu: 03_top_movies_by_rating.png")
        plt.close()
    # Xác định phim phổ biến và chất lượng cao trong tập dữ liệu

    def genre_analysis(self):
        """Phân tích theo thể loại"""
        print("\n" + "=" * 80)
        print("3. PHÂN TÍCH THỂ LOẠI")
        print("=" * 80)

        genre_columns = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                         'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                         'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

        # Genre count
        genre_counts = self.movies_df[genre_columns].sum().sort_values(ascending=False)
        print("\n=====Số phim theo thể loại:")
        for genre, count in genre_counts.head(10).items():
            print(f"   {genre:15} : {count:4} movies")

        # điểm trung bình của mỗi thể loại dựa trên ratings_df
        genre_ratings = {}
        for genre in genre_columns:
            genre_movies = self.movies_df[self.movies_df[genre] == 1]['movieId']
            genre_ratings[genre] = self.ratings_df[self.ratings_df['movieId'].isin(genre_movies)]['rating'].mean()

        genre_ratings_sorted = dict(sorted(genre_ratings.items(), key=lambda x: x[1], reverse=True))
        print("\n⭐ Rating trung bình theo thể loại:")
        # top 10 thể loại theo số lượng phim và điểm trung bình
        for genre, rating in list(genre_ratings_sorted.items())[:10]:
            print(f"   {genre:15} : {rating:.2f}")

        # Biểu đồ cột của số lượng phim theo thể loại
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(genre_counts)), genre_counts.values, color='teal', edgecolor='darkslategray')
        ax.set_xticks(range(len(genre_counts)))
        ax.set_xticklabels(genre_counts.index, rotation=45, ha='right')
        ax.set_ylabel('Number of Movies')
        ax.set_title('Genre Distribution', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / '04_genre_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Đã lưu: 04_genre_distribution.png")
        plt.close()

        # Biểu đồ cột ngang của điểm trung bình theo thể loại
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(range(len(genre_ratings_sorted)), list(genre_ratings_sorted.values()),
                color='salmon', edgecolor='darkred')
        ax.set_yticks(range(len(genre_ratings_sorted)))
        ax.set_yticklabels(list(genre_ratings_sorted.keys()))
        ax.set_xlabel('Average Rating')
        ax.set_title('Average Rating by Genre', fontweight='bold')
        ax.set_xlim([0, 5])
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / '05_avg_rating_by_genre.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Đã lưu: 05_avg_rating_by_genre.png")
        plt.close()

    def user_analysis(self):
        """3. Phân tích người dùng"""
        print("\n" + "=" * 80)
        print("4. PHÂN TÍCH NGƯỜI DÙNG")
        print("=" * 80)

        # Tính thống kê cho mỗi người dùng
        user_stats = self.ratings_df.groupby('userId').agg({
            'rating': ['count', 'mean', 'std']
        }).reset_index()
        user_stats.columns = ['userId', 'rating_count', 'rating_mean', 'rating_std']
        # Gộp với users_df lấy thông tin nhân khẩu học
        user_stats = user_stats.merge(self.users_df, on='userId', how='left')

        print(f"\n=====Thống kê người dùng:")
        print(f"   • Số rating TB/user: {user_stats['rating_count'].mean():.1f}")
        print(f"   • User ít rating nhất: {user_stats['rating_count'].min()}")
        print(f"   • User nhiều rating nhất: {user_stats['rating_count'].max()}")
        print(f"   • Rating TB/user: {user_stats['rating_mean'].mean():.2f}")

        # người dùng dễ, khó
        harsh_users = user_stats[user_stats['rating_mean'] < 3]
        easy_users = user_stats[user_stats['rating_mean'] > 4]
        print(f"\n   • Harsh raters (<3): {len(harsh_users)} users")
        print(f"   • Easy raters (>4): {len(easy_users)} users")

        # Phân phối số lượng đánh giá mỗi người dùng
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(user_stats['rating_count'], bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Number of Ratings')
        ax.set_ylabel('Number of Users')
        ax.set_title('User Activity Distribution', fontweight='bold')
        ax.axvline(user_stats['rating_count'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {user_stats["rating_count"].mean():.1f}')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / '06_user_activity.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Đã lưu: 06_user_activity.png")
        plt.close()

        # Phân phối điểm trung bình mỗi người dùng
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(user_stats['rating_mean'], bins=30, color='lightblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Average Rating')
        ax.set_ylabel('Number of Users')
        ax.set_title('User Average Rating Distribution', fontweight='bold')
        ax.axvline(user_stats['rating_mean'].mean(), color='red', linestyle='--',
                   label=f'Mean: {user_stats["rating_mean"].mean():.2f}')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / '07_user_avg_rating.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Đã lưu: 07_user_avg_rating.png")
        plt.close()
    # mức độ tích cực và xu hướng đánh giá của người dùng

    def demographic_analysis(self):
        """4. Phân tích nhân khẩu học"""
        print("\n" + "=" * 80)
        print("5. PHÂN TÍCH NHÂN KHẨU HỌC")
        print("=" * 80)

        # Gộp ratings_df + users_df
        merged_df = self.ratings_df.merge(self.users_df, on='userId', how='left')

        # Tính điểm trung bình và số lượng đánh giá theo giới tính
        print("\n=====Phân tích theo giới tính:")
        gender_stats = merged_df.groupby('gender')['rating'].agg(['mean', 'count'])
        # giới tính
        for gender, row in gender_stats.iterrows():
            gender_name = "Nam" if gender == 'M' else "Nữ"
            print(f"   {gender_name}: {row['mean']:.2f} (n={row['count']:,})")

        # Tính điểm trung bình và số lượng đánh giá theo nhóm tuổi
        print("\n=====Phân tích theo độ tuổi:")
        merged_df['age_group'] = pd.cut(merged_df['age'], bins=[0, 18, 25, 35, 50, 100],
                                        labels=['<18', '18-25', '26-35', '36-50', '50+'])
        # nhóm tuổi
        age_stats = merged_df.groupby('age_group')['rating'].agg(['mean', 'count'])
        print(age_stats)

        # top 10 nghề nghiệp
        print("\n=====Top 10 nghề nghiệp:")
        occ_counts = self.users_df['occupation'].value_counts().head(10)
        for occ, count in occ_counts.items():
            print(f"   {occ:20} : {count:3} users")

        # Biểu đồ cột điểm trung bình theo giới tính
        fig, ax = plt.subplots(figsize=(8, 6))
        gender_means = merged_df.groupby('gender')['rating'].mean()
        bars = ax.bar(['Female', 'Male'], gender_means.values, color=['#e74c3c', '#3498db'],
                      edgecolor='black', alpha=0.7)
        ax.set_ylabel('Average Rating')
        ax.set_title('Average Rating by Gender', fontweight='bold')
        ax.set_ylim([0, 5])
        ax.grid(axis='y', alpha=0.3)
        for bar, v in zip(bars, gender_means.values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1, f'{v:.2f}',
                    ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / '08_rating_by_gender.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Đã lưu: 08_rating_by_gender.png")
        plt.close()

        # Biểu đồ đường điểm trung bình theo nhóm tuổi
        fig, ax = plt.subplots(figsize=(10, 6))
        age_means = merged_df.groupby('age_group')['rating'].mean()
        ax.plot(range(len(age_means)), age_means.values, marker='o', linewidth=2,
                markersize=10, color='purple', alpha=0.7)
        ax.set_xticks(range(len(age_means)))
        ax.set_xticklabels(age_means.index)
        ax.set_ylabel('Average Rating')
        ax.set_xlabel('Age Group')
        ax.set_title('Average Rating by Age Group', fontweight='bold')
        ax.set_ylim([0, 5])
        ax.grid(alpha=0.3)
        for i, v in enumerate(age_means.values):
            ax.text(i, v + 0.05, f'{v:.2f}', ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / '09_rating_by_age.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Đã lưu: 09_rating_by_age.png")
        plt.close()

        # Biểu đồ cột ngang của top 10 nghề nghiệp
        fig, ax = plt.subplots(figsize=(12, 6))
        top_10_occ = self.users_df['occupation'].value_counts().head(10)
        ax.barh(range(len(top_10_occ)), top_10_occ.values, color='steelblue', edgecolor='navy')
        ax.set_yticks(range(len(top_10_occ)))
        ax.set_yticklabels(top_10_occ.index)
        ax.set_xlabel('Number of Users')
        ax.set_title('Top 10 Occupations', fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / '10_top_occupations.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Đã lưu: 10_top_occupations.png")
        plt.close()
    # mối quan hệ giữa nhân khẩu học và hành vi đánh giá

    def temporal_analysis(self):
        """5. Phân tích theo thời gian"""
        print("\n" + "=" * 80)
        print("6. PHÂN TÍCH THEO THỜI GIAN")
        print("=" * 80)

        # số lượng
        ratings_by_month = self.ratings_df.groupby(['year', 'month']).size().reset_index(name='count')
        ratings_by_month['date'] = pd.to_datetime(ratings_by_month[['year', 'month']].assign(day=1))

        print(f"\n=====Thời gian:")
        print(f"   • Từ: {self.ratings_df['datetime'].min()}")
        print(f"   • Đến: {self.ratings_df['datetime'].max()}")
        print(f"   • Tổng tháng: {len(ratings_by_month)}")

        # điểm trung bình đánh giá
        avg_rating_by_month = self.ratings_df.groupby(['year', 'month'])['rating'].mean().reset_index()
        avg_rating_by_month['date'] = pd.to_datetime(avg_rating_by_month[['year', 'month']].assign(day=1))

        print(f"\n=====Rating TB theo năm:")
        yearly_avg = self.ratings_df.groupby('year')['rating'].mean()
        for year, rating in yearly_avg.items():
            print(f"   {year}: {rating:.2f}")

        # Biểu đồ đường của số lượng đánh giá theo thời giane
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(ratings_by_month['date'], ratings_by_month['count'], linewidth=2, color='steelblue')
        ax.fill_between(ratings_by_month['date'], ratings_by_month['count'], alpha=0.3)
        ax.set_xlabel('Time')
        ax.set_ylabel('Number of Ratings')
        ax.set_title('Number of Ratings Over Time', fontweight='bold')
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / '11_ratings_over_time.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Đã lưu: 11_ratings_over_time.png")
        plt.close()

        # Biểu đồ đường của điểm trung bình theo thời gian
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(avg_rating_by_month['date'], avg_rating_by_month['rating'],
                linewidth=2, color='darkred', marker='o', markersize=4)
        ax.set_xlabel('Time')
        ax.set_ylabel('Average Rating')
        ax.set_title('Average Rating Over Time', fontweight='bold')
        ax.set_ylim([0, 5])
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / '12_avg_rating_over_time.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Đã lưu: 12_avg_rating_over_time.png")
        plt.close()

    def correlation_analysis(self):
        """6. Phân tích tương quan"""
        print("\n" + "=" * 80)
        print("7. PHÂN TÍCH TƯƠNG QUAN")
        print("=" * 80)

        # DataFrame: điểm số, tuổi, điểm trung bình của người dùng, phim
        corr_data = self.ratings_df[['userId', 'movieId', 'rating']].copy()
        corr_data['user_avg_rating'] = corr_data.groupby('userId')['rating'].transform('mean')
        corr_data['movie_avg_rating'] = corr_data.groupby('movieId')['rating'].transform('mean')
        corr_data = corr_data.merge(self.users_df[['userId', 'age']], on='userId', how='left')

        correlation_matrix = corr_data[['rating', 'age', 'user_avg_rating', 'movie_avg_rating']].corr()
        print("\n=====Ma trận tương quan:")
        print(correlation_matrix)

        # heatmap của ma trận tương quan
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)

        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_yticks(range(len(correlation_matrix.columns)))
        labels = ['Rating', 'Age', 'User Avg', 'Movie Avg']
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)

        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                        ha="center", va="center", color="black", fontweight='bold')

        ax.set_title('Correlation Matrix', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / '13_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Đã lưu: 13_correlation_heatmap.png")
        plt.close()
    # mối quan hệ giữa các đặc trưng

    def genre_preference_by_demographics(self):
        """Phân tích thể loại theo nhân khẩu học"""
        print("\n" + "=" * 80)
        print("8. THỂ LOẠI YÊU THÍCH THEO NHÂN KHẨU HỌC")
        print("=" * 80)

        genre_columns = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                         'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                         'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

        # ratings_df + users_df + movies_df
        merged_df = self.ratings_df.merge(self.users_df, on='userId', how='left')
        merged_df = merged_df.merge(self.movies_df[['movieId'] + genre_columns], on='movieId', how='left')

        print("\n=====Thể loại yêu thích theo giới tính:")
        gender_genre = {}
        # tính điểm trung bình mỗi thể loại theo giới tính
        for gender in ['M', 'F']:
            gender_ratings = merged_df[merged_df['gender'] == gender]
            genre_avg = {}
            for genre in genre_columns:
                genre_movies = gender_ratings[gender_ratings[genre] == 1]['rating'].mean()
                genre_avg[genre] = genre_movies
            gender_genre[gender] = genre_avg

        # top 5 thể loại
        for gender in ['M', 'F']:
            gender_name = "Nam" if gender == 'M' else "Nữ"
            sorted_genres = dict(sorted(gender_genre[gender].items(), key=lambda x: x[1], reverse=True))
            print(f"\n   {gender_name}:")
            for i, (genre, rating) in enumerate(list(sorted_genres.items())[:5], 1):
                print(f"      {i}. {genre:15} : {rating:.2f}")

        # biểu đồ cột so sánh điểm tb thể loại nam-nữ
        fig, ax = plt.subplots(figsize=(14, 6))

        male_ratings = [gender_genre['M'][g] for g in genre_columns]
        female_ratings = [gender_genre['F'][g] for g in genre_columns]

        x = np.arange(len(genre_columns))
        width = 0.35

        bars1 = ax.bar(x - width / 2, male_ratings, width, label='Male', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width / 2, female_ratings, width, label='Female', color='#e74c3c', alpha=0.8)

        ax.set_xlabel('Genre')
        ax.set_ylabel('Average Rating')
        ax.set_title('Genre Preference by Gender', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(genre_columns, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 5])

        plt.tight_layout()
        plt.savefig(self.output_dir / '14_genre_by_gender.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\n✓ Đã lưu: 14_genre_by_gender.png")
        plt.close()

        # tính điểm trung bình thể loại theo độ tuổi
        print("\n=====Thể loại yêu thích theo độ tuổi:")
        merged_df['age_group'] = pd.cut(merged_df['age'], bins=[0, 25, 35, 50, 100],
                                        labels=['<25', '25-35', '36-50', '50+'])

        age_genre = {}
        for age_grp in ['<25', '25-35', '36-50', '50+']:
            age_ratings = merged_df[merged_df['age_group'] == age_grp]
            genre_avg = {}
            for genre in genre_columns:
                genre_movies = age_ratings[age_ratings[genre] == 1]['rating'].mean()
                genre_avg[genre] = genre_movies
            age_genre[age_grp] = genre_avg

        for age_grp in ['<25', '25-35', '36-50', '50+']:
            sorted_genres = dict(sorted(age_genre[age_grp].items(), key=lambda x: x[1], reverse=True))
            print(f"\n   {age_grp}:")
            for i, (genre, rating) in enumerate(list(sorted_genres.items())[:3], 1):
                print(f"      {i}. {genre:15} : {rating:.2f}")

    def movie_release_year_analysis(self):
        """Phân tích theo năm phát hành"""
        print("\n" + "=" * 80)
        print("9. PHÂN TÍCH THEO NĂM PHÁT HÀNH")
        print("=" * 80)

        # rating_df + movies_df
        movie_ratings = self.ratings_df.merge(
            self.movies_df[['movieId', 'release_year']],
            on='movieId', how='left'
        )
        movie_ratings = movie_ratings.dropna(subset=['release_year'])

        # điểm tb
        yearly_avg = movie_ratings.groupby('release_year')['rating'].mean()
        # số lượng
        yearly_count = movie_ratings.groupby('release_year').size()

        print(f"\n=====Thống kê theo năm:")
        print(f"   • Năm sớm nhất: {int(yearly_avg.index.min())}")
        print(f"   • Năm muộn nhất: {int(yearly_avg.index.max())}")
        print(f"   • Số năm: {len(yearly_avg)}")

        print(f"\n=====Top 5 năm có rating cao nhất:")
        top_years = yearly_avg.nlargest(5)
        for year, rating in top_years.items():
            print(f"   {int(year)}: {rating:.2f} ({yearly_count[year]} ratings)")

        # biểu đồ cột số lượng phim theo năm
        movies_by_year = self.movies_df.dropna(subset=['release_year'])['release_year'].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(movies_by_year.index, movies_by_year.values, color='steelblue', alpha=0.7)
        ax.set_xlabel('Release Year')
        ax.set_ylabel('Number of Movies')
        ax.set_title('Number of Movies by Release Year', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / '15_movies_by_year.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\n✓ Đã lưu: 15_movies_by_year.png")
        plt.close()

        # biểu đồ đường điểm tb theo năm
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(yearly_avg.index, yearly_avg.values, linewidth=2, color='darkred', marker='o', markersize=3)
        ax.set_xlabel('Release Year')
        ax.set_ylabel('Average Rating')
        ax.set_title('Average Rating by Release Year', fontweight='bold')
        ax.set_ylim([0, 5])
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / '16_rating_by_year.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Đã lưu: 16_rating_by_year.png")
        plt.close()
    # xu hướng sản xuất phim + đánh giá theo thời gian

    def cross_feature_analysis(self):
        """Phân tích chéo các đặc trưng"""
        print("\n" + "=" * 80)
        print("10. PHÂN TÍCH CHÉO CÁC ĐẶC TRƯNG")
        print("=" * 80)

        merged_df = self.ratings_df.merge(self.users_df, on='userId', how='left')

        # điểm tb theo nghề nghiệp
        print("\n=====Rating trung bình theo nghề nghiệp:")
        occ_rating = merged_df.groupby('occupation')['rating'].mean().sort_values(ascending=False)
        for i, (occ, rating) in enumerate(occ_rating.head(10).items(), 1):
            print(f"   {i:2}. {occ:20} : {rating:.2f}")

        # biểu đồ cột ngang điểm tb theo top10 nghề nghiệp
        fig, ax = plt.subplots(figsize=(12, 8))
        top_10_occ = occ_rating.head(10)
        ax.barh(range(len(top_10_occ)), top_10_occ.values, color='teal', edgecolor='darkslategray')
        ax.set_yticks(range(len(top_10_occ)))
        ax.set_yticklabels(top_10_occ.index)
        ax.set_xlabel('Average Rating')
        ax.set_title('Average Rating by Occupation (Top 10)', fontweight='bold')
        ax.set_xlim([0, 5])
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / '17_rating_by_occupation.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\n✓ Đã lưu: 17_rating_by_occupation.png")
        plt.close()

        # điểm tb theo nhóm tuổi, giới tính
        merged_df['age_group'] = pd.cut(merged_df['age'], bins=[0, 25, 35, 50, 100],
                                        labels=['<25', '25-35', '36-50', '50+'])

        age_gender_pivot = merged_df.groupby(['age_group', 'gender'])['rating'].mean().unstack()

        print("\n=====Rating theo tuổi và giới tính:")
        print(age_gender_pivot)

        # biểu đồ cột so điểm tb theo tuổi, giới tính
        fig, ax = plt.subplots(figsize=(10, 6))
        age_gender_pivot.plot(kind='bar', ax=ax, color=['#e74c3c', '#3498db'], alpha=0.8)
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Average Rating')
        ax.set_title('Average Rating by Age Group and Gender', fontweight='bold')
        ax.set_ylim([0, 5])
        ax.legend(['Female', 'Male'])
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / '18_age_gender_interaction.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Đã lưu: 18_age_gender_interaction.png")
        plt.close()
    # tác động của các đặc trưng nhân khẩu học đến điểm số

    def behavioral_insights(self):
        """Phân tích hành vi người dùng"""
        print("\n" + "=" * 80)
        print("11. PHÂN TÍCH HÀNH VI NGƯỜI DÙNG")
        print("=" * 80)

        # thống kê user
        user_stats = self.ratings_df.groupby('userId').agg({
            'rating': ['count', 'mean', 'std']
        }).reset_index()
        user_stats.columns = ['userId', 'rating_count', 'rating_mean', 'rating_std']
        user_stats = user_stats.merge(self.users_df, on='userId', how='left')

        # phân khúc user
        user_stats['segment'] = 'Normal'
        user_stats.loc[user_stats['rating_mean'] < 2.5, 'segment'] = 'Harsh Critic'
        user_stats.loc[user_stats['rating_mean'] > 4.2, 'segment'] = 'Easy Rater'
        user_stats.loc[user_stats['rating_count'] > 200, 'segment'] = 'Cinephile'
        user_stats.loc[user_stats['rating_count'] < 50, 'segment'] = 'Casual Viewer'

        segment_counts = user_stats['segment'].value_counts()
        # phân phối
        print("\n=====Phân loại người dùng:")
        for segment, count in segment_counts.items():
            pct = (count / len(user_stats)) * 100
            print(f"   {segment:20} : {count:3} users ({pct:.1f}%)")

        # biểu đồ tròn phân khúc
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        wedges, texts, autotexts = ax.pie(segment_counts.values, labels=segment_counts.index,
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('User Segments Distribution', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / '19_user_segments.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\n✓ Đã lưu: 19_user_segments.png")
        plt.close()

        # độ biến thiên điểm
        print("\n=====Phân tích độ biến thiên rating:")
        print(f"   • Std TB: {user_stats['rating_std'].mean():.2f}")
        print(f"   • User có std cao nhất: {user_stats['rating_std'].max():.2f}")
        print(f"   • User có std thấp nhất: {user_stats['rating_std'].min():.2f}")

        # histogram độ biến thiên điểm
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(user_stats['rating_std'].dropna(), bins=30, color='orange', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Rating Standard Deviation')
        ax.set_ylabel('Number of Users')
        ax.set_title('Distribution of Rating Variance Across Users', fontweight='bold')
        ax.axvline(user_stats['rating_std'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {user_stats["rating_std"].mean():.2f}')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / '20_rating_variance.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Đã lưu: 20_rating_variance.png")
        plt.close()

    def run_complete_eda(self):
        """Chạy toàn bộ phân tích"""
        print("\n" + "=" * 80)
        print("ĐANG CHẠY PHÂN TÍCH TOÀN DIỆN")
        print("=" * 80 + "\n")

        self.basic_overview()
        self.movie_analysis()
        self.genre_analysis()
        self.user_analysis()
        self.demographic_analysis()
        self.temporal_analysis()
        self.correlation_analysis()
        self.genre_preference_by_demographics()
        self.movie_release_year_analysis()
        self.cross_feature_analysis()
        self.behavioral_insights()

        print("\n" + "=" * 80)
        print("HOÀN TẤT! Tất cả biểu đồ đã được lưu vào:")
        print(f"{self.output_dir}")
        print("=" * 80)

    def interactive_menu(self):
        """Menu tương tác"""
        print("\n" + "=" * 80)
        print("MENU KHÁM PHÁ TƯƠNG TÁC")
        print("=" * 80)

        while True:
            print("\n=====Chọn phân tích:")
            print("  1. Tổng quan cơ bản")
            print("  2. Phân tích phim")
            print("  3. Phân tích thể loại")
            print("  4. Phân tích người dùng")
            print("  5. Phân tích nhân khẩu học")
            print("  6. Phân tích theo thời gian")
            print("  7. Phân tích tương quan")
            print("  8. Thể loại theo nhân khẩu học")
            print("  9. Phân tích năm phát hành")
            print(" 10. Phân tích chéo")
            print(" 11. Phân tích hành vi")
            print(" 12. CHẠY TẤT CẢ")
            print("  0. Thoát")

            choice = input("\nChọn (0-12): ").strip()

            if choice == '0':
                print("\nCảm ơn bạn đã sử dụng!")
                break
            elif choice == '1':
                self.basic_overview()
            elif choice == '2':
                self.movie_analysis()
            elif choice == '3':
                self.genre_analysis()
            elif choice == '4':
                self.user_analysis()
            elif choice == '5':
                self.demographic_analysis()
            elif choice == '6':
                self.temporal_analysis()
            elif choice == '7':
                self.correlation_analysis()
            elif choice == '8':
                self.genre_preference_by_demographics()
            elif choice == '9':
                self.movie_release_year_analysis()
            elif choice == '10':
                self.cross_feature_analysis()
            elif choice == '11':
                self.behavioral_insights()
            elif choice == '12':
                self.run_complete_eda()
                break
            else:
                print("Lựa chọn không hợp lệ!")

            input("\nNhấn Enter để tiếp tục...")


def main():
    import sys

    print("\n" + "=" * 80)
    print("MOVIELENS 100K - ADVANCED EDA")
    print("=" * 80 + "\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '../../data/raw/ml-100k')

    print(f"Script: {script_dir}")
    print(f"Data: {data_path}")

    if not os.path.exists(data_path):
        print(f"\nKhông tìm thấy dữ liệu tại {data_path}")
        return

    print(f"✓ Tìm thấy dữ liệu!\n")

    explorer = AdvancedMovieDataExplorer(data_path=data_path)

    print("\nChọn chế độ:")
    print("1. Tự động (chạy tất cả)")
    print("2. Tương tác (từng bước)")

    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        choice = '1'
    else:
        choice = input("\nLựa chọn (1/2): ").strip()

    if choice == '1':
        explorer.run_complete_eda()
    elif choice == '2':
        explorer.interactive_menu()
    else:
        print("Lựa chọn không hợp lệ. Chạy tự động...")
        explorer.run_complete_eda()

    print(f"\n✓ Hoàn tất! Kết quả: {explorer.output_dir}\n")


if __name__ == "__main__":
    main()