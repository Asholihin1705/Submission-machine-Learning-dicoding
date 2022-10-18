# Laporan Proyek Machine Learning - Ahmad Sholihin
## _Machine Learning Terapan Dicoding_

[(Perlihatkan Kelas)](https://www.dicoding.com/academies/319)


Menurut riset yang berjudul 2020 Emerging Job Report, trend perekrutan untuk peran AI specialist tumbuh 74% selama 4 tahun terakhir. AI specialist dinobatkan sebagai peringkat pertama the most emerging job in the US in 2020. Di Indonesia, AI specialist juga menempati urutan pertama. Hal ini karena otomatisasi telah mengubah cara orang hidup dan bekerja setiap harinya. Posisi Machine Learning Developer sangat dicari. Sehingga, di Amerika, posisi ini bisa mendapatkan pekerjaan senilai Rp 1,9 miliar per tahun (Kompas)..

- Machine Learning dapat digunakan untuk meningkatkan efisiensi dari berbagai pekerjaan.
- Machine Learning dapat diimplementasikan ke berbagai industri dan berbagai jenis data sehingga kegunaannya sangat luas.
- Banyak perusahaan memiliki jumlah data yang sangat besar sehingga perlu diproses dengan machine learning untuk mendapatkan informasi  yang berarti.
- Kebutuhan karier di bidang Machine Learning sangatlah tinggi karena jumlah praktisi yang masih sedikit sehingga peluangnya masih sangat besar.
-  Pemahaman tentang Machine Learning, TensorFlow, dan Keras adalah keharusan untuk menjadi seorang Machine Learning Developer ataupun Data Scientist.
- Mengerjakan proyek-proyek Machine Learning sebagai portofolio merupakan keahlian yang harus dimiliki untuk mereka yang ingin memulai karier menjadi Machine Learning Developer.

## Domain Proyek --> Penerapan Algoritma LSTM dan RNN untuk Prediksi Harga Saham UNVR
---
Istilah investasi sekarang sedang sering dibahas dan sangat populer di kalangan masyarakat umum. Menurut Kamus Besar Bahasa Indonesia modern, investasi adalah aktivitas berbisnis dalam menanamkan modal berupa modal seperti uang ke dalam wadah sebagai contoh usaha atau proyek, untuk menghasilkan keuntungan di masa depan. Seiring dengan perkembangan bidang industri dan teknologi, serta berkembangnya semangat komunitas investasi global, investasi telah berkembang secara fisik, namun telah berkembang menjadi sekuritas seperti obligasi, saham, deposito dan sebagainya. Untuk dapat meningkatkan jumlah investasi, investor harus memiliki strategi untuk mampu membuat portofolio dengan pengembalian dana yang tinggi dan risiko yang rendah[1].

Saham adalah suatu pilihan investasi yang sangat menarik dikarenakan dapat menghasilkan pengembalian modal yang besar dibandingkan dengan investasi lain. Walaupun juga memiliki resiko kerugiannya sangat besar dalam waktu sekejap. Untuk meminimalkan risiko kerugian yang cukup besar, diperlukan perhatian yang khusus untuk melihat pergerakan saham yang sedang terjadi. Untuk meramalkan harga saham dapat dilakukan dengan 3 faktor yang mempengaruhi, yaitu faktor teknis, faktor fundamental dan faktor emosional. Faktor teknis adalah pergerakan dari pengamatan harga masa lalu, Faktor fundamental adalah analisis teknis tentang bagaimana pendekatan perdagangan terjadi, Faktor psikologis adalah bahwa pergerakan harga suatu saham dipengaruhi oleh elemen perdagangan, berita dan aktivitas bisnis[2].

Perkembangan zaman modern ini, menjadikan ilmu di bidang teknologi informasi serta ilmu lainnya telah menghadapi kemajuan cukup pesat, seperti metode jaringan syaraf tiruan. Jaringan syaraf tiruan merupakan proses komputasi pada pemrograman komputer yang terinspirasi cara kerja otak manusia. Jaringan syaraf tiruan berfungsi untuk mengolah data untuk memperoleh informasi dari data tersebut. Jaringan syaraf tiruan juga bisa mendapatkan dan mengidentifikasi data sebelumnya. Data historis yang diproses dari jaringan saraf ini bakal diperiksa oleh sistem yang akan membantu membuat keputusan tentang data yang tidak terlatih di masa mendatang.

Recurrent Neural Network (RNN) adalah algoritma deep learning yang mempunyai model struktur pada jaringan syaraf tiruan yang memiliki proses alur kerja iteratif dengan satu input[3]. RNN umumnya menggunakan data bertipe sekuensial untuk penggunaan arsitekturnya. Pemodelan data dengan RNN cocok untuk diterapkan[4].Long short term memory (LSTM) adalah salah satu metode yang dikembangkan menggunakan metode RNN. Algoritma LSTM mempertimbangkan data dan output masa lalu. Karena karakteristiknya, LSTM cocok untuk data sekuensial yang bergantung pada rilis sebelumnya, seperti nilai inventaris[5]. Jaringan LSTM ini banyak dimanfaatkan untuk pengolahan data deret waktu, pengolahan kata, pengolahan video serta pengolahan lainnya.

Berdasarkan permasalahan yang dijabarkan diatas menerapkan algoritma Long Short Term Memory(LSTM) dan Recurrent Neural Network (RNN) yang merupakan metode dari jaringan syaraf tiruan, yang akan digunakan untuk penerapan algoritma LSTM dan RNN dalam memprediksi harga saham PT Unilever Indonesia Tbk

##### _Referensi Didapatkan dari beberapa jurnal dibawah ini_
- [1] W. Hastomo, A. S. B. Karno, and N. Kalbuana, “Optimasi Deep Learning untuk Prediksi Saham di Masa Pandemi Covid-19,” vol. 7, no. 2, pp. 133–140, 2021.
- [2] M. A. D. Suyudi, E. C. Djamal, A. Maspupah, J. Informatika, and F. Sains, “Recurrent Neural Network,” pp. 33–38, 2019.
- [3] Moreta, E. S. (2020) ‘MENGGUNAKANDEEP LEARNING DALAM BAHASA PYTHON’, 4(September).
- [4] J. K. Lubis and I. Kharisudin, “Metode Long Short Term Memory dan Generalized Autoregressive Conditional Heteroscedasticity untuk Pemodelan Data Saham,” Prism. Pros. Semin. Nas. …, vol. 4, pp. 652–658, 2021, [Online]. Available: https://journal.unnes.ac.id/sju/index.php/prisma/article/view/44897.
- [5] A. B. Nurjaman, A. Hasim, and A. M. Zakiri, “Long Short-Term Memory ( LSTM ) untuk Prediksi Harga Saham Pfizer Inc,” 2021.


## Business Understanding
---

### Rumusan Masalah
Berlandaskan dari pemaparan latar belakang, Saya dapat merumuskan beberapa masalah yang ingin diselesaikan.
- Bagiamana proses penerapan Algoritma Deep Learning untuk melakukan prediksi harga saham
- Bagaimana kinerja algoritma Deep Learning untuk prediksi pregerakan harga saham
- Mengetahui akurasi terbaik antara algoritma LSTM dan RNN untuk prediksi.

### Tujuan
Tujuan dari proyek yang hendak diperoleh berdasarkan permasalahan yang dirumuskan di atas adalah sebagai berikut.
- Cara penerapan algoritma Deep Learning yang sesuai sehingga dapat digunakan untuk memprediksi saham
- Untuk menghitung tingkat akurasi ketepatan agoritma Deep Learning yang dibuat berdasarkan data saham
- Memperoleh hasil akurasi yang terbaik antara algoritma LSTM dan RNN.

### Batasan Proyek
Untuk melakukan proyek ini makin intensif berdasarkan definisi masalah yang dipaparkan, peneliti menentukan batasan masalah sebagai berikut:
- Penerapan algoritma LSTM dan RNN untuk melakukan peramalan terhadap harga saham.
- Keakuratan algoritma LSTM dan RNN saat digunakan untuk memprediksi harga saham.
- Data yang diaplikasikan untuk penelitian ini merupakaan data dari harga saham harian PT Unilever Indonesia Tk yang diambil dari situs web yahoo finance.
- Tools yang digunakan di dalam penelitian ini adalah Google Colab dengan menggunakan bahasa python.

### Solusi dari Permasalahan
- Untuk mendapatkan penerapan algoritma deep learning, saya menggunakan 2 algoritma yaitu LSTM dan RNN supaya dapat mengetahui akurasi yang lebih bagus.
- Evaluasi Akurasi yang digunakan adalah perhitungan MAE, MAPE, dan R2 Square
- Hasil dari akurasi MAE di visualisasikan ke bentuk line grafik agar mempermudah pembacaannya perbandingan dari kedua algoritma.
- Saya menggunakan parameter yang sama agar hasil yang didapatkan adil untuk kedua algoritma.


## Data Understanding
---

Data understanding adalah sebuah tahapan di dalam metodologi sains data dan pengembangan AI yang bertujuan untuk mendapatkan pemahaman awal mengenai data yang dibutuhkan untuk memecahkan permasalahan bisnis yang diberikan. Data understanding memberikan gambaran awal tentang:

- Kekuatan data.
- Kekurangan dan batasan penggunaan data.
- Tingkat kesesuaian data dengan masalah bisnis yang akan dipecahkan.
- Ketersediaan data (terbuka/tertutup, biaya akses, dsb).

### Dataset dan Library
Dataset yang digunakan untuk melakukan penelitian ini merupakan harga penutupan harian lembar saham (closing price) PT Unilever Indonesia Tbk dengan kode saham UNVR. Data yang digunakan adalah data harga penutupan harian (closing price), dan periode harian dari data saham adalah dari 03 September 2003 sampai dengan 02 September 2022. Berikut adalah grafik pergerakan harga saham PT Unilever Indonesia Tbk yang terdiri dari 4724 baris dan 7 kolom. Library yang digunakan yaitu Pandas, Numpy, Matplotlib, Seaborn, Scikit-Learn dan Keras. Serta Algoritma yang akan digunakan adalah Long Sort Term Memory(LSTM) dan Recurrent Neural Network(RNN).
##### Variabel - Variabel yang terdapat di dataset
1. Date : Menjelaskan tanggal dari harga saham
2. Open : Menjelaskan pembukaan harga saham pada tanggal itu
3. High : Menjelaskan harga saham yang tertinggi pada tanggal itu
4. Low : Menjelaskan harga saham yang terendah pada tanggal itu
5. Close Menjelaskan penutupan harga saham pada tanggal itu
6. Adj Close : Harga penutupan yang disesuaikan dengan aksi korporasi seperti right issue, stock split atau stock reverse
7. Volume : Volume transaksi biasanya dalam jumlah lembar

### Visualisasi dan exploratory data analysis(EDA)
Mencari apakah ada data yang null dari dataset yang digunakan
```sh
df.isna().sum()
```
Melakukan Visualisasi data dengan menggunakan variabel harga penutupanya saja. Berikut ini hasil dari visualisasi yang dilakukan
![Visualisasi harga penutupan saham](https://user-images.githubusercontent.com/56246122/188430736-0fa733b4-5d7d-4003-a948-1a92f4f84f91.png) 

Melakukan filtering terhadap variable harga close, karena pada proyek ini yang akan digunakan untuk melakukan prediksi adalah data penutupannya saja.
```sh
data = df.filter(['Close'])
data
```
Setelah mendapatkan harga penutupannya saja. Selanjutnya dilakukan pengecekan statistiknya menggunakan fungsi describe yang ada di python.
```sh
data.describe()
```
Selanjutnya adalah mengambil value harga penutupannya saja menjadi sebuah array.


## Data Preparation
---
Data preparation adalah proses mengambil data mentah dan menyiapkannya untuk diserap dalam platform analitik. Untuk mencapai tahap akhir persiapan, data harus dibersihkan, diformat, dan diubah menjadi sesuatu yang dapat dicerna oleh alat analisis. Salah satu fungsi utama data preparation adalah memastikan keakuratan dan konsistensi data mentah yang disiapkan untuk pemrosesan dan analisis.

Ada beberapa tahapan yang dilakukan saat data preparation untuk menyelesaikan proyek ini, yaitu:
### Normalisasi Data
Selanjutnya setelah memilih atribut yang digunakan adalah menormalisasi data yang ada. Normalisasi data adalah proses membuat beberapa variabel memiliki rentang nilai yang sama, tidak ada yang terlalu besar maupun terlalu kecil sehingga dapat membuat analisis statistik menjadi lebih mudah. Metode normalisasi data yang digunakan untuk melakukan proyek ini adalah metode MinMax. Perhitungan normalisasi minmax dilakukan dengan cara setiap nilai pada sebuah data akan dikurangi dengan nilai terkecil data tersebut. Kemudian dibagi dengan nilai yang terbesar dikurangi dengan nilai yang terkecil dari data tersebut. Normalisasi minmax akan menghasilkan nilai yang baru dengan rentang angka antara 0 hingga 1. Berikut adalah code untuk melakukan normalisasi minmax.
```sh
sc = MinMaxScaler(feature_range=(0, 1))
scaled_data = sc.fit_transform(data_len)
scaled_data
```
- variabel sc untuk menampung dari fungsi normalisasi minmax yang nilainya dari 0 hingga 1
- scaled data untuk menampung data hasil yang telah dilakukan normalisasi dengan mengtransform dari data len yang telah dibuat sebelumnya.

| Penutupan Harga | Normalisasi Data |
| ------ | ------ |
| 675 | 0.00520833 |
| 685 | 0.0061553 |
| 690 | 0.00662879 |
| ... | ... |
| 4590 | 0.37594697 |
| 4540 | 0.37121212 |
| 4550 | 0.37215909 |

### Pembagian Dataset
Prose awal saat melakukan pembangunan model machine learning adalah melakukan pembagian data menjadi data latih dan data uji yang sudah melalui proses normalisasi data. Data latih/uji merupakan suatu metode data yang akan digunakan untuk melakukan pelatihan model machine learning hingga untuk mengevaluasi performa dari model tersebut. Proporsi pembagian data ini tergantung dari banyaknya data yang dimiliki. Sering kali pembagian data yang dilakukan memiliki proporsi data latih lebih banyak dibandingkan data testing. Karena data latih akan digunakan untuk melakukan pelatihan untuk algoritma machine learning. Sedangkan untuk data testing digunakan untuk melakukan pengujian performa dari pemodelan yang telah dilakukan sebelumnya hingga menemukan data baru yang belum pernah dilihat sebelumnya. Pembagian data yang digunakan memiliki proporsi sebesar 80:20. Data latih yang digunakan sebesar 80% sedangkan untuk data uji sebesar 20%. Berikut code untuk melakukan pembagian dataset. Berikut merupakan hasil pembagian dataset yang telah dilakukan
- Data Training yang didapatkan yaitu sebesar 3779 data.
- Data testing yang didapatkan yaitu sebesar 945 data.

Setelah mendaptkan masing-masing pembagian data, selanjutnya dilakukan visualisasi dari hasil pembagian.
![Visualisasi dari hasil pembagian dataset](https://user-images.githubusercontent.com/56246122/188427737-8542cc02-1a20-46a1-8d88-ebc2e232ff89.png) 

## Modeling
---
### Pembuatan Parameter untuk modeling
Sebelum Komparasi pembuatan 2 model algoritma Deep Learning, dilakukan pembuatan parameter yang ingin digunakan. Pembuatan parameter diawal supaya komparasi yang dilakukan dengan parameter yang sama semua. Berikut Parameter yang digunakan untuk pemodelan Deep Learning.
```sh
timestep = 60
unit_neuron = 50
epoch = 100
batch = 32
```
1. Timesteps : Timesteps adalah banyaknya jumlah timesteps sudah didefinisikan sebelumnya. banyaknya pola waktu yang digunakan untuk mempelajari model machine learning serta memiliki fungsi untuk melakukan pembagian dataset pada tiap – tiap segmen.Timestep yang digunakan untuk proyek ini yakni 60.
2. Banyaknya jumlah neuron dengan variabel units=50 pada masing-masing hidden layer. Semakin banyak neuron per-lapisan, akan meningkatkan spesialisasi model untuk melatih data. Namun jumlah neuron juga akan berbeda pada tiap penelitian bergantung pada data yang dipakai.
3. Jumlah epoch adalah hyperparameter yang menentukan berapa kali algoritma pembelajaran akan bekerja mengolah seluruh dataset training. Satu epoch berarti bahwa setiap sampel dalam dataset training memiliki kesempatan untuk memperbarui parameter model internal. Jumlah epochsyang digunakan adalah 100.
4. Ukuran batch adalah hyperparameter yang menentukan jumlah sampel untuk dikerjakan sebelum memperbarui parameter model internal. Batch size adalah banyaknya data sampel yang digunakan sebagai input untuk melatih model. Jumlah batch Size untuk proyek ini adalah 32.

Jumlah parameter yang diinisialisasi pada penelitian ini merupakan rujukan dari beberapa parameter penelitian yang telah dilakukan terdahulu dengan metode yang sama. Penentuan parameter dari algoritma LSTM tidak memiliki aturan yang tetap, sehingga diharapkan untuk melakukan percobaan berulang-ulang untuk mendapatkan hasil yang optimal. Parameter – parameter yang sudah dilakukan oleh penelitian sebelumnya dapat digunakan untuk sebagai acuan, tetapi perlu diperhatikan juga untuk selalu melakukan berbagai percobaan hingga mendapatkan hasil yang optimal.

### Pengenalan Algoritma RNN dan LSTM
Recurrent Neural Networks (RNN) merupakan salah satu bentuk arsitektur Artificial Neural Networks (ANN) yang dirancang khusus untuk memproses data yang bersambung/ berurutan (sequential data). RNN biasanya digunakan untuk menyelesaikan tugas yang terkait dengan data time series, misalnya data ramalan cuaca. Sebagai contoh, cuaca hari ini dapat bergantung pada cuaca hari sebelumnya, jika hari sebelumnya mendung, maka kemungkinan hari ini akan hujan. RNN tidak membuang begitu saja informasi dari masa lalu dalam proses pembelajarannya. Hal inilah yang membedakan RNN dari ANN biasa. RNN mampu menyimpan memori/ ingatan (feedback loop) yang memungkinkan untuk mengenali pola data dengan baik, kemudian menggunakannya untuk membuat prediksi yang akurat. Cara yang dilakukan RNN untuk dapat menyimpan informasi dari masa lalu adalah dengan melakukan looping di dalam arsitekturnya, yang secara otomatis membuat informasi dari masa lalu tetap tersimpan.

Long short term memory network (LSTM) adalah salah satu modifikasi dari recurrent neural network atau RNN. Banyak modifikasi dari RNN, tetapi LSTM merupakan salah satu yang populer di antaranya. LSTM hadir untuk melengkapi kekurangan RNN yang tidak dapat memprediksi kata berdasarkan informasi lampau yang disimpan dalam jangka waktu lama. Dengan demikian, LSTM mampu mengingat kumpulan informasi yang telah disimpan dalam jangka waktu panjang, sekaligus menghapus informasi yang tidak lagi relevan. LSTM lebih efisien dalam memproses, memprediksi, sekaligus mengklasifikasikan data berdasarkan urutan waktu tertentu.

Perbedaan mendasar dari LSTM dan RNN adalah bahwa LSTM melengkapi kekurangan-kekurangan yang dimiliki oleh pendahulunya, recurrent neural network, yang tidak dapat memprediksi data berdasarkan informasi yang telah disimpan dalam waktu cukup lama. Dengan kata lain, persoalan jangka waktu penyimpanan tidak menjadi permasalahan dalam LSTM. Sistem yang menerapkan LSTM dapat memproses, memprediksi, dan mengklasifikasikan informasi berdasarkan data deret waktu. Sesuai dengan konsepnya, LSTM dapat mengingat dan menghapus data-data lawas yang sudah tidak relevan lagi. Dengan demikian, manajemen informasi akan lebih komplet sekaligus aktual.

LSTM memiliki beberapa gerbang yang memiliki fungsi dan tugasnya masing-masing. Berikut penjelasan singkat mengenai struktur LSTM.
1. FORGET GATE
Gerbang pertama dalam LSTM disebut dengan forget gate. Mudahnya, gerbang ini bertugas untuk melupakan beberapa informasi yang tidak relevan dan sudah tidak diperlukan oleh sebuah sistem. Alhasil, LSTM dapat menyajikan kumpulan informasi yang lengkap, tetapi tetap aktual sesuai dengan kebutuhan.
2. INPUT GATE
Berikutnya, ada gerbang kedua, yakni input gate yang bertugas untuk memasukkan informasi yang berguna untuk mendukung keakuratan data. Tugas input gate adalah untuk menambahkan informasi yang sebelumnya telah diseleksi terlebih dahulu melalui gerbang forget gate. Gerbang ini tidak dimiliki oleh RNN yang hanya memungkinkan satu input data untuk satu output data.
Dalam input gate kemudian dikenal istilah input modulation gate yang sering tidak ditulis dalam beberapa ulasan tentang LSTM. Sesuai namanya, input modulation gate berfungsi untuk memodulasi informasi yang ada, sehingga dapat mengurangi kecepatan konvergensi dari data zero-mean.
3. OUTPUT GATE
Terakhir adalah output gate yang menjadi gerbang terakhir untuk menghasilkan informasi data yang komplet dan aktual. Gerbang ini bisa menjadi yang terakhir atas sebuah informasi atau hanya menjadi bagian dari tahap pertama saja, sebelum akhirnya informasi akan diproses lewat input gate di sel berikutnya.

##### Model LSTM
Setelah mengetahui tentang algoritma LSTM, selanjutnya pembuatan model dengan algoritma LSTM. Berikut ini merupakan hasil dari pemodelan algoritma LSTM dengan parameter yang telah ditentukan.
![Visualisasi model LSTM](https://user-images.githubusercontent.com/56246122/188428024-af06a902-326a-413f-9f22-b059da44de85.png) 

##### Model RNN
Setelah membuat model dengan algoritma LSTM, selanjutnya membuat model dengan algoritma RNN. Berikut ini merupakan hasil dari pemodelan algoritma RNN dengan parameter yang telah ditentukan.
![Visualisasi model RNN](https://user-images.githubusercontent.com/56246122/188428032-f2c5921f-641d-4825-a4f2-b8e2c20a126b.png) 

Berikut ini penjelasan tentang pemodelan yang telah dibuat
1. Masing – masing hidden layer memiliki jumlah neuron sebanyak 50 yang ditunjukkan pada variabel units = 50. Semakin banyak neuron yang diterapkan per lapisan, makan akan meningkatkan spesialisasi model untuk melatih data.
2. Perintah return_sequences berarti hasil dari LSTM akan berlanjut pada layer LSTM berikutnya, sehingga nilai defaultnya adalah False. Jika Anda ingin menambahkan lapisan LSTM lain, itu harus diubah menjadi True.
3. Parameter lainnya adalah input shape yang cukup melakukan inisialisasi 2 dimensi yang merupakan representasi dari (timestep, feature).
4. Layer dropout yang akan menonaktifkan sebagian node pada layer dense secara acak di setiap putaran epochs. Nilai dari droupout berkisar antara 0 hingga 1, namun yang sering digunakan antara 0,2 sampai 0,5. Jika nilai mendekati nilai 0 maka ankan cenderung mengalami overfitting. Sebaliknya jika mendekati nilai 1 maka memiliki risiko underfitting.

## Evaluasi
---
Ada dua alasan utama untuk mempertimbangkan keakuratan prediksi model deret waktu. Pertama, menggunakan pengukuran akurasi untuk proses pengembangan model serta fase definisi untuk membandingkan model alternatif serta menetapkan nilai parameter yang ditampilkan dalam hasil fungsi prediksi. Untuk menetapkan model prediksi yang terbaik dengan mempertimbangkan setiap model yang diterapkan pada data historis dan pilih model dengan nilai kesalahan total yang paling sedikit. Model yang dianggap valid untuk data historis dan memiliki nilai kesalahan total terendah dipilih.
Kedua, ketika model peramalan dikembangkan dan digunakan untuk menghasilkan ramalan masa depan, akurasinya secara teratur disesuaikan untuk mengidentifikasi anomali dan cacat pada model prediksi pola yang mungkin perlu dievaluasi. Mengevaluasi keakuratan ramalan pada titik ini akan membantu menentukan apakah model tersebut masih benar dengan menggunakan perhitungan Mean Absolute error (MAE) atau root mean square error (RMSE) dan mean absolute persen error (MAPE).
Model peramalan dikembangkan dan digunakan untuk menghasilkan ramalan masa depan, akurasinya secara teratur disesuaikan untuk mengidentifikasi anomali dan cacat pada model prediksi pola yang mungkin perlu dievaluasi. Mengevaluasi keakuratan ramalan pada titik ini akan membantu menentukan apakah model tersebut masih benar dengan menggunakan perhitungan Mean Absolute error (MAE) dan mean absolute persen error (MAPE).
1. MAE atau Mean Absolute Error menunjukkan nilai kesalahan rata-rata yang error dari nilai sebenarnya dengan nilai prediksi. MAE sendiri secara umum digunakan untuk pengukuran prediksi error pada analisis time series.
2. Mean Absolut Percentage error (MAPE) adalah persentase kesalahan rata-rata secara multak.(absolut). Pengertian Mean Absolute Percentage Error adalah Pengukuran statistik tentang akurasi perkiraan (prediksi) pada metode peramalan. Pengukuran dengan menggunakan Mean Absolute Percentage Error (MAPE) dapat digunakan oleh masyarakat luas karena MAPE mudah difahami dan diterapkan dalam memprediksi akurasi peramalan. Metode Mean Abosolute Percentage Error (MAPE) memberikan informasi seberapa besar kesalahan peramalan dibandingkan dengan nilai sebenarnya dari series tersebut. Semakin kecil nilai presentasi kesalahan (percentage error) pada MAPE maka semakin akurat hasil peramalan tersebut.

| Range MAPE | Arti Nilai |
| ------ | ------ |
| < 10% | Kemampuan model peramalan sangat baik |
| 10% - 20% | Kemampuan model peramalan sangat baik |
| 20% - 50% | Kemampuan model peramalan sangat layak |
| > 50% | Kemampuan model peramalan sangat buruk |

### Tabel perbandingan algoritma LSTM dan RNN
Selanjutnya adalah menampilkan hasil dari model yang telah dilakukan serta dibandingkan dari harga asli, prediksi LSTM dan prediksi RNN. Berikut code yang digunakan.
```sh
predictionLSTM = np.array(predicted_LSTM)
predictionRNN = np.array(predicted_RNN)
df_hasil = pd.DataFrame()
df_hasil['Real Price'] = df_test['Close'].reset_index(drop = True)
df_hasil['Prediction LSTM'] = predictionLSTM
df_hasil['Prediction RNN'] = predictionRNN
df_hasil
```

Berikut ini adalah hasil dari perbandingan yang sudah dilakukan
| Real Price | Prediksi LSTM | Prediksi RNN |
| ------ | ------ | ------ |
| 8065.0 | 8734.052734 | 8775.239258 |
| 7875.0 | 8613.247070 | 8560.471680 |
| 7990.0 | 8457.843750 | 8351.792969 |
| 8000.0 | 8358.924805 | 8358.291016 |
| ... | .... | .... |
| 4580.0 | 4876.233398 | 4668.181641 |
| 4590.0 | 4805.312012 | 4656.586914 |
| 4540.0 | 4769.444336 | 4658.923828 |
| 4550.0 | 4752.182129 | 4629.797363 |

##### Evaluasi MAE
Selanjutnya melakukan evaluasi dengan menggunakan perhitungan MAE. Berikut hasil dari kedua algoritma untuk MAE.
1. Nilai MAE yang didapatkan untuk algoritma LSTM adalah sebesar 231.1011
2. Nilai MAE yang didapatkan untuk algoritma RNN adalah sebesar 233.8674.

Hasil evaluasi MAE dari kedua algoritma menyimpulkan bahwa algoritma LSTM memiliki hasil prediksi yang lebih bagus dengan nilai MAE yang lebih kecil yaitu sebesar 231.1011. 
##### Evaluasi MAPE
Selanjutnya melakukan evaluasi dengan menggunakan perhitungan MAE. Berikut merupakan hasil dari kedua algoritma untuk MAPE.
Dari hasil evaluasi MAPE yang telah dilakukan menghasilkan
1. Nilai MAPE yang didapatkan untuk algoritma LSTM adalah sebesar 3.303%
2. Nilai MAPE yang didapatkan untuk algoritma RNN adalah sebesar 3.646%

Dari hasil MAPE yang didapatkan dapat disimulkan bahwa algoritma LSTM lebih bagus untuk melakukan prediksi harga saham dibandingkan dengan algoritma RNN dengan hasil evaluasi MAPE sebesar 3,303%. Tetapi walaupun algoritma LSTM lebih bagus, kedua algoritma menghasilkan evaluasi MAPE dengan nilai dibawah 10%. Jadi dapat disimpulkan juga bahwa kedua algoritma tersebut dapat melaukan prediksi harga saham dengan sangat baik.
Berikut merupakan hasil evaluasi untuk perbandingan kedua algoritma dengan harga aktualnya.
![Visualisasi model RNN dan LSTM](https://user-images.githubusercontent.com/56246122/188428038-1042a730-1731-42a9-a3e7-da4c2514eb6e.png) 

## Kesimpulan
---
Dari hasil kedua algoritma yang telah dilakukan disimpulkan bahwa nilai dari MAPE keduanya menunjukan hasil prediksi yang sangat baik. Hasil MAPE dari algoritma LSTM didapatkan dengan nilai 3.303% dan nilai MAE sebesar 231.1011. Sedangkan algoritma RNN mendapatkan nilai MAPE sebesar 3.646% dan nilai MAE sebesar 233.8674. Hasil keduanya mengartikan bahwa model peramalan peramalan yang dilakukan menunjukan hasil yang sangat baik.

Berikut profil saya:

- [Github](https://github.com/Asholihin1705) - Profil Github
- [Linkedin](https://www.linkedin.com/in/ahmadsholihin/) - Profil Linkedin
- [Medium](https://medium.com/@ahmadsholihin1705) - Profil Medium

