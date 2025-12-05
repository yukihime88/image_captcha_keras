import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import random
import math
from scipy.ndimage import rotate, shift, zoom
from skimage.util import random_noise
from skimage.transform import swirl
from skimage.filters import gaussian
import imgaug.augmenters as iaa
import warnings
warnings.filterwarnings('ignore')

# Menghilangkan warning TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Menghilangkan informasi TF
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# --- KONFIGURASI DAN KONSTANTA ---
# Sesuaikan berdasarkan karakteristik CAPTCHA Anda
IMAGE_WIDTH = 200     # Lebar target
IMAGE_HEIGHT = 100     # Tinggi target
CAPTCHA_LENGTH = 6    # Jumlah karakter dalam setiap CAPTCHA
CHARACTERS = "0123456789" # Set karakter yang mungkin
NUM_CLASSES = len(CHARACTERS)
MAPPING = {char: i for i, char in enumerate(CHARACTERS)}
FOLDER_PATH = 'Samples/'
MODEL_NAME = 'best_captcha_solver_model.keras'

# --- 1. FUNGSI ADAPTIVE THRESHOLD DITINGKATKAN (Dari realtime_fine_tuning.py) ---
def apply_adaptive_threshold(img_gray, method='gaussian', block_size=11, C=2, blur_size=3):
    """
    Terapkan binerisasi adaptif dengan parameter yang dapat disesuaikan
    """
    # Preprocessing dengan blur untuk mengurangi noise
    if blur_size > 0:
        img_blur = cv2.medianBlur(img_gray, blur_size)
    else:
        img_blur = img_gray.copy()

    # Pilih metode adaptive threshold
    if method == 'gaussian':
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    else:  # mean
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C

    # Terapkan adaptive threshold
    img_threshold = cv2.adaptiveThreshold(
        img_blur, 255,
        adaptive_method,
        cv2.THRESH_BINARY_INV,
        block_size, C
    )

    return img_threshold

def apply_improved_threshold(img_gray):
    """
    Terapkan binerisasi adaptif yang ditingkatkan dengan parameter optimal
    untuk deteksi angka/digit yang lebih baik
    """
    # Parameter yang dioptimalkan khusus untuk deteksi digit/angka
    block_size = 19  # Ukuran blok yang lebih besar untuk menangani variasi ukuran digit
    C = 6            # Konstanta yang lebih tinggi untuk kontras yang lebih tajam
    blur_size = 3    # Blur yang lebih ringan untuk menjaga detail digit

    # Coba kedua metode dan pilih yang terbaik untuk deteksi digit
    img_gaussian = apply_adaptive_threshold(img_gray, 'gaussian', block_size, C, blur_size)
    img_mean = apply_adaptive_threshold(img_gray, 'mean', block_size, C, blur_size)

    # Analisis khusus untuk deteksi digit - fokus pada jumlah piksel putih yang optimal
    white_pixels_gaussian = cv2.countNonZero(img_gaussian)
    white_pixels_mean = cv2.countNonZero(img_mean)

    # Untuk deteksi digit, kita ingin jumlah piksel putih yang optimal (tidak terlalu banyak noise, tidak terlalu sedikit detail)
    # Target sekitar 30-50% piksel putih untuk hasil terbaik
    target_white_ratio = 0.4
    target_pixels = int(img_gray.size * target_white_ratio)

    # Pilih metode yang paling dekat dengan target optimal
    if abs(white_pixels_gaussian - target_pixels) < abs(white_pixels_mean - target_pixels):
        return img_gaussian
    else:
        return img_mean

def apply_digit_enhancement(img_gray):
    """
    Terapkan peningkatan khusus untuk deteksi digit dengan morphological operations
    """
    # Terapkan threshold yang dioptimalkan untuk digit
    img_threshold = apply_improved_threshold(img_gray)

    # Morphological operations untuk membersihkan noise dan memperjelas digit
    kernel = np.ones((2, 2), np.uint8)

    # Dilasi untuk menutup celah kecil dalam digit
    img_dilated = cv2.dilate(img_threshold, kernel, iterations=1)

    # Erosi untuk menghilangkan noise kecil
    img_eroded = cv2.erode(img_dilated, kernel, iterations=1)

    # Tambahkan sedikit blur untuk menghaluskan tepi
    img_smoothed = cv2.medianBlur(img_eroded, 3)

    return img_smoothed

# --- 1. FUNGSI PEMUATAN DAN PRA-PEMROSESAN DATA ---
def load_data(folder_path):
    """
    Memuat gambar CAPTCHA dan mengkonversi label nama file ke format one-hot.
    """
    images = []
    labels = []

    # Memeriksa apakah direktori Samples/ ada
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' tidak ditemukan. Pastikan gambar ada di folder tersebut.")
        return None, None

    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            # Mendapatkan Label dari Nama File (misalnya '011685.jpg' -> '011685')
            label_str = filename.split('.')[0]
            if len(label_str) != CAPTCHA_LENGTH:
                print(f"Warning: Melewatkan file '{filename}' karena panjang label tidak sesuai.")
                continue

            # Memuat dan Pra-pemrosesan Gambar
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            # Konversi ke Grayscale (Penting untuk mengatasi warna yang mengganggu)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Resize ke ukuran standar (Normalisasi Ukuran)
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

            # Terapkan adaptive threshold yang ditingkatkan untuk deteksi digit
            img = apply_digit_enhancement(img)

            # Normalisasi Intensitas Pixel (0.0 hingga 1.0)
            img = img / 255.0

            images.append(img)

            # Encoding Label (One-Hot Encoding untuk Setiap Posisi Karakter)
            # Label berbentuk (CAPTCHA_LENGTH, NUM_CLASSES)
            one_hot_label = np.zeros((CAPTCHA_LENGTH, NUM_CLASSES), dtype=np.uint8)
            for i, char in enumerate(label_str):
                if char in MAPPING:
                    one_hot_label[i, MAPPING[char]] = 1
                else:
                    # Handle jika ada karakter di luar set yang ditentukan
                    print(f"Warning: Karakter '{char}' pada file '{filename}' tidak ada dalam set CHARACTERS.")
                    one_hot_label = None
                    break

            if one_hot_label is not None:
                labels.append(one_hot_label)

    X = np.array(images)[..., np.newaxis] # Bentuk: (N, H, W, 1)
    y = np.array(labels)

    # Memisahkan output untuk setiap posisi karakter (sesuai kebutuhan Multi-Output Keras)
    y_list = [y[:, i, :] for i in range(CAPTCHA_LENGTH)]

    return X, y_list

# --- 1.5. FUNGSI DATA AUGMENTATION LANJUTAN ---
def advanced_augmentation(images, labels_list, augmentation_factor=10):
    """
    Menerapkan berbagai teknik augmentasi untuk memaksimalkan data pelatihan.
    """
    augmented_images = []
    augmented_labels = []

    print(f"Memulai augmentasi data dengan faktor {augmentation_factor}x...")

    # Inisialisasi augmentor imgaug
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential([
        # Apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Crop(percent=(0, 0.1)), # random crops

        # Strengthen or weaken the contrast in each image
        iaa.LinearContrast((0.75, 1.5)),

        # Add gaussian noise
        sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),

        # Make some images brighter and some darker
        iaa.Multiply((0.8, 1.2), per_channel=0.2),

        # Apply affine transformations
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        ))
    ], random_order=True)

    for i in range(len(images)):
        original_img = images[i]

        # Gabungkan label dari list menjadi array lengkap
        original_label = np.stack([labels_list[j][i] for j in range(CAPTCHA_LENGTH)], axis=0)

        # Tambahkan gambar asli
        augmented_images.append(original_img)
        augmented_labels.append(original_label)

        # Buat augmentasi tambahan
        for _ in range(augmentation_factor - 1):
            # Konversi ke format yang sesuai untuk imgaug
            img_aug = (original_img.squeeze() * 255).astype(np.uint8)

            # Terapkan augmentasi
            img_aug = seq(image=img_aug)

            # Konversi kembali ke format model
            img_aug = img_aug.astype(np.float32) / 255.0
            img_aug = np.expand_dims(img_aug, axis=-1)

            augmented_images.append(img_aug)
            augmented_labels.append(original_label)

    print(f"Augmentasi selesai. Data bertambah dari {len(images)} menjadi {len(augmented_images)} sampel.")

    # Konversi ke numpy arrays
    X_aug = np.array(augmented_images)
    y_aug = np.array(augmented_labels)

    # Memisahkan output untuk setiap posisi karakter
    y_aug_list = [y_aug[:, i, :] for i in range(CAPTCHA_LENGTH)]

    return X_aug, y_aug_list

# --- 1.6. FUNGSI AUGMENTASI KUSTOM TAMBAHAN ---
def custom_augmentation(images, labels_list, factor=5):
    """
    Augmentasi kustom tambahan untuk variasi yang lebih spesifik.
    """
    augmented_images = []
    augmented_labels = []

    for i in range(len(images)):
        original_img = images[i]

        # Gabungkan label dari list menjadi array lengkap
        original_label = np.stack([labels_list[j][i] for j in range(CAPTCHA_LENGTH)], axis=0)

        # Tambahkan gambar asli
        augmented_images.append(original_img)
        augmented_labels.append(original_label)

        # Buat variasi augmentasi
        for _ in range(factor):
            img = original_img.squeeze().copy()

            # Pilih augmentasi secara acak
            aug_type = random.choice([
                'rotate', 'shift', 'zoom', 'noise', 'blur', 'swirl', 'contrast'
            ])

            if aug_type == 'rotate':
                angle = random.uniform(-15, 15)
                img = rotate(img, angle, reshape=False, mode='reflect')

            elif aug_type == 'shift':
                shift_x = random.uniform(-5, 5)
                shift_y = random.uniform(-5, 5)
                img = shift(img, [shift_y, shift_x], mode='reflect')

            elif aug_type == 'zoom':
                zoom_factor = random.uniform(0.9, 1.1)
                img = zoom(img, zoom_factor)

            elif aug_type == 'noise':
                img = random_noise(img, mode='gaussian', var=0.001)

            elif aug_type == 'blur':
                sigma = random.uniform(0.5, 1.5)
                img = gaussian(img, sigma=sigma)

            elif aug_type == 'swirl':
                strength = random.uniform(5, 15)
                radius = random.uniform(20, 50)
                img = swirl(img, strength=strength, radius=radius)

            elif aug_type == 'contrast':
                contrast_factor = random.uniform(0.8, 1.2)
                img = np.clip(img * contrast_factor, 0, 1)

            # Pastikan ukuran tetap konsisten
            if img.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
                img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

            # Normalisasi dan tambahkan dimensi channel
            img = np.clip(img, 0, 1)
            img = np.expand_dims(img, axis=-1)

            augmented_images.append(img)
            augmented_labels.append(original_label)

    # Konversi ke numpy arrays
    X_aug = np.array(augmented_images)
    y_aug = np.array(augmented_labels)

    # Memisahkan output untuk setiap posisi karakter
    y_aug_list = [y_aug[:, i, :] for i in range(CAPTCHA_LENGTH)]

    return X_aug, y_aug_list

# --- 2. MEMBANGUN MODEL CNN MULTI-OUTPUT (DITINGKATKAN) ---
def create_cnn_model(width, height, num_classes, length):
    """
    Membangun model CNN yang ditingkatkan dengan arsitektur yang lebih dalam
    dan teknik regularisasi untuk performa yang lebih baik.
    """
    # Input Layer (H, W, 1 channel untuk Grayscale)
    input_layer = Input(shape=(height, width, 1), name='captcha_input')

    # Blok Konvolusi yang Ditingkatkan dengan Batch Normalization
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Blok Konvolusi Tambahan untuk Fitur yang Lebih Dalam
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    # Flatten dan Shared Dense Layer yang Ditingkatkan
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Kepala Output (Output Heads) dengan Learning Rate yang Disesuaikan
    output_layers = []
    for i in range(length):
        # Softmax digunakan karena setiap posisi adalah masalah klasifikasi multi-kelas (0-9)
        name = f'output_{i+1}'
        output_head = Dense(num_classes, activation='softmax', name=name)(x)
        output_layers.append(output_head)

    model = Model(inputs=input_layer, outputs=output_layers)

    # Kompilasi dengan Optimizer yang Ditingkatkan
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# --- 2.1. FUNGSI PEMBANGUN MODEL ALTERNATIF (LEBIH RINGAN) ---
def create_lightweight_model(width, height, num_classes, length):
    """
    Model alternatif yang lebih ringan untuk pelatihan yang lebih cepat.
    """
    input_layer = Input(shape=(height, width, 1), name='captcha_input')

    # Arsitektur yang Lebih Ringan
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    output_layers = []
    for i in range(length):
        name = f'output_{i+1}'
        output_head = Dense(num_classes, activation='softmax', name=name)(x)
        output_layers.append(output_head)

    model = Model(inputs=input_layer, outputs=output_layers)

    # Optimizer dengan Learning Rate yang Lebih Konservatif
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# --- 3. FUNGSI PREDIKSI ---
def solve_captcha(image_path, model):
    """
    Memuat model tersimpan dan memecahkan satu gambar CAPTCHA baru.
    """
    
    # Pra-pemrosesan Gambar Baru (Harus sama persis dengan saat pelatihan)
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Gagal memuat gambar."

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # Terapkan adaptive threshold yang ditingkatkan untuk deteksi digit (sama dengan pelatihan)
    img = apply_digit_enhancement(img)

    # Normalisasi Intensitas Pixel (0.0 hingga 1.0)
    img = img / 255.0

    # Ubah bentuk array untuk input model (1, H, W, 1)
    img = np.expand_dims(img, axis=0)  # Tambahkan dimensi batch
    img = np.expand_dims(img, axis=-1) # Tambahkan dimensi channel

    # Prediksi
    predictions = model.predict(img, verbose=0)

    # Mengubah Output One-Hot ke String Karakter
    predicted_captcha = ""
    for i in range(CAPTCHA_LENGTH):
        # Ambil indeks kelas dengan probabilitas tertinggi untuk setiap posisi
        char_index = np.argmax(predictions[i][0])
        # Konversi indeks kembali ke karakter (0-9)
        predicted_captcha += CHARACTERS[char_index]

    return predicted_captcha

# --- 4. EKSEKUSI UTAMA ---
if __name__ == '__main__':
    print("--- 1. MEMUAT DATA ---")
    X, y_list = load_data(FOLDER_PATH)

    if X is None:
        print("Proses dihentikan karena kegagalan memuat data.")
    else:
        print(f"Data Asli: {X.shape[0]} sampel")

        # Terapkan augmentasi data lanjutan
        print("\n--- 1.5. AUGMENTASI DATA LANJUTAN ---")
        X_aug, y_aug_list = advanced_augmentation(X, y_list, augmentation_factor=15)

        # Terapkan augmentasi kustom tambahan
        print("\n--- 1.6. AUGMENTASI KUSTOM TAMBAHAN ---")
        X_final, y_final_list = custom_augmentation(X_aug, y_aug_list, factor=3)

        # Membagi Data (dengan data yang sudah di-augmentasi)
        X_train, X_val, *y_split = train_test_split(
            X_final, *y_final_list, test_size=0.15, random_state=42
        )
        # Mengembalikan y_split ke format list of arrays
        y_train_list = y_split[::2]
        y_val_list = y_split[1::2]

        print(f"\n--- 1.7. DATA SETELAH AUGMENTASI ---")
        print(f"Total Sampel Setelah Augmentasi: {X_final.shape[0]}")
        print(f"Data Pelatihan: {X_train.shape[0]}")
        print(f"Data Validasi: {X_val.shape[0]}")
        print(f"Peningkatan Data: {X_final.shape[0] / X.shape[0]:.1f}x dari data asli")

        print("\n--- 2. MEMBANGUN DAN MELATIH MODEL ---")
        model = create_cnn_model(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CLASSES, CAPTCHA_LENGTH)
        # model.summary() # Aktifkan untuk melihat detail arsitektur

        # Callbacks dengan penyesuaian untuk data yang lebih besar dan optimasi yang ditingkatkan
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
            # Simpan model menggunakan format SavedModel (.keras)
            ModelCheckpoint(MODEL_NAME, save_best_only=True, monitor='val_loss', verbose=1),
            # Reduce Learning Rate pada Plateau untuk optimasi yang lebih baik
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Pelatihan dengan data yang di-augmentasi dan parameter yang dioptimalkan
        try:
            history = model.fit(
                X_train, y_train_list,
                epochs=300, # Jumlah epoch maksimum yang ditingkatkan
                validation_data=(X_val, y_val_list),
                batch_size=32,  # Batch size yang lebih kecil untuk stabilitas yang lebih baik
                callbacks=callbacks,
                verbose=1,
                shuffle=True,  # Pastikan data diacak untuk pelatihan yang lebih baik
                use_multiprocessing=True,  # Gunakan multiproses untuk performa yang lebih baik
                workers=4  # Gunakan 4 worker untuk pemrosesan paralel
            )
            print(f"\nPelatihan Selesai. Model terbaik disimpan sebagai '{MODEL_NAME}'.")

            # Simpan riwayat pelatihan untuk analisis
            np.save('training_history.npy', history.history)

        except Exception as e:
            print(f"\nTerjadi kesalahan selama pelatihan: {e}")

        print("\n--- 3. PENGUJIAN PREDIKSI ---")
        try:
            # Muat model terbaik yang disimpan
            final_model = tf.keras.models.load_model(MODEL_NAME)

            # Uji pada beberapa sampel acak dari data validasi
            test_indices = np.random.choice(X_val.shape[0], min(5, X_val.shape[0]), replace=False)

            print("--- Uji Beberapa Sampel Acak ---")
            correct_predictions = 0

            for random_index in test_indices:
                # Ambil label aslinya
                original_label = "".join([CHARACTERS[np.argmax(y_val_list[i][random_index])] for i in range(CAPTCHA_LENGTH)])

                # Buat file sementara untuk pengujian agar sesuai dengan fungsi solve_captcha
                test_img_array = (X_val[random_index] * 255.0).squeeze().astype(np.uint8)
                temp_test_path = f'temp_test_image_{random_index}.jpg'
                cv2.imwrite(temp_test_path, test_img_array)

                predicted_result = solve_captcha(temp_test_path, final_model)

                print(f"Jawaban Seharusnya: {original_label}")
                print(f"Hasil Prediksi Model: {predicted_result}")
                print(f"Benar: {'✓' if original_label == predicted_result else '✗'}")
                print("-" * 40)

                if original_label == predicted_result:
                    correct_predictions += 1

                # Bersihkan file sementara
                os.remove(temp_test_path)

            accuracy = correct_predictions / len(test_indices) * 100
            print(f"\nAkurasi pada {len(test_indices)} sampel uji: {accuracy:.1f}%")

        except FileNotFoundError:
             print(f"Error: Model file '{MODEL_NAME}' tidak ditemukan. Pastikan pelatihan berhasil.")
        except Exception as e:
            print(f"Terjadi kesalahan saat pengujian: {e}")
