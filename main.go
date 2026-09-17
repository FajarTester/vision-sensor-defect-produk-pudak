package main

import (
	"fmt"
	"image"
	"image/color"
	"log"

	"gocv.io/x/gocv"
)

func main() {
	// 1. Buka Kamera Utama (0 adalah indeks webcam bawaan laptop)
	webcam, err := gocv.OpenVideoCapture(0)
	if err != nil {
		log.Fatalf("Gagal membuka kamera: %v", err)
	}
	defer webcam.Close()

	// 2. Buka Jendela GUI untuk Menampilkan Video
	window := gocv.NewWindow("Golang YOLO Camera")
	defer window.Close()

	// 3. Muat Model YOLO (Format ONNX) menggunakan modul DNN OpenCV
	net := gocv.ReadNetFromONNX("yolov8n.onnx")
	if net.Empty() {
		log.Fatal("Gagal memuat model YOLO ONNX")
	}
	defer net.Close()

	// Gunakan CPU untuk pemrosesan (bisa diubah ke GPU jika OpenCV Anda mendukung CUDA)
	net.SetPreferableBackend(gocv.NetBackendDefault)
	net.SetPreferableTarget(gocv.NetTargetCPU)

	// Siapkan matriks kosong untuk menampung frame gambar dari kamera
	img := gocv.NewMat()
	defer img.Close()

	fmt.Println("Menjalankan kamera... Tekan tombol apa saja pada jendela video untuk keluar.")

	for {
		// 4. Baca frame terbaru dari kamera
		if ok := webcam.Read(&img); !ok || img.Empty() {
			fmt.Println("Gagal membaca frame dari kamera")
			break
		}

		// 5. Pre-processing: Ubah frame kamera menjadi "Blob" yang dimengerti YOLO
		// YOLOv8 membutuhkan ukuran 640x640, skala warna dikali 1/255.0, tanpa swap RGB
		blob := gocv.BlobFromImage(img, 1.0/255.0, image.Pt(640, 640), gocv.NewScalar(0, 0, 0, 0), true, false)
		net.SetInput(blob, "images")

		// 6. Jalankan Inference (Prediksi)
		output := net.Forward("output0")

		// 7. Proses Hasil Deteksi (Post-processing sederhana)
		// Catatan: Output YOLOv8 berbentuk matriks [1][84][8400].
		// Di bawah ini adalah contoh logika dasar untuk menggambar kotak jika mendeteksi sesuatu.
		if !output.Empty() {
			// Kita ambil dimensi matriks output
			size := output.Size() // size[1] = 84, size[2] = 8400
			
			// Untuk demo sederhana, kita cek apakah ada objek dengan confidence tinggi.
			// Di aplikasi produksi, Anda harus melakukan loop pada baris matriks ini
			// untuk mengekstrak koordinat X, Y, W, H dan menerapkan Non-Maximum Suppression (NMS).
			
			// CONTOH: Menggambar kotak simulasi di tengah layar jika model aktif merespon
			if size[2] > 0 {
				pt1 := image.Pt(100, 100)
				pt2 := image.Pt(400, 400)
				// Gambar kotak hijau (RGB: 0, 255, 0) dengan ketebalan 3
				gocv.Rectangle(&img, image.Rectangle{Min: pt1, Max: pt2}, color.RGBA{0, 255, 0, 0}, 3)
				gocv.PutText(&img, "Objek Terdeteksi", image.Pt(100, 90), gocv.FontHersheySimplex, 0.8, color.RGBA{0, 255, 0, 0}, 2)
			}
		}
		output.Close()
		blob.Close()

		// 8. Tampilkan frame yang sudah diproses ke jendela GUI
		window.IMShow(img)

		// Tunggu 1 milidetik, jika ada tombol ditekan, keluar dari aplikasi
		if window.WaitKey(1) >= 0 {
			break
		}
	}
}
