import argparse, concurrent.futures, cv2, glob, os, random
import numpy as np

def trim(img): # サイズ変更
    h, w = img.shape[:2]
    s = min(h, w) # 正方形の一辺の長さ
    if h > w:
        d = random.randint(0, h - s) # ズレの量
        cropped_img = img[d:d+s, 0:s]
    else:
        d = random.randint(0, w - s) # ズレの量
        cropped_img = img[0:s, d:d+s]
    # 正方形にクロップ, 位置はランダム

    resized_img = cv2.resize(cropped_img, (512, 512))
    # 512*512にリサイズ
    
    return resized_img

def denoiser(img): # ノイズ除去
    diameter = 5
    sigma_color = 8
    sigma_space = 8
    radius = 4
    eps = 16
    
    y = img.copy()
    for _ in range(64):
        y = cv2.bilateralFilter(y, diameter, sigma_color, sigma_space)

    for _ in range(4):
        y = cv2.ximgproc.guidedFilter(img, y, radius, eps)
    
    return y

def proc(file, output_dir): # 統合処理
    img = cv2.imread(file)
    trimmed_img = trim(img)
    denoised_img = denoiser(trimmed_img)
    output_path = os.path.join(output_dir, os.path.basename(file))
    print(output_path)
    cv2.imwrite(output_path, denoised_img)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, default="input", help="Input directory")
    parser.add_argument("-o", "--output_dir", type=str, default="output", help="Output directory")
    args = parser.parse_args()
    # 引数の処理

    files = glob.glob(args.input_dir + "/*.jpg")
    files.extend(glob.glob(args.input_dir + "/*.jpeg"))
    files.extend(glob.glob(args.input_dir + "/*.png"))
    files.extend(glob.glob(args.input_dir + "/*.bmp"))
    # 入力ファイルを拡張子でフィルタ

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # 出力フォルダが存在しない場合, 作成する

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count())
    for file in files:
        executor.submit(proc, file, args.output_dir)
    executor.shutdown()
    # 並列処理

if __name__ == "__main__":
    main()
