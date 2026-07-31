import urllib.request
import urllib.parse
import os
from pathlib import Path

# Try to import PIL for image format conversion
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

filenames = {
    0: 'Conidiophores_of_corn_gray_leaf_spot_fungus.jpg',
    1: 'Puccinia_sorghi_Schwein._5465563.jpg',
    2: 'Zea_mays_leaf_L.jpg',
    3: 'Northern_corn_leaf_blight.JPG',
    4: 'Good_msv_3.jpg',
    5: 'Bacterial_leaf_spot_of_pepper_%2814954536360%29.jpg',
    6: 'Paprika_Passalora_capsicicola_-1-_Bedlan-Gem%C3%BCse_11-2012.jpg',
    7: 'Alternaria_solani_-_leaf_lesions.jpg',
    8: 'Fusarium_wilt_on_peppers.jpg',
    9: 'Bell_Pepper_Plant_from_Senegal_08.jpg',
    10: 'Late_blight_on_pepper_stem.jpg',
    11: '5411484-PPT-Phytophthora_blight_%28Phytophthora_capsici%29.jpg',
    12: 'Yellow_curl_leaf_disease_Pj_IMG_3162.jpg',
    13: 'Tobacco_mosaic_virus_on_pepper.jpg',
    14: 'Septoria_on_pepper_leaf.jpg',
    15: 'Bacterial_spot_on_tomato_leaves.jpg',
    16: 'Early_blight_on_tomato_leaves_%287871930010%29.jpg',
    17: 'Tomaquera_amb_Fusarium_HV.JPG',
    18: 'Tomato_Greenhouse_in_a_Libyan_Farm.jpg',
    19: 'Tomato_late_blight_leaf_curl_2_%285816171413%29.jpg',
    20: 'Yellow_curl_leaf_disease_Pj_IMG_3162.jpg',
    21: 'Leaf_with_ToMV.jpg',
    22: 'Septoria_lycopersici_on_tomato.jpg'
}

# The target output directory in the project
drawable_dir = Path(__file__).resolve().parent / "app" / "src" / "commonMain" / "composeResources" / "drawable"
drawable_dir.mkdir(parents=True, exist_ok=True)

headers = {'User-Agent': 'CropDiseaseDetectionMobile/1.0 (musahibrahimali@gmail.com) Mozilla/5.0'}

print("Starting download of 23 crop disease reference images...")
print(f"Target directory: {drawable_dir}")
if HAS_PIL:
    print("Pillow is available. JPG files will be converted to true PNG format.")
else:
    print("Pillow is not available. Saving downloaded bytes directly (decoders will still load them).")

for disease_id, filename in filenames.items():
    url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{filename}"
    out_file = drawable_dir / f"disease_{disease_id}.png"
    
    print(f"Downloading disease_{disease_id}.png from {url}...")
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            img_data = response.read()
            
            # Temporary file to save bytes
            temp_path = drawable_dir / f"temp_{disease_id}"
            with open(temp_path, "wb") as f:
                f.write(img_data)
                
            if HAS_PIL:
                try:
                    with Image.open(temp_path) as img:
                        img.save(out_file, "PNG")
                    temp_path.unlink()
                except Exception as e:
                    print(f"  Pillow conversion failed, saving raw bytes directly: {e}")
                    temp_path.rename(out_file)
            else:
                temp_path.rename(out_file)
                
            print(f"  Successfully saved -> disease_{disease_id}.png")
    except Exception as e:
        print(f"  FAILED to download disease_{disease_id}.png: {e}")

print("\nFinished downloading reference images.")
