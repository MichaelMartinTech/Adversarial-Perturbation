import os
from PIL import Image

def find_clean_images(clean_dir):
	print(f"\nScanning {clean_dir}...")
	train_dir = os.path.join(clean_dir, "train")
	if not os.path.isdir(train_dir):
		print("   No train directory.")
		return []

	images = []
	for f in sorted(os.listdir(train_dir)):
		print("   ", f)
		if not f.lower().endswith((".png", ".jpg", ".jpeg")):
			continue

		base = os.path.splitext(f)[0]
		if "_glazed" in base or "_shaded" in base:
			continue

		images.append(os.path.join(train_dir, f))

	print("   Clean images:", len(images))
	return images


def make_collage(images, outfile):
	if not images:
		print("   No images found, skipping collage.")
		return

	loaded = [Image.open(p).convert("RGB") for p in images]

	thumb = 256
	thumbs = []
	for im in loaded:
		im = im.copy()
		im.thumbnail((thumb, thumb))
		thumbs.append(im)

	cols = 7
	rows = 3
	w = cols * thumb
	h = rows * thumb

	collage = Image.new("RGB", (w, h), (255, 255, 255))

	i = 0
	for r in range(rows):
		for c in range(cols):
			if i >= len(thumbs):
				break
			x = c * thumb
			y = r * thumb
			collage.paste(thumbs[i], (x, y))
			i += 1

	collage.save(outfile)
	print("   Saved", outfile)


# --- PROCESS BOTH CLEAN SETS ---

comic_clean = "comic_clean"
materials_clean = "materials_clean"

comic_images = find_clean_images(comic_clean)
make_collage(comic_images, "dataset_visual_comic.png")

materials_images = find_clean_images(materials_clean)
make_collage(materials_images, "dataset_visual_materials.png")

print("\nDone.")