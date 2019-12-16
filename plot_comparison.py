import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--comp_dir', type=str, default='result_ref', help="directory to compare")
opt = parser.parse_args()

refdir = 'result_ref'
compdir = opt.comp_dir

fl1 = sorted(os.listdir(refdir))
fl2 = sorted(os.listdir(compdir))
i = 0
sz = len(fl2)
print(fl2)

f, axarr = plt.subplots(sz, 3, figsize=(15, 5))

for f2 in fl2: 
    if '.png' in f2 and '_input' not in f2:
        img1 = mpimg.imread(os.path.join(compdir, f2[:-4] + '_input.png'))
        img2 = mpimg.imread(os.path.join(refdir, f2))
        img3 = mpimg.imread(os.path.join(compdir, f2))

        axarr[i,0].imshow(img1)
        axarr[i,1].imshow(img2)
        axarr[i,2].imshow(img3)
        i += 1
        if i == sz:
            break
f.savefig("result.png", dpi=f.dpi)