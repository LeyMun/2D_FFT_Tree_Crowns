from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def normalize_complex_arr(a: np.ndarray) -> np.ndarray:
    # normalise to -1 to 1
    a_oo = a - a.real.min() - 1j * a.imag.min()
    return a_oo / np.abs(a_oo).max()


def get_2D_FFT(
        ar: np.ndarray,  # J: Accept the image as an array and use type hints.
        # name,
):
    # ℹ️ J: If you just made the function return the FT, it'll be easier to
    #       use this function. See the modified references of this function
    #       below.
    def calculate_2dft(input):
        ft = np.fft.ifftshift(input)
        ft = np.fft.fft2(ft)
        return np.fft.fftshift(ft)

    # ℹ️ J: You can use the ellipsis to select all channels
    image = ar[..., :3].mean(axis=2)

    ft = calculate_2dft(image)
    # ft_real_01 = normalize_complex_arr(ft)
    # ft_real_01 = normalize_complex_arr(np.clip(ft_real_01, 0, 0.015)

    # ft_real = np.real(normalize_complex_arr(ft))
    ft_real = np.absolute(ft)
    ft_real_01 = ft_real / np.max(ft_real)
    # ⚠️ J: We'll use a quantile to clip the image to get rid of outliers
    #       This is a more robust way of clipping the image
    # ℹ️ J: You can take a look at the histogram of the image to see if
    #       clipping is necessary
    ft_real_01 = np.clip(ft_real_01, 0, np.quantile(ft_real_01, 0.99))
    ft_log = np.log(ft_real_01 + 1e-6)
    return ft_log


def get_fft(
        filename,
        # ⚠️ J: Don't use folderdir as a name, it's not descriptive
        #       Also, you should usually make it relate to the project as other
        #       people may have different dev environments.
        save_dir=".",
        # ⚠️ J: Don't use magic numbers, use a variable
        #       Also use type hints and a default
        im_crop_size: int = 160,
        rot_angle: int = 90,
        # channel, # J: I don't get why this is used
):
    # ℹ️ J: I highly recommend to use pathlib.Path instead of a string.
    #       It has inbuilt functions that make it easier to work with paths
    #       such as creating directories, joining paths, etc. which you'll
    #       see below.
    save_dir = Path(save_dir)

    # ℹ️ J: If the directory doesn't exist, we'll create it
    save_dir.mkdir(exist_ok=True)

    # Load the TIFF image
    im = Image.open(filename)

    # ⚠️ J: I recommend to parameterize the bounds file name, I've left this
    #       as an exercise for you
    # Read Excel file containing boundary values
    bounds = "chestnut_nature_park_20201218_bounds.csv"

    # ⚠️ J: Use a more descriptive name over df
    df_bounds = pd.read_csv(bounds)

    # Extract boundary values
    # J: If you're not using the index, you can use _ instead of a name
    for _, row in df_bounds.iterrows():
        x1, x2, y1, y2 = row["x0"], row["x1"], row["y0"], row["y1"]
        name = row["name"]

        # Crop the image using boundary values
        cropped_image = im.crop((x1, y1, x2, y2))

        # J: Removed dependency on argument channel
        # saved_image = foldername + "/Chestnut_Dec_" + name + ".png"
        # cropped_image.save(saved_image)

        # ✅ J: Tuple unpacking is a good way to assign variables
        width, height = cropped_image.size

        # ⚠️ J: Don't use magic numbers, use a variable. This can help you
        #       change the crop size easily via a function argument
        if width >= im_crop_size and height >= im_crop_size:
            # Cropped square image size in px
            # Setting the points for cropped image (getting the middle 160x160 px crop)

            # ✅ J: This is fine and clean
            left = (width / 2) - (im_crop_size / 2)
            top = (height / 2) - (im_crop_size / 2)
            right = (width / 2) + (im_crop_size / 2)
            bottom = (height / 2) + (im_crop_size / 2)

            # Cropped image of above dimension
            # (It will not change original image)
            normalised_image = cropped_image.crop((left, top, right, bottom))

            # J: Removed dependency on argument channel
            # saved_image_2 = foldername + "/Chestnut_Dec_cropped_" + name + ".png"
            # normalised_image.save(saved_image_2)

            # to test the equivariance of the 2D FFTs
            # ⚠️ J: We can parameterize the rotation angle
            rotated_image = normalised_image.rotate(rot_angle)

            # J: Removed dependency on argument channel
            # saved_image_3 = (
            #     foldername + "/Chestnut_Dec_cropped_rotated_" + name + ".png"
            # )
            # rotated_image.save(saved_image_3)

            # ❗J: Do not save the image, just send the array as an argument
            #      We do this to avoid having to save the image to disk, which
            #      can be slow and dependent on the development environment
            # get_2D_FFT(saved_image_2, name)
            # get_2D_FFT(saved_image_3, name)

            plt.set_cmap("gray")

            # ℹ️ J: This is the Supporting Title, which is the title of the
            #       entire figure. The title of each subplot is the Title
            plt.suptitle(name)

            # ℹ️ J: By moving the plotting code here, can you see the benefit
            #       of just returning the FT from the function?
            #       By rule of thumb, functions should only do one thing, much
            #       like LEGO bricks. Rarely, we see complex LEGO bricks that do
            #       multiple things.
            plt.subplot(221)
            plt.title("Original")
            plt.imshow(normalised_image)
            plt.axis("off")

            plt.subplot(222)
            plt.title("FFT")
            plt.imshow(get_2D_FFT(np.asarray(normalised_image)))
            plt.axis("off")

            plt.subplot(223)
            plt.title(f"Rotated by {rot_angle} degrees")
            plt.imshow(rotated_image)
            plt.axis("off")

            plt.subplot(224)
            plt.title("FFT of rotated")
            plt.imshow(get_2D_FFT(np.asarray(rotated_image)))
            plt.axis("off")

            # ⚠️ J: You left a plt.pause(2), which made the program run
            #       really slow. Usually, we don't need to use it.
            # plt.pause(2)

            # ℹ️ J: We can use the save_dir variable here
            #       Though unconventional, Pathlib allows us to use the /
            #       operator to join paths. This is a lot cleaner than using
            #       os.path.join or string concatenation
            plt.savefig(save_dir / f"{name}.png")

            # ℹ️ J: MatPlotLib will retain the figure in memory, so we need to
            #       close it to prevent memory leaks.
            plt.close("all")


get_fft("chestnut_nature_park_20201218_result.tif", save_dir="images")
