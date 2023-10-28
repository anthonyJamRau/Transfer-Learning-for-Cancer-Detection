import cv2
import random
import string

def rotate_augmentation(img):
    opt = random.randint(1, 3)
    if opt == 1:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif opt == 2:
        img = cv2.rotate(img, cv2.ROTATE_180)
    else:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def contrast_brightness_augmentation(img, alpha_range: [float], beta_range: [float]):
    opt = random.randint(1, 3)
    alpha = random.uniform(alpha_range[0], alpha_range[1])
    beta = random.uniform(beta_range[0], beta_range[1])
    if opt == 1:
        #Change contrast of image
        img = cv2.convertScaleAbs(img, alpha=alpha)
    elif opt == 2:
        #Change brightness of image
        img = cv2.convertScaleAbs(img, beta=beta)
    else:
        #Change both
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img

def blur_augmentation(img):
    img = cv2.GaussianBlur(img, (5,5), 0)
    return img

def flip_augmentation(img):
    opt = random.randint(1, 3)
    if opt == 1:
        #flip image vertically
        img = cv2.flip(img, 0)
    elif opt == 2:
        #flip image horizontally
        img = cv2.flip(img, 1)
    else:
        #flip image horizontally and vertically
        img = cv2.flip(img, -1)
    return img

def augment_image(path_prefix: str, image_path: str, chance: float, dest_path: str):
    try:
        img = img = cv2.imread(path_prefix + image_path)
        if random.random() < chance:
            img = contrast_brightness_augmentation(img, [.8, 1.2], [-20, 20])
        if random.random() < chance:
            img = flip_augmentation(img)
        if random.random() < chance:
            img = blur_augmentation(img)
        uid = ''.join(random.choice(string.ascii_letters) for i in range(4))
        image_path = uid + '_' + image_path
        cv2.imwrite(dest_path + image_path, img)
    except:
        print('Could not access: ', image_path)

def augment_images(df, chance: float, path_prefix: str, dest_path: str, loops: int):
    for i in range(loops):
        df.map(lambda x: augment_image(path_prefix, x, chance, dest_path))